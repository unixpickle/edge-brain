import ArgumentParser
import DrawGraph
import EdgeBrain
import Foundation
import Honeycrisp
import MNIST

@main struct Main: AsyncParsableCommand {

  struct State: Codable {
    let model: Classifier
    let step: Int?
  }

  // Model config
  @Option(name: .long, help: "Number of hidden nodes.") var hiddenCount: Int = 1024
  @Option(name: .long, help: "Softmax weight of each output edge.") var edgeWeight: Float = 1.0

  // Initialization
  @Option(name: .long, help: "Examples for initial batch.") var initBatchSize: Int = 100
  @Option(name: .long, help: "Initial hidden reachability fraction.") var initReachableFrac:
    Double = 0.2
  @Option(name: .long, help: "Initial hidden reachability fraction.") var initHiddenGroups: Int = 1

  // Training config
  @Option(name: .long, help: "Batch size per step.") var batchSize: Int = 10000
  @Option(name: .long, help: "Number of mutations to try.") var mutationCount: Int = 5
  @Option(name: .long, help: "Changes per mutation.") var mutationSize: Int = 3
  @Option(name: .long, help: "Edge deletion probability.") var mutationDeleteProb: Double = 0.25
  @Option(name: .long, help: "Mutations to test at smaller batch size.")
  var preliminaryMutationCount: Int = 5
  @Option(name: .long, help: "Batch size for testing more mutations.")
  var preliminaryMutationBatchSize: Int = 100
  @Option(name: .long, help: "Mutation selection confidence bound")
  var mutationSelectionConfidence: Double? = nil

  @Flag(name: .long, help: "Evaluate the model with a linear classifier")
  var evaluateLinearLoss: Bool = false

  // Saving
  @Option(name: .shortAndLong, help: "Path to save train state.") var modelPath: String =
    "state.plist"
  @Option(name: .shortAndLong, help: "Path to save animation to.") var animationPath: String =
    "classifier_animation"
  @Option(name: .long, help: "Save interval.") var saveInterval: Int = 10

  mutating func run() async {
    print("Command:", CommandLine.arguments.joined(separator: " "))

    let selectionRule: Classifier.MutationSelectionRule =
      if let p = mutationSelectionConfidence {
        .confidence(p)
      } else {
        .mean
      }

    do {
      Backend.defaultBackend = try MPSBackend()

      print("loading dataset...")
      let dataset = try await MNISTDataset.download(toDir: "mnist_data")

      var model = Classifier(
        inputCount: 28 * 28,
        labelCount: 10,
        hiddenCount: hiddenCount,
        edgeWeight: edgeWeight
      )
      var step = 0

      if FileManager.default.fileExists(atPath: modelPath) {
        print("loading from checkpoint: \(modelPath) ...")
        let data = try Data(contentsOf: URL(fileURLWithPath: modelPath))
        let decoder = PropertyListDecoder()
        let state = try decoder.decode(State.self, from: data)
        model = state.model
        step = state.step ?? 0
      } else {
        print("initializing model...")
        let (imgs, _) = select(count: initBatchSize, fromImages: dataset.train)
        let mutationCount = model.randomlyInitialize(
          data: imgs,
          reachableFrac: initReachableFrac,
          hiddenGroups: initHiddenGroups
        )
        let equiv = model.equivalentHiddenNodes(data: imgs)
        print(
          " => initialized with \(mutationCount) insertions with \(equiv.count) unique hidden nodes"
        )
      }

      while true {
        let (batchInputs, batchTargets) = select(count: batchSize, fromImages: dataset.train)

        try renderAnimation(
          classifier: model,
          features: batchInputs[0],
          vertexRadius: 5.0,
          drawEdges: true,
          drawInputs: false,
          outputDir: URL(filePath: animationPath)
        )

        let (testIns, testLabels) = select(count: dataset.test.count, fromImages: dataset.test)
        let (testAcc, testLoss) = try await evaluate(
          model: model, inputs: testIns, targets: testLabels)
        let (acc, oldLoss) = try await evaluate(
          model: model, inputs: batchInputs, targets: batchTargets)

        var linearLoss: Float? = nil
        if evaluateLinearLoss {
          let linearClf = LinearClassifier(
            featureCount: model.program.hiddenIDs().count,
            outputCount: model.outputIDs.count
          )
          try await linearClf.fit(
            program: model.program,
            data: model.run(data: batchInputs).map { $0.reachable },
            labels: batchTargets
          )
          linearLoss = try await linearClf.loss(
            program: model.program,
            data: model.run(data: testIns).map { $0.reachable },
            labels: testLabels
          )
        }

        let preliminaryMutations = (0..<preliminaryMutationCount).map { _ in
          var newModel = model
          for _ in 0..<mutationSize {
            newModel.program.mutate(
              newModel.program.randomMutation(deleteProb: mutationDeleteProb)
            )
          }
          return newModel.greedilyMutatedForData(
            data: Array(batchInputs[..<preliminaryMutationBatchSize]),
            labels: Array(batchTargets[..<preliminaryMutationBatchSize]),
            selectionRule: selectionRule
          )
        }

        let keptMutations = preliminaryMutations.sorted(by: { $0.loss < $1.loss })[..<mutationCount]
        var mutatedPrograms = keptMutations.map { mutation in
          mutation.classifier.greedilyMutatedForData(
            data: batchInputs,
            labels: batchTargets,
            selectionRule: selectionRule
          )
        }

        // Include greedy model with no mutations.
        let greedyMutated = model.greedilyMutatedForData(
          data: batchInputs,
          labels: batchTargets,
          selectionRule: selectionRule
        )
        mutatedPrograms.append(greedyMutated)

        let losses = mutatedPrograms.map {
          $0.loss
        }
        let minLoss = losses.min()!
        if minLoss < oldLoss {
          let minIdx = losses.firstIndex(of: minLoss)!
          model = mutatedPrograms[minIdx].classifier
        }

        step += 1
        var logStr =
          "step \(step): loss=\(oldLoss) acc=\(acc) test_loss=\(testLoss) test_acc=\(testAcc) "
          + "greedy=\(greedyMutated.loss) greedy_count=\(greedyMutated.mutations.count) "
          + "min=\(minLoss) max=\(losses.max()!)"
        if let linearLoss = linearLoss {
          logStr += " linear_loss=\(linearLoss)"
        }
        print(logStr)

        if step % saveInterval == 0 {
          let state = State(
            model: model,
            step: step
          )
          let stateData = try PropertyListEncoder().encode(state)
          try stateData.write(to: URL(filePath: modelPath), options: .atomic)
        }
      }
    } catch { print("FATAL ERROR: \(error)") }
  }

  func select(count: Int, fromImages: [MNISTDataset.Image]) -> ([Bitmap], [Int]) {
    func sampleImage(_ img: MNISTDataset.Image) -> (Bitmap, Int) {
      (Bitmap(img.pixels.map { UInt8.random(in: 0..<0xff) < $0 }), img.label)
    }

    var sampledImages = [(Bitmap, Int)]()
    var remaining = count
    while remaining > fromImages.count {
      sampledImages.append(contentsOf: fromImages.map(sampleImage))
      remaining -= fromImages.count
    }
    sampledImages.append(contentsOf: fromImages.randomSample(count: remaining).map(sampleImage))
    return (sampledImages.map { $0.0 }, sampledImages.map { $0.1 })
  }

  func evaluate(
    model: Classifier,
    inputs: [Bitmap],
    targets: [Int]
  ) async throws -> (acc: Double, loss: Double) {
    let allPreds = model.predictions(data: inputs)
    var accSum = 0.0
    var lossSum = 0.0
    for (pred, target) in zip(allPreds, targets) {
      let logProbs = pred.logProbs
      let maxLogProb = logProbs.max()!

      if logProbs[target] == maxLogProb {
        // Deal with ties.
        let maxCount = logProbs.count { $0 == maxLogProb }
        accSum += 1 / Double(maxCount)
      }

      lossSum -= Double(logProbs[target])
    }
    return (accSum / Double(inputs.count), lossSum / Double(inputs.count))
  }

}
