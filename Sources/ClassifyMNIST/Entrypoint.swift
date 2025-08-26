import ArgumentParser
import DrawGraph
import EdgeBrain
import Foundation
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

  // Training config
  @Option(name: .long, help: "Batch size per step.") var batchSize: Int = 20
  @Option(name: .long, help: "Number of mutations to try.") var mutationCount: Int = 20
  @Option(name: .long, help: "Changes per mutation.") var mutationSize: Int = 3
  @Option(name: .long, help: "Changes per mutation.") var mutationDeleteProb: Double = 0.25

  // Saving
  @Option(name: .shortAndLong, help: "Path to save train state.") var modelPath: String =
    "state.plist"
  @Option(name: .shortAndLong, help: "Path to save animation to.") var animationPath: String =
    "classifier_animation"
  @Option(name: .long, help: "Save interval.") var saveInterval: Int = 10

  mutating func run() async {
    print("Command:", CommandLine.arguments.joined(separator: " "))

    do {
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
        let mutationCount = model.randomlyInitialize(data: imgs, reachableFrac: initReachableFrac)
        print(" => initialized with \(mutationCount) insertions")
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
        let testAcc = accuracy(model: model, inputs: testIns, targets: testLabels)
        let acc = accuracy(model: model, inputs: batchInputs, targets: batchTargets)
        let oldLoss = evaluateLoss(model: model, inputs: batchInputs, targets: batchTargets)

        var mutatedPrograms = (0..<mutationCount).map { _ in
          var newModel = model
          for _ in 0..<mutationSize {
            newModel.program.mutate(
              newModel.program.randomMutation(deleteProb: mutationDeleteProb))
          }
          newModel =
            newModel.greedilyMutatedForData(
              data: batchInputs,
              labels: batchTargets
            ).classifier
          return newModel
        }

        // Include greedy model with no mutations.
        let greedyMutated = model.greedilyMutatedForData(data: batchInputs, labels: batchTargets)
        mutatedPrograms.append(greedyMutated.classifier)

        let losses = mutatedPrograms.map {
          evaluateLoss(model: $0, inputs: batchInputs, targets: batchTargets)
        }
        let minLoss = losses.min()!
        if minLoss < oldLoss {
          let minIdx = losses.firstIndex(of: minLoss)!
          model = mutatedPrograms[minIdx]
        }

        step += 1
        print(
          "step \(step): loss=\(oldLoss) acc=\(acc) test_acc=\(testAcc) greedy=\(losses.last!) "
            + "greedy_count=\(greedyMutated.mutations.count) min=\(minLoss) max=\(losses.max()!)"
        )

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

  func select(count: Int, fromImages: [MNISTDataset.Image]) -> ([[Bool]], [Int]) {
    let inputsAndLabels = fromImages.randomSample(count: count).map { img in
      (img.pixels.map { UInt8.random(in: 0...0xff) < $0 }, img.label)
    }
    return (inputsAndLabels.map { $0.0 }, inputsAndLabels.map { $0.1 })
  }

  func evaluateLoss(model: Classifier, inputs: [[Bool]], targets: [Int]) -> Float {
    model.loss(data: inputs, labels: targets)
  }

  func accuracy(model: Classifier, inputs: [[Bool]], targets: [Int]) -> Double {
    let allPreds = model.run(data: inputs).map { $0.prediction }
    var accSum = 0.0
    for (pred, target) in zip(allPreds, targets) {
      let logProbs = pred.logProbs
      let maxLogProb = logProbs.max()!

      if logProbs[target] == maxLogProb {
        // Deal with ties.
        let maxCount = logProbs.count { $0 == maxLogProb }
        accSum += 1 / Double(maxCount)
      }
    }
    return accSum / Double(inputs.count)
  }

}
