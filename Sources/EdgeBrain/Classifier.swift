import Foundation

/// A wrapper around a program that tracks input variables and output classes.
public struct Classifier: Codable, Sendable {

  public typealias Output = (prediction: Prediction, reachable: Set<Int>)

  internal typealias ReachableBitmap = [Bool]
  internal typealias BitmapOutput = (prediction: Prediction, reachable: ReachableBitmap)

  public struct InputID: Codable, Sendable {
    public var off: Int
    public var on: Int
  }

  public var program: Program
  public var inputIDs: [InputID] = []
  public var outputIDs: [Int] = []
  public var edgeWeight: Float = 1.0

  public init(inputCount: Int, labelCount: Int, hiddenCount: Int, edgeWeight: Float = 1.0) {
    program = Program()
    self.edgeWeight = edgeWeight
    for _ in 0..<inputCount {
      self.inputIDs.append(
        InputID(
          off: program.addNode(kind: .input),
          on: program.addNode(kind: .input)
        )
      )
    }
    for _ in 0..<labelCount {
      self.outputIDs.append(program.addNode(kind: .output))
    }
    for _ in 0..<hiddenCount {
      program.addNode(kind: .hidden)
    }
  }

  /// Compute class log probabilities for an input feature map.
  public func classify(features: [Bool]) -> Prediction {
    countsToPrediction(
      outCounts: program.run(
        withInputs: featuresToInputIDs(features: features)
      ).outputs
    )
  }

  /// Compute the loss on all of the data examples.
  public func loss(data: [[Bool]], labels: [Int], program: Program? = nil) -> Float {
    zip(run(data: data), labels).map {
      -$0.0.prediction.logProb(label: $0.1)
    }.reduce(0, +) / Float(data.count)
  }

  /// Compute the output maps and hidden sets for all of the data examples.
  public func run(data: [[Bool]], program: Program? = nil) -> [Output] {
    let program = program ?? self.program
    let count = data.count
    let agg = SyncArray<Program.Output?>(repeating: nil, count: count)

    let featuresToInputIDs = self.featuresToInputIDs
    DispatchQueue.global().sync(execute: {
      DispatchQueue.concurrentPerform(iterations: count) { i in
        let features = data[i]
        agg.set(program.run(withInputs: featuresToInputIDs(features)), at: i)
      }
    })

    return agg.value.map { out in
      (countsToPrediction(outCounts: out!.outputs), out!.reachable)
    }
  }

  /// Like run() but return reachable bitmaps.
  internal func runBitmap(data: [[Bool]], program: Program? = nil) -> [BitmapOutput] {
    let idCount = (program ?? self.program).maximumID + 1
    return run(data: data, program: program).map { (prediction, reachableSet) in
      var reachableArr = Array(repeating: false, count: idCount)
      for x in reachableSet {
        reachableArr[x] = true
      }
      return (prediction, reachableArr)
    }
  }

  /// Compute a sequence of greedily-selected mutations to minimize loss on the
  /// provided data.
  public mutating func greedilyMutatedForData(data: [[Bool]], labels: [Int])
    -> (classifier: Classifier, mutations: [Mutation])
  {
    let idToLabel = Dictionary(uniqueKeysWithValues: zip(outputIDs, outputIDs.indices))

    /// Mutate the outputs, though note that this will not preserve the
    /// order of the outputs.
    func mutate(outputs: [BitmapOutput], mutation: Mutation) -> [BitmapOutput] {
      let delta: Int
      let hiddenNode: Int
      let label: Int
      switch mutation {
      case .addEdge(let h, let edge):
        hiddenNode = h
        delta = 1
        label = idToLabel[edge.to]!
      case .removeEdge(let h, let edge):
        hiddenNode = h
        delta = -1
        label = idToLabel[edge.to]!
      }

      return outputs.map { (prediction, reachable) in
        var newCounts = prediction.counts
        if reachable[hiddenNode] {
          newCounts[label] += delta
        }
        return (Prediction(counts: newCounts, edgeWeight: edgeWeight), reachable)
      }
    }

    func evaluate(outputs: [BitmapOutput]) -> Float {
      let workerCount = min(outputs.count, ProcessInfo.processInfo.activeProcessorCount)
      let agg = SyncArray<Float>(
        repeating: 0,
        count: workerCount
      )
      DispatchQueue.global().sync {
        DispatchQueue.concurrentPerform(iterations: workerCount) { workerIdx in
          var result: Float = 0
          for i in stride(from: workerIdx, to: outputs.count, by: workerCount) {
            let pred = outputs[i].0
            let label = labels[i]
            result -= pred.logProb(label: label)
          }
          agg.set(result, at: workerIdx)
        }
      }
      return agg.value.reduce(0, +)
    }

    var curOutputs = runBitmap(data: data)
    var curLoss = evaluate(outputs: curOutputs)
    var result = self
    var appliedMutations: [Mutation] = []
    let sortedMutations = greedyHiddenMutations(outputs: curOutputs, labels: labels).sorted {
      $0.1 > $1.1
    }
    for (mutation, delta) in sortedMutations {
      if delta <= 0 {
        // This mutation didn't help the original program.
        // It might help now, but it seems unlikely, so we should save time
        // by skipping all remaining (even worse) mutations.
        break
      }
      let newOutputs = mutate(outputs: curOutputs, mutation: mutation)
      let newLoss = evaluate(outputs: newOutputs)
      if newLoss < curLoss {
        curOutputs = newOutputs
        curLoss = newLoss
        result.program.mutate(mutation)
        appliedMutations.append(mutation)
      }
    }
    return (classifier: result, mutations: appliedMutations)
  }

  /// Compute, for each edge between a hidden node and an output, the loss
  /// improvement from either adding or removing the edge from the hidden node.
  ///
  /// The improvement is positive when the loss would decrease, i.e. the
  /// log-prob would increase by adding this edge.
  private func greedyHiddenMutations(outputs: [BitmapOutput], labels: [Int]) -> [(
    Mutation, Float
  )] {
    let idCount = program.maximumID + 1
    let mapCount = outputIDs.count * idCount
    let hiddenIDs = program.hiddenIDs()

    @Sendable
    func mapIdx(label: Int, hidden: Int) -> Int {
      label * idCount + hidden
    }

    var hasEdgesMut = [Bool](repeating: false, count: mapCount)
    for node in program.nodes.values {
      if node.kind != .hidden {
        continue
      }
      for (label, id) in outputIDs.enumerated() {
        if node.edges.contains(Edge(from: node.id, to: id)) {
          hasEdgesMut[mapIdx(label: label, hidden: node.id)] = true
        }
      }
    }
    let hasEdges = hasEdgesMut

    let workerCount = min(outputs.count, ProcessInfo.processInfo.activeProcessorCount)
    let agg = SyncArray<[Float]>(
      repeating: [],
      count: workerCount
    )
    DispatchQueue.global().sync {
      DispatchQueue.concurrentPerform(iterations: workerCount) { workerIdx in
        var accumulator = [Float](repeating: 0, count: mapCount)
        for i in stride(from: workerIdx, to: outputs.count, by: workerCount) {
          let (pred, reachable) = outputs[i]
          let label = labels[i]
          let oldLoss = -pred.logProb(label: label)
          for (outputLabel, _) in outputIDs.enumerated() {
            let addChange =
              oldLoss + pred.logProb(label: label, withChange: 1, toLabel: outputLabel)
            let removeChange =
              oldLoss + pred.logProb(label: label, withChange: -1, toLabel: outputLabel)
            for hiddenID in hiddenIDs {
              if !reachable[hiddenID] {
                continue
              }
              let idx = mapIdx(label: outputLabel, hidden: hiddenID)
              let hasEdge = hasEdges[idx]
              let change = hasEdge ? removeChange : addChange
              accumulator[idx] += change
            }
          }
        }
        agg.set(accumulator, at: workerIdx)
      }
    }

    var results = [Float](repeating: 0, count: mapCount)
    for x in agg.value {
      for (i, v) in x.enumerated() {
        results[i] += v
      }
    }
    var mutations = [(Mutation, Float)]()
    for hiddenID in hiddenIDs {
      for (label, id) in outputIDs.enumerated() {
        let idx = mapIdx(label: label, hidden: hiddenID)
        let hasEdge = hasEdges[idx]
        let mut: Mutation =
          hasEdge
          ? .removeEdge(hiddenID, Edge(from: hiddenID, to: id))
          : .addEdge(hiddenID, Edge(from: hiddenID, to: id))
        mutations.append((mut, results[idx]))
      }
    }
    return mutations
  }

  /// Convert a feature vector to a collection of active input node Ds.
  public func featuresToInputIDs(features: [Bool]) -> [Int] {
    precondition(features.count == inputIDs.count)
    return zip(inputIDs, features).map { $0.1 ? $0.0.on : $0.0.off }
  }

  /// Map program outputs to a model prediction.
  public func countsToPrediction(outCounts: [Int: Int]) -> Prediction {
    Prediction(counts: outputIDs.map { outCounts[$0]! }, edgeWeight: edgeWeight)
  }

  /// A classification prediction, as described by counts for each label.
  ///
  /// The prediction probabilities are proportional to exp(count_i) for each
  /// label i.
  public struct Prediction: Codable, Sendable {
    public let counts: [Int]
    public let edgeWeight: Float

    public var logProbs: [Float] {
      let maxValue = counts.max()!
      var logitSum = Float(0)
      for x in counts {
        logitSum += exp(Float(x - maxValue))
      }
      let normalizer = log(logitSum)
      return counts.map { Float($0 - maxValue) - normalizer }
    }

    public init(counts: [Int], edgeWeight: Float) {
      self.counts = counts
      self.edgeWeight = edgeWeight
    }

    public func logProb(label: Int) -> Float {
      let maxValue = counts.max()!
      var logitSum = Float(0)
      for x in counts {
        logitSum += exp(Float(x - maxValue) * edgeWeight)
      }
      let normalizer = log(logitSum)
      return Float(counts[label] - maxValue) - normalizer
    }

    public func logProb(label: Int, withChange: Int, toLabel: Int) -> Float {
      let almostMaxValue = counts.max()!
      var logitSum = Float(0)
      for (i, var x) in counts.enumerated() {
        if i == toLabel {
          x += withChange
        }
        logitSum += exp(Float(x - almostMaxValue) * edgeWeight)
      }
      let normalizer = log(logitSum)
      let count = label == toLabel ? counts[label] + withChange : counts[label]
      return Float(count - almostMaxValue) - normalizer
    }
  }

}
