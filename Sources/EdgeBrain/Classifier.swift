import Foundation

/// A wrapper around a program that tracks input variables and output classes.
public struct Classifier: Codable, Sendable {

  public struct InputID: Codable, Sendable {
    public var off: Int
    public var on: Int
  }

  public var program: Program
  public var inputIDs: [InputID] = []
  public var outputIDs: [Int] = []

  public init(inputCount: Int, labelCount: Int, hiddenCount: Int) {
    program = Program()
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
  public func classify(features: [Bool]) -> [Float] {
    precondition(features.count == inputIDs.count)
    let outCounts = program.run(withInputs: inputNodes(features: features)).outputs
    return logProbs(outCounts: outCounts)
  }

  /// Compute the loss on all of the data examples.
  public func loss(data: [[Bool]], labels: [Int], program: Program? = nil) -> Float {
    let program = program ?? self.program
    let count = data.count
    let agg = SyncSum()

    let logProbs = self.logProbs
    let inputNodes = self.inputNodes

    DispatchQueue.global().sync(execute: {
      DispatchQueue.concurrentPerform(iterations: count) { i in
        let features = data[i]
        let label = labels[i]
        let (outCounts, _) = program.run(withInputs: inputNodes(features))
        let logProb = logProbs(outCounts)[label]
        agg.add(-logProb)
      }
    })

    return agg.value / Float(data.count)
  }

  /// Compute the output maps and hidden sets for all of the data examples.
  public func run(data: [[Bool]], program: Program? = nil) -> [Program.Output] {
    let program = program ?? self.program
    let count = data.count
    let agg = SyncArray<Program.Output?>(repeating: nil, count: count)

    let inputNodes = self.inputNodes
    DispatchQueue.global().sync(execute: {
      DispatchQueue.concurrentPerform(iterations: count) { i in
        let features = data[i]
        agg.set(program.run(withInputs: inputNodes(features)), at: i)
      }
    })

    return agg.value.map { $0! }
  }

  /// Compute, for each edge between a hidden node and an output, the loss
  /// improvement from either adding or removing the edge from the hidden node.
  ///
  /// The improvement is positive when the loss would decrease, i.e. the
  /// log-prob would increase by adding this edge.
  public func greedyHiddenMutations(data: [[Bool]], labels: [Int]) -> [(Mutation, Float)] {
    let hiddenIDs = program.nodes.compactMap { $0.1.kind == .hidden ? $0.0 : nil }
    let outputs = run(data: data)

    let agg = SyncArray<[Mutation: Float]>(repeating: [:], count: data.count)
    DispatchQueue.global().sync {
      DispatchQueue.concurrentPerform(iterations: data.count) { i in
        var deltas = [Mutation: Float]()
        let (outCounts, reachable) = outputs[i]
        let label = labels[i]
        let oldLoss = -logProbs(outCounts: outCounts)[label]
        for hidden in hiddenIDs {
          for outputID in outputIDs {
            if reachable.contains(hidden) {
              var change: Float = 0
              let edge = Edge(from: hidden, to: outputID)
              let hasEdge = program.nodes[hidden]!.edges.contains(edge)
              let mutation: Mutation =
                if hasEdge {
                  .removeEdge(hidden, edge)
                } else {
                  .addEdge(hidden, edge)
                }
              var newCounts = outCounts
              if !program.nodes[hidden]!.edges.contains(edge) {
                newCounts[outputID, default: 0] += 1
              } else {
                newCounts[outputID, default: 0] -= 1
              }
              let newLoss = -logProbs(outCounts: newCounts)[label]
              change = oldLoss - newLoss
              deltas[mutation, default: 0.0] += change
            }
          }
        }
        agg.set(deltas, at: i)
      }
    }

    var result = [Mutation: Float]()
    for mapping in agg.value {
      for (k, v) in mapping {
        result[k, default: 0] += v
      }
    }
    return result.map { ($0.key, $0.value) }
  }

  /// Compute a sequence of mutations from greedyHiddenMutations() to attempt
  /// to minimize loss on the data.
  public mutating func greedilyMutatedForData(data: [[Bool]], labels: [Int])
    -> (classifier: Classifier, mutations: [Mutation])
  {
    func mutateOutput(_ out: Program.Output, _ mutation: Mutation) -> Program.Output {
      var newOut = out
      switch mutation {
      case .addEdge(let hiddenNode, let edge):
        if newOut.reachable.contains(hiddenNode) {
          newOut.outputs[edge.to, default: 0] += 1
        }
      case .removeEdge(let hiddenNode, let edge):
        if newOut.reachable.contains(hiddenNode) {
          newOut.outputs[edge.to, default: 0] -= 1
        }
      }
      return newOut
    }

    func evaluateOutputs(_ out: [Program.Output]) -> Float {
      var result: Float = 0
      for ((counts, _), label) in zip(out, labels) {
        result -= logProbs(outCounts: counts)[label]
      }
      return result
    }

    let allMutations = greedyHiddenMutations(data: data, labels: labels).sorted { $0.1 > $1.1 }

    var curOutputs = run(data: data)
    var curLoss = evaluateOutputs(curOutputs)
    var result = self
    var appliedMutations: [Mutation] = []
    for (mutation, delta) in allMutations {
      if delta <= 0 {
        // This mutation didn't help the original program.
        // It might help now, but it seems unlikely, so we should save time
        // by skipping all remaining (even worse) mutations.
        break
      }
      let newOutputs = curOutputs.map { mutateOutput($0, mutation) }
      let newLoss = evaluateOutputs(newOutputs)
      if newLoss < curLoss {
        curOutputs = newOutputs
        curLoss = newLoss
        result.program.mutate(mutation)
        appliedMutations.append(mutation)
      }
    }
    return (classifier: result, mutations: appliedMutations)
  }

  public func inputNodes(features: [Bool]) -> [Int] {
    zip(inputIDs, features).map { $0.1 ? $0.0.on : $0.0.off }
  }

  /// Map program outputs to log probabilities.
  public func logProbs(outCounts: [Int: Int]) -> [Float] {
    let outLogits = outputIDs.map { Float(outCounts[$0]!) }
    let logitMax = outLogits.max()!
    let unscaledProbs = outLogits.map { exp($0 - logitMax) }
    let normalizer = log(unscaledProbs.reduce(0, +))
    return outLogits.map { $0 - (logitMax + normalizer) }
  }

}
