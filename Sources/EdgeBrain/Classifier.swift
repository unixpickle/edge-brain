import Foundation

/// A wrapper around a program that tracks input variables and output classes.
public struct Classifier: Codable {

  public struct InputID: Codable {
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

  public func inputNodes(features: [Bool]) -> [Int] {
    zip(inputIDs, features).map { $0.1 ? $0.0.on : $0.0.off }
  }

  /// Compute, for each edge between a hidden node and an output, the loss
  /// improvement from either adding or removing the edge from the hidden node.
  ///
  /// The improvement is positive when the loss would decrease, i.e. the
  /// log-prob would increase by adding this edge.
  public func greedyHiddenMutations(data: [[Bool]], labels: [Int]) -> [(Mutation, Float)] {
    var deltas = [Mutation: Float]()
    let hiddenIDs = program.nodes.compactMap { $0.1.kind == .hidden ? $0.0 : nil }
    for (features, label) in zip(data, labels) {
      let (outCounts, reachable) = program.run(withInputs: inputNodes(features: features))
      let oldLoss = -logProbs(outCounts: outCounts)[label]
      for hidden in hiddenIDs {
        for outputID in outputIDs {
          var change: Float = 0
          let edge = Edge(from: hidden, to: outputID)
          let hasEdge = program.nodes[hidden]!.edges.contains(edge)
          let mutation: Mutation =
            if hasEdge {
              .removeEdge(hidden, edge)
            } else {
              .addEdge(hidden, edge)
            }
          if reachable.contains(hidden) {
            var newCounts = outCounts
            if !program.nodes[hidden]!.edges.contains(edge) {
              newCounts[outputID, default: 0] += 1
            } else {
              newCounts[outputID, default: 0] -= 1
            }
            let newLoss = -logProbs(outCounts: newCounts)[label]
            change = oldLoss - newLoss
          }
          deltas[mutation, default: 0.0] += change
        }
      }
    }
    return deltas.map { ($0.key, $0.value) }
  }

  /// Compute a sequence of mutations from greedyHiddenMutations() to attempt
  /// to minimize loss on the data.
  public mutating func greedilyMutatedForData(data: [[Bool]], labels: [Int]) -> Classifier {
    let mutations = greedyHiddenMutations(data: data, labels: labels).sorted { $0.1 > $1.1 }

    func evalLoss(_ p: Program) -> Float {
      var result: Float = 0
      for (features, label) in zip(data, labels) {
        let (outCounts, _) = p.run(withInputs: inputNodes(features: features))
        result -= logProbs(outCounts: outCounts)[label]
      }
      return result
    }

    var newProgram = self.program
    var curLoss = evalLoss(newProgram)
    for (mutation, _) in mutations {
      var nextProgram = newProgram
      nextProgram.mutate(mutation)
      let nextLoss = evalLoss(nextProgram)
      if nextLoss < curLoss {
        newProgram = nextProgram
        curLoss = nextLoss
      }
    }

    var result = self
    result.program = newProgram
    return result
  }

  private func logProbs(outCounts: [Int: Int]) -> [Float] {
    let outLogits = outputIDs.map { Float(outCounts[$0]!) }
    let logitMax = outLogits.max()!
    let unscaledProbs = outLogits.map { exp($0 - logitMax) }
    let normalizer = log(unscaledProbs.reduce(0, +))
    return outLogits.map { $0 - (logitMax + normalizer) }
  }

}
