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
    let outLogits = outputIDs.map { Float(outCounts[$0]!) }
    let logitMax = outLogits.max()!
    let unscaledProbs = outLogits.map { exp($0 - logitMax) }
    let normalizer = log(unscaledProbs.reduce(0, +))
    return outLogits.map { $0 - (logitMax + normalizer) }
  }

  public func inputNodes(features: [Bool]) -> [Int] {
    zip(inputIDs, features).map { $0.1 ? $0.0.on : $0.0.off }
  }

}
