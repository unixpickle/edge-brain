extension Classifier {
  /// Get equivalence groups of hidden nodes under the model.
  public func equivalentHiddenNodes(data: [Bitmap], program: Program? = nil) -> [Set<Int>] {
    let outputs = run(data: data, program: program)
    let transposed = transposeBitmaps(bitmaps: outputs.map { $0.reachable })
    var result = [Bitmap: Set<Int>]()
    for (i, x) in transposed.enumerated() {
      if (program ?? self.program).nodes[i]!.kind == .hidden {
        result[x, default: []].insert(i)
      }
    }
    return Array(result.values)
  }
}

private func transposeBitmaps(bitmaps: [Bitmap]) -> [Bitmap] {
  let featureCount = bitmaps[0].count
  let exampleCount = bitmaps.count
  var result = [Bitmap](repeating: Bitmap(count: exampleCount), count: featureCount)

  let featureChunk = 32
  let exampleChunk = 32
  for exampleStart in stride(from: 0, to: exampleCount, by: exampleChunk) {
    for featureStart in stride(from: 0, to: featureCount, by: featureChunk) {
      for example in exampleStart..<min(exampleCount, exampleStart + exampleChunk) {
        for feature in featureStart..<min(featureCount, featureStart + featureChunk) {
          result[feature][example] = bitmaps[example][feature]
        }
      }
    }
  }

  return result
}
