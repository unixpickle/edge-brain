extension Program {

  /// Randomly apply mutations to approximately cover a given fraction of
  /// reachable hidden nodes.
  ///
  /// Also returns the number of applied edge insertions.
  public func randomlyInitialized<C: Collection<Int>>(data: [C], reachableFrac: Double) -> (
    Program, Int
  ) {
    var allMutations: [Mutation] = []

    // Fill up end until it has too much coverage
    var end = self
    while end.reachableHiddenFraction(data: data) < reachableFrac {
      let addCount = max(1, allMutations.count)
      for _ in 0..<addCount {
        guard let (v, e) = end.randomAddition(destKind: [.hidden]) else {
          break
        }
        let m = Mutation.addEdge(v, e)
        allMutations.append(m)
        end.mutate(m)
      }
    }

    var start = self
    var startIdx = 0
    var endIdx = allMutations.count

    while startIdx + 1 < endIdx {
      let midIdx = (startIdx + endIdx) / 2
      var mid = start
      for mutation in allMutations[startIdx..<midIdx] {
        mid.mutate(mutation)
      }
      if mid.reachableHiddenFraction(data: data) < reachableFrac {
        startIdx = midIdx
        start = mid
      } else {
        endIdx = midIdx
      }
    }

    return (start, startIdx)
  }

  /// Compute the fraction of hidden nodes reachable on average over the
  /// dataset.
  public func reachableHiddenFraction<C: Collection<Int>>(data: [C]) -> Double {
    var total = 0.0
    for x in data {
      let hCount = run(withInputs: x).reachable.count { nodes[$0]!.kind == .hidden }
      total += Double(hCount)
    }
    let hiddenCount = nodes.values.count { $0.kind == .hidden }
    return total / (Double(data.count) * Double(hiddenCount))
  }

}

extension Classifier {
  /// Apply the program's randomlyInitialized() method and return the number of
  /// mutations applied.
  public mutating func randomlyInitialize(data: [[Bool]], reachableFrac: Double) -> Int {
    let (newProgram, mutationCount) = program.randomlyInitialized(
      data: data.map(featuresToInputIDs),
      reachableFrac: reachableFrac
    )
    program = newProgram
    return mutationCount
  }
}
