extension Program {

  /// Randomly apply mutations to approximately cover a given fraction of
  /// reachable hidden nodes.
  public func randomlyInitialized(data: [Set<Int>], reachableFrac: Double) -> Program {
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
      if end.reachableHiddenFraction(data: data) < reachableFrac {
        startIdx = midIdx
        start = mid
      } else {
        endIdx = midIdx
      }
    }

    return start
  }

  /// Compute the fraction of hidden nodes reachable on average over the
  /// dataset.
  public func reachableHiddenFraction(data: [Set<Int>]) -> Double {
    var total = 0.0
    for x in data {
      let hCount = run(withInputs: x).reachable.count { nodes[$0]!.kind == .hidden }
      total += Double(hCount)
    }
    let hiddenCount = nodes.values.count { $0.kind == .hidden }
    return total / (Double(data.count) * Double(hiddenCount))
  }

}
