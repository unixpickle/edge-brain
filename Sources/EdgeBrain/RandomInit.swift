extension Program {

  /// Randomly apply mutations to approximately cover a given fraction of
  /// reachable hidden nodes.
  ///
  /// Also returns the number of applied edge insertions.
  public func randomlyInitialized<C: Collection<Int>>(
    data: [C],
    reachableFrac: Double,
    hiddenGroups: Int = 1
  ) -> (
    Program, Int
  ) {
    var allMutations: [Mutation] = []

    let hiddenIDs = hiddenIDs()
    let hiddenToGroup = Dictionary(
      uniqueKeysWithValues: zip(hiddenIDs, hiddenIDs.indices.map { $0 % hiddenGroups })
    )

    func mutationFilter(_: Int, edge: Edge<Int>) -> Bool {
      if let g1 = hiddenToGroup[edge.from], let g2 = hiddenToGroup[edge.to], g1 != g2 {
        return false
      }
      return true
    }

    // Fill up end until it has too much coverage
    var end = self
    while end.reachableHiddenFraction(data: data) < reachableFrac {
      let addCount = max(1, allMutations.count)
      for _ in 0..<addCount {
        guard let (v, e) = end.randomAddition(destKind: [.hidden], filter: mutationFilter) else {
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
    let hiddenIDs = Set(nodes.values.compactMap { $0.kind == .hidden ? $0.id : nil })
    for x in data {
      let hCount = run(withInputs: x).reachable.enumerated().count { (id, present) in
        present && hiddenIDs.contains(id)
      }
      total += Double(hCount)
    }
    let hiddenCount = nodes.values.count { $0.kind == .hidden }
    return total / (Double(data.count) * Double(hiddenCount))
  }

}

extension Classifier {
  /// Apply the program's randomlyInitialized() method and return the number of
  /// mutations applied.
  public mutating func randomlyInitialize(
    data: [Bitmap],
    reachableFrac: Double,
    hiddenGroups: Int = 1
  ) -> Int {
    let (newProgram, mutationCount) = program.randomlyInitialized(
      data: data.map(featuresToInputIDs),
      reachableFrac: reachableFrac,
      hiddenGroups: hiddenGroups
    )
    program = newProgram
    return mutationCount
  }
}
