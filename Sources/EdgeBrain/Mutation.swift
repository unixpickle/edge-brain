public enum Mutation: Hashable, Codable, Sendable {
  case addEdge(Int, Edge<Int>)
  case removeEdge(Int, Edge<Int>)
}

extension Program {

  /// If any edge can be added to any node, select a random addition.
  public func randomAddition(
    nodeKind: Set<Node.Kind> = [.input, .hidden],
    sourceKind: Set<Node.Kind> = [.input, .hidden],
    destKind: Set<Node.Kind> = [.hidden, .output],
    filter: ((Int, Edge<Int>) -> Bool)? = nil
  ) -> (Int, Edge<Int>)? {
    let sources = nodes.values.compactMap { sourceKind.contains($0.kind) ? $0.id : nil }.shuffled()
    let dests = nodes.values.compactMap { destKind.contains($0.kind) ? $0.id : nil }.shuffled()

    for v in nodes.values.shuffled() {
      if !nodeKind.contains(v.kind) {
        continue
      }
      for someSource in sources {
        if someSource == v.id {
          continue
        }
        for someDest in dests {
          if someDest == v.id {
            continue
          }
          if someSource == someDest {
            continue
          }
          let edge = Edge(from: someSource, to: someDest)
          if let f = filter, !f(v.id, edge) {
            continue
          }
          if !v.edges.contains(edge) {
            return (v.id, edge)
          }
        }
      }
    }
    return nil
  }

  public func randomDeletion() -> (Int, Edge<Int>)? {
    nodes.values.flatMap { node in node.edges.map { e in (node.id, e) } }.randomElement()
  }

  public func randomMutation(deleteProb: Double = 0.25) -> Mutation {
    if Double.random(in: 0..<1) > deleteProb {
      if let (v, e) = randomAddition() {
        return .addEdge(v, e)
      }
    }
    if let (v, e) = randomDeletion() {
      return .removeEdge(v, e)
    }
    guard let (v, e) = randomAddition() else {
      fatalError("cannot add or delete an edge; not enough nodes")
    }
    return .addEdge(v, e)
  }

  public mutating func mutate(_ mutation: Mutation) {
    switch mutation {
    case .addEdge(let v, let e):
      nodes[v]!.edges.insert(e)
    case .removeEdge(let v, let e):
      nodes[v]!.edges.remove(e)
    }
  }

}
