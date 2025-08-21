public struct Node: Codable, Sendable {
  public enum Kind: Codable, Sendable {
    case input
    case hidden
    case output
  }

  public let id: Int
  public let kind: Kind
  public var edges: Set<Edge<Int>> = []
}

public struct Program: Codable, Sendable {

  public typealias Output = (outputs: [Int: Int], reachable: Set<Int>)

  public var nodes: [Int: Node] = [:]
  private var idCounter: Int = 0

  public init() {
  }

  @discardableResult
  public mutating func addNode(kind: Node.Kind) -> Int {
    let newID = idCounter
    idCounter += 1
    nodes[newID] = Node(id: newID, kind: kind)
    return newID
  }

  public mutating func addRandomEdge() -> Bool {
    let sources = nodes.values.compactMap { $0.kind == .output ? nil : $0.id }.shuffled()
    let dests = nodes.values.compactMap { $0.kind == .input ? nil : $0.id }.shuffled()

    for var v in nodes.values.shuffled() {
      var found = false
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
          if !v.edges.contains(edge) {
            found = true
            v.edges.insert(edge)
          }
        }
        if found {
          break
        }
      }
      if found {
        nodes[v.id] = v
        return true
      }
    }
    return false
  }

  /// Run the program for a set of active inputs.
  ///
  /// The outputs mapping indicates, for each node ID, the number of edges to
  /// it from within the reachable set.
  ///
  /// Also returns the entire reachable set.
  public func run<C: Collection<Int>>(withInputs: C) -> Output {
    var active = Set(withInputs)
    var g = DiGraph(vertices: nodes.keys)
    for r in active {
      for edge in nodes[r]!.edges {
        g.insert(edge: edge)
      }
    }

    while true {
      let reachable = g.reachable(from: active)
      if reachable.count == active.count {
        break
      }
      for added in reachable.subtracting(active) {
        for edge in nodes[added]!.edges {
          g.insert(edge: edge)
        }
      }
      active = reachable
    }

    return (
      outputs: Dictionary(
        uniqueKeysWithValues: nodes.compactMap { kv -> (Int, Int)? in
          if kv.value.kind != .output {
            return nil
          }
          return (kv.key, g.neighbors(to: kv.key).intersection(active).count)
        }
      ),
      reachable: active
    )
  }

}
