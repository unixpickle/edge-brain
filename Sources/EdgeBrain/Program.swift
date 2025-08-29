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

  public typealias Output = (outputs: [Int: Int], reachable: Bitmap)

  public var nodes: [Int: Node] = [:]
  private var idCounter: Int = 0

  public var maximumID: Int {
    idCounter - 1
  }

  public init() {
  }

  func hiddenIDs() -> [Int] {
    nodes.values.compactMap { $0.kind == .hidden ? $0.id : nil }.sorted()
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
    var insertQueue = active.flatMap { nodes[$0]!.edges }
    var g = DiGraph(vertices: nodes.keys)

    while let insertEdge = insertQueue.popLast() {
      let fromActive = active.contains(insertEdge.from)
      let toActive = active.contains(insertEdge.to)
      let toOutput = nodes[insertEdge.to]!.kind == .output
      if toOutput || (!fromActive && !toActive) {
        g.insert(edge: insertEdge)
      }
      if fromActive && !toActive {
        var queue = [insertEdge.to]
        active.insert(insertEdge.to)
        while let newActive = queue.popLast() {
          for edge in nodes[newActive]!.edges {
            insertQueue.append(edge)
          }
          for n in g.neighbors(from: newActive) {
            if active.insert(n).inserted {
              queue.append(n)
            }
          }
        }
      }
    }

    let idCount = maximumID + 1
    var reachableArr = Bitmap(count: idCount)
    for x in active {
      reachableArr[x] = true
    }

    return (
      outputs: Dictionary(
        uniqueKeysWithValues: nodes.compactMap { (k, v) -> (Int, Int)? in
          if v.kind != .output {
            return nil
          }
          return (k, g.neighbors(to: k).intersection(active).count)
        }
      ),
      reachable: reachableArr
    )
  }

}
