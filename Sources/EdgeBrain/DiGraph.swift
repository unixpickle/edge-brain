public struct Edge<V: Hashable>: Hashable {
  public let from: V
  public let to: V

  public init(from: V, to: V) {
    self.from = from
    self.to = to
  }
}

/// A directed graph with constant-time insertion of edges and vertices.
public struct DiGraph<V: Hashable> {

  private var vertices: Set<V>
  private var outgoing: [V: Set<V>] = [:]
  private var incoming: [V: Set<V>] = [:]

  public var edges: Set<Edge<V>> {
    Set(
      outgoing.flatMap { (source, dests) in
        dests.map { dest in Edge(from: source, to: dest) }
      }
    )
  }

  public init() {
    self.vertices = []
  }

  public init<C: Collection<V>>(vertices: C) {
    self.vertices = Set(vertices)
  }

  public init<C: Collection<V>, E: Collection<Edge<V>>>(vertices: C, edges: E) {
    self.vertices = Set(vertices)
    for e in edges {
      insert(edge: e)
    }
  }

  public mutating func insert(edge: Edge<V>) {
    insertEdge(from: edge.from, to: edge.to)
  }

  public mutating func insertEdge(from: V, to: V) {
    assert(vertices.contains(from) && vertices.contains(to))
    outgoing[from, default: []].insert(to)
    incoming[to, default: []].insert(from)
  }

  public mutating func insert(vertex: V) {
    vertices.insert(vertex)
  }

  public mutating func remove(edge: Edge<V>) {
    removeEdge(from: edge.from, to: edge.to)
  }

  public mutating func removeEdge(from: V, to: V) {
    assert(vertices.contains(from) && vertices.contains(to))
    outgoing[from]?.remove(to)
    incoming[to]?.remove(from)
  }

  public mutating func remove(vertex: V) {
    for neighbor in outgoing[vertex, default: []] {
      incoming[neighbor]!.remove(vertex)
    }
    for neighbor in incoming[vertex, default: []] {
      outgoing[neighbor]!.remove(vertex)
    }
    incoming.removeValue(forKey: vertex)
    outgoing.removeValue(forKey: vertex)
    vertices.remove(vertex)
  }

  public func contains(vertex: V) -> Bool {
    vertices.contains(vertex)
  }

  public func contains(edge: Edge<V>) -> Bool {
    containsEdge(from: edge.from, to: edge.to)
  }

  public func containsEdge(from: V, to: V) -> Bool {
    outgoing[from]?.contains(to) ?? false
  }

  public func neighbors(from: V) -> Set<V> {
    outgoing[from, default: []]
  }

  public func neighbors(to: V) -> Set<V> {
    incoming[to, default: []]
  }

  /// Find all vertices reachable from a starting point.
  public func reachable<C: Collection<V>>(from: C) -> Set<V> {
    var queue = Array(from)
    var seen = Set(from)
    while let item = queue.popLast() {
      for neighbor in neighbors(from: item) {
        if seen.insert(neighbor).inserted {
          queue.append(neighbor)
        }
      }
    }
    return seen
  }

}
