import EdgeBrain
import Testing

@Test
func testDiGraphOps() {
  var g = DiGraph(vertices: 0..<4)
  g.insertEdge(from: 0, to: 1)
  g.insertEdge(from: 1, to: 2)
  g.insertEdge(from: 2, to: 3)
  #expect(g.containsEdge(from: 0, to: 1))
  #expect(!g.containsEdge(from: 0, to: 2))
  #expect(!g.containsEdge(from: 1, to: 0))
  g.insertEdge(from: 1, to: 0)
  #expect(g.containsEdge(from: 1, to: 0))
  g.removeEdge(from: 0, to: 1)
  #expect(!g.containsEdge(from: 0, to: 1))
  #expect(g.containsEdge(from: 1, to: 0))

  #expect(g.containsEdge(from: 1, to: 2))
  #expect(g.containsEdge(from: 2, to: 3))
  #expect(g.neighbors(from: 2) == [3])
  #expect(g.neighbors(from: 1) == [0, 2])
  g.remove(vertex: 2)
  #expect(!g.containsEdge(from: 1, to: 2))
  #expect(!g.containsEdge(from: 2, to: 3))
  #expect(g.neighbors(from: 2) == [])
  #expect(g.neighbors(from: 1) == [0])
}

@Test
func testDiGraphReachable() {
  let g = DiGraph(
    vertices: 0..<7,
    edges: [
      .init(from: 0, to: 1), .init(from: 1, to: 2), .init(from: 2, to: 1), .init(from: 0, to: 3),
      .init(from: 6, to: 5), .init(from: 5, to: 2),
    ])
  #expect(g.reachable(from: [0]) == [0, 1, 2, 3])
  #expect(g.reachable(from: [1]) == [1, 2])
  #expect(g.reachable(from: [2]) == [1, 2])
  #expect(g.reachable(from: [3]) == [3])
  #expect(g.reachable(from: [2, 3]) == [1, 2, 3])
  #expect(g.reachable(from: [1, 2, 3]) == [1, 2, 3])
  #expect(g.reachable(from: [0, 1]) == [0, 1, 2, 3])
  #expect(g.reachable(from: [5]) == [5, 1, 2])
  #expect(g.reachable(from: [6]) == [6, 5, 1, 2])
  #expect(g.reachable(from: [6, 0]) == [6, 5, 0, 1, 2, 3])
}
