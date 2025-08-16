import EdgeBrain
import Testing

@Test
func testProgramXor() {
  var prog = Program()
  let a0 = prog.addNode(kind: .input)
  let a1 = prog.addNode(kind: .input)
  let b0 = prog.addNode(kind: .input)
  let b1 = prog.addNode(kind: .input)
  let outNode = prog.addNode(kind: .output)
  prog.nodes[a0]!.edges.insert(.init(from: b1, to: outNode))
  prog.nodes[b0]!.edges.insert(.init(from: a1, to: outNode))

  for a in [0, 1] {
    for b in [0, 1] {
      let inputs = [a == 1 ? a1 : a0, b == 1 ? b1 : b0]
      let expectedOut = a ^ b
      let actualOut = prog.run(withInputs: inputs).outputs[outNode]
      #expect(expectedOut == actualOut)
    }
  }
}
