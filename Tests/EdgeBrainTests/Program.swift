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

@Test
func testProgramRun() {
  let featureCount = 32
  let hiddenCount = 64
  var data = [Bitmap](repeating: Bitmap(count: featureCount), count: 100)
  for i in 0..<data.count {
    for j in 0..<featureCount {
      if Int.random(in: 0..<2) == 0 {
        data[i][j] = true
      }
    }
  }

  var program = Program()
  var inputIDs = [Int]()
  for _ in 0..<featureCount {
    inputIDs.append(program.addNode(kind: .input))
  }
  for _ in 0..<hiddenCount {
    program.addNode(kind: .hidden)
  }

  program =
    program.randomlyInitialized(
      data: data.map { $0.enumerated().compactMap { (i, x) in x ? inputIDs[i] : nil } },
      reachableFrac: 0.5
    ).0

  for datum in data {
    let inputs = datum.enumerated().compactMap { (i, x) in x ? inputIDs[i] : nil }
    let actual = program.run(withInputs: inputs)
    let expected = naiveRun(program: program, withInputs: inputs)
    #expect(actual.outputs == expected.outputs)
    #expect(actual.reachable == expected.reachable)
  }
}

private func naiveRun<C: Collection<Int>>(program: Program, withInputs: C) -> Program.Output {
  var active = Set(withInputs)
  var g = DiGraph(vertices: program.nodes.keys)
  for r in active {
    for edge in program.nodes[r]!.edges {
      g.insert(edge: edge)
    }
  }

  while true {
    let reachable = g.reachable(from: active)
    if reachable.count == active.count {
      break
    }
    for added in reachable.subtracting(active) {
      for edge in program.nodes[added]!.edges {
        g.insert(edge: edge)
      }
    }
    active = reachable
  }

  let idCount = program.maximumID + 1
  var reachableArr = Bitmap(count: idCount)
  for x in active {
    reachableArr[x] = true
  }

  return (
    outputs: Dictionary(
      uniqueKeysWithValues: program.nodes.compactMap { kv -> (Int, Int)? in
        if kv.value.kind != .output {
          return nil
        }
        return (kv.key, g.neighbors(to: kv.key).intersection(active).count)
      }
    ),
    reachable: reachableArr
  )
}
