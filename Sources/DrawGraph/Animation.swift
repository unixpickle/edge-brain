import EdgeBrain
import Foundation

public func renderAnimation(
  classifier: Classifier,
  features: [Bool],
  vertexRadius: CGFloat,
  drawEdges: Bool,
  outputDir: URL
) throws {
  try renderAnimation(
    program: classifier.program,
    withInputs: classifier.featuresToInputIDs(features: features),
    vertexRadius: vertexRadius,
    drawEdges: drawEdges,
    outputDir: outputDir
  )
}

public func renderAnimation<C: Collection<Int>>(
  program: Program,
  withInputs: C,
  vertexRadius: CGFloat,
  drawEdges: Bool,
  outputDir: URL
) throws {
  if !FileManager.default.fileExists(atPath: outputDir.path()) {
    try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
  }
  let inputIDs = program.nodes.values.compactMap { $0.kind == .input ? $0.id : nil }
  let outputIDs = program.nodes.values.compactMap { $0.kind == .output ? $0.id : nil }
  let hiddenIDs = program.nodes.values.compactMap { $0.kind == .hidden ? $0.id : nil }

  let hiddenSquareSize = Int(ceil(sqrt(Double(hiddenIDs.count))))
  let verticesTall = max(inputIDs.count, outputIDs.count, hiddenSquareSize)
  let hiddenWidth = Double(hiddenSquareSize) * vertexRadius * 4
  let canvasHeight = CGFloat(verticesTall) * vertexRadius * 4  // Each vertex gets an extra vertex of spacing
  let canvasWidth = vertexRadius * 4 * 2 + hiddenWidth

  var coordinates = [Int: CGPoint]()

  let inputIDSpace = canvasHeight / Double(inputIDs.count)
  for (i, inputID) in inputIDs.enumerated() {
    coordinates[inputID] = CGPoint(x: vertexRadius * 2, y: inputIDSpace * (Double(i) + 0.5))
  }

  let outputIDSpace = canvasHeight / Double(outputIDs.count)
  for (i, outputID) in outputIDs.enumerated() {
    coordinates[outputID] = CGPoint(
      x: canvasWidth - vertexRadius * 2, y: outputIDSpace * (Double(i) + 0.5))
  }

  for hiddenID in hiddenIDs {
    coordinates[hiddenID] = CGPoint(
      x: vertexRadius * 4 + Double.random(in: 0..<hiddenWidth),
      y: Double.random(in: 0..<(canvasHeight - vertexRadius * 2)) + vertexRadius
    )
  }

  var active = Set(withInputs)
  var g = DiGraph(vertices: program.nodes.keys)
  for r in active {
    for edge in program.nodes[r]!.edges {
      g.insert(edge: edge)
    }
  }

  var frameIdx = 0
  func renderFrame() throws {
    try renderGraphImage(
      graph: g,
      vertexPositions: coordinates,
      highlightedReachable: active,
      canvasSize: CGSize(width: canvasWidth, height: canvasHeight),
      vertexRadius: vertexRadius,
      drawEdges: drawEdges,
      outputURL: outputDir.appending(component: "frame_\(frameIdx).png")
    )
    frameIdx += 1
  }

  try renderFrame()
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
    try renderFrame()
  }
}
