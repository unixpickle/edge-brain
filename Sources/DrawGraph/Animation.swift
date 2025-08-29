import EdgeBrain
import Foundation

public func renderAnimation(
  classifier: Classifier,
  features: Bitmap,
  vertexRadius: CGFloat,
  drawEdges: Bool = true,
  drawInputs: Bool = true,
  outputDir: URL
) throws {
  try renderAnimation(
    program: classifier.program,
    withInputs: classifier.featuresToInputIDs(features: features),
    vertexRadius: vertexRadius,
    drawEdges: drawEdges,
    drawInputs: drawInputs,
    outputDir: outputDir
  )
}

public func renderAnimation<C: Collection<Int>>(
  program: Program,
  withInputs: C,
  vertexRadius: CGFloat,
  drawEdges: Bool = true,
  drawInputs: Bool = true,
  outputDir: URL
) throws {
  if !FileManager.default.fileExists(atPath: outputDir.path()) {
    try FileManager.default.createDirectory(at: outputDir, withIntermediateDirectories: true)
  }
  let inputIDs = program.nodes.values.compactMap { $0.kind == .input ? $0.id : nil }
  let outputIDs = program.nodes.values.compactMap { $0.kind == .output ? $0.id : nil }
  let hiddenIDs = program.nodes.values.compactMap { $0.kind == .hidden ? $0.id : nil }

  let hiddenSquareSize = Int(ceil(sqrt(Double(hiddenIDs.count))))
  let verticesTall = max(drawInputs ? inputIDs.count : 0, outputIDs.count, hiddenSquareSize)
  let hiddenWidth = Double(hiddenSquareSize) * vertexRadius * 4
  let inputsWidth = drawInputs ? vertexRadius * 4 : 0
  let outputsWidth = vertexRadius * 4
  let canvasHeight = CGFloat(verticesTall) * vertexRadius * 4
  let canvasWidth = outputsWidth + inputsWidth + hiddenWidth

  var coordinates = [Int: CGPoint]()

  if drawInputs {
    let inputIDSpace = canvasHeight / Double(inputIDs.count)
    for (i, inputID) in inputIDs.enumerated() {
      coordinates[inputID] = CGPoint(x: vertexRadius * 2, y: inputIDSpace * (Double(i) + 0.5))
    }
  }

  let outputIDSpace = canvasHeight / Double(outputIDs.count)
  for (i, outputID) in outputIDs.enumerated() {
    coordinates[outputID] = CGPoint(
      x: canvasWidth - vertexRadius * 2, y: outputIDSpace * (Double(i) + 0.5))
  }

  for hiddenID in hiddenIDs {
    coordinates[hiddenID] = CGPoint(
      x: inputsWidth + Double.random(in: 0..<hiddenWidth),
      y: Double.random(in: 0..<(canvasHeight - vertexRadius * 2)) + vertexRadius
    )
  }

  var active = Set(withInputs)
  var g = DiGraph(vertices: program.nodes.keys)

  var frameIdx = 0
  func renderFrame(_ reachable: Set<Int>) throws {
    try renderGraphImage(
      graph: drawInputs ? g : g.removing(vertices: inputIDs),
      vertexPositions: coordinates,
      highlightedReachable: reachable,
      canvasSize: CGSize(width: canvasWidth, height: canvasHeight),
      vertexRadius: vertexRadius,
      drawEdges: drawEdges,
      outputURL: outputDir.appending(component: "frame_\(frameIdx).png")
    )
    frameIdx += 1
  }

  try renderFrame(active)

  for r in active {
    for edge in program.nodes[r]!.edges {
      g.insert(edge: edge)
    }
  }

  try renderFrame(active)

  while true {
    let reachable = g.reachable(from: active)
    if reachable.count == active.count {
      break
    }
    try renderFrame(reachable)
    for added in reachable.subtracting(active) {
      for edge in program.nodes[added]!.edges {
        g.insert(edge: edge)
      }
    }
    active = reachable
    try renderFrame(active)
  }
}
