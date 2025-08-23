import AppKit
import EdgeBrain

enum RenderError: Error {
  case internalError
  case writeError(Error)
}

@discardableResult
public func renderGraphImage<V: Hashable>(
  graph: DiGraph<V>,
  vertexPositions: [V: CGPoint],
  highlightedReachable: Set<V>,
  canvasSize: CGSize,
  vertexRadius: CGFloat,
  drawEdges: Bool,
  outputURL: URL? = nil
) throws -> NSImage {
  let image = NSImage(size: canvasSize)
  image.lockFocus()

  guard let ctx = NSGraphicsContext.current?.cgContext else {
    image.unlockFocus()
    throw RenderError.internalError
  }

  ctx.setFillColor(NSColor.white.cgColor)
  ctx.fill(CGRect(origin: .zero, size: canvasSize))

  if drawEdges {
    for from in graph.vertices {
      guard let fromPt = vertexPositions[from] else { continue }

      for to in graph.neighbors(from: from) {
        guard let toPt = vertexPositions[to] else { continue }

        ctx.setStrokeColor(NSColor.black.cgColor)
        ctx.setLineWidth(2.0)
        ctx.move(to: fromPt)
        ctx.addLine(to: toPt)
        ctx.strokePath()
      }
    }
  }

  for (vertex, pos) in vertexPositions {
    let rect = CGRect(
      x: pos.x - vertexRadius,
      y: pos.y - vertexRadius,
      width: vertexRadius * 2,
      height: vertexRadius * 2
    )

    let color = highlightedReachable.contains(vertex) ? NSColor.systemRed : NSColor.systemBlue
    ctx.setFillColor(color.cgColor)
    ctx.fillEllipse(in: rect)
  }

  image.unlockFocus()

  // Optionally save to disk
  if let url = outputURL {
    if let tiffData = image.tiffRepresentation,
      let bitmap = NSBitmapImageRep(data: tiffData),
      let pngData = bitmap.representation(using: .png, properties: [:])
    {
      do {
        try pngData.write(to: url)
      } catch {
        throw RenderError.writeError(error)
      }
    } else {
      throw RenderError.internalError
    }
  }

  return image
}
