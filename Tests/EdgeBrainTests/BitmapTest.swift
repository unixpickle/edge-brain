import EdgeBrain
import Testing

@Test
func testBitmap() {
  var bmp = Bitmap(count: 1000)
  var actual = [Bool](repeating: false, count: 1000)
  for _ in 0..<10 {
    for i in 0..<1000 {
      switch (0..<3).randomElement()! {
      case 0:
        bmp[i] = true
        actual[i] = true
      case 1:
        bmp[i] = false
        actual[i] = false
      default: ()
      }
    }
  }
  for i in 0..<1000 {
    #expect(bmp[i] == actual[i])
  }
  #expect(Array(bmp) == actual)
  #expect(Array(bmp.enumerated().map { $0.offset }) == Array(actual.enumerated().map { $0.offset }))
  #expect(
    Array(bmp.enumerated().map { $0.element }) == Array(actual.enumerated().map { $0.element })
  )
}
