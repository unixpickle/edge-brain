/// An efficient representation of an array of bits.
public struct Bitmap: Sendable, RandomAccessCollection, MutableCollection {
  public let count: Int
  private var values: [UInt64]

  public init(count: Int) {
    self.count = count
    values = [UInt64](repeating: 0, count: (count + 63) / 64)
  }

  public init<C: Collection<Bool>>(_ data: C) {
    self.count = data.count
    values = [UInt64](repeating: 0, count: (data.count + 63) / 64)
    for (i, x) in data.enumerated() {
      if x {
        self[i] = x
      }
    }
  }

  public subscript(_ i: Int) -> Bool {
    get {
      assert(i >= 0 && i < count)
      return values[i / 64] & (UInt64(1) << (i % 64)) != 0
    }

    set {
      assert(i >= 0 && i < count)
      if newValue {
        values[i / 64] = values[i / 64] | (UInt64(1) << (i % 64))
      } else {
        values[i / 64] = values[i / 64] & ~(UInt64(1) << (i % 64))
      }
    }
  }

  public typealias Index = Int
  public typealias Element = Bool

  public var startIndex: Int { 0 }
  public var endIndex: Int { count }

  public func index(after i: Int) -> Int {
    i + 1
  }

  public func index(before i: Int) -> Int {
    i - 1
  }

}
