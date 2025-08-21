import Foundation

class SyncSum: @unchecked Sendable {
  private var sum: Double = 0
  private var lock = NSLock()

  func add(_ x: Float) {
    lock.withLock { sum += Double(x) }
  }

  var value: Float {
    lock.withLock { Float(sum) }
  }
}

class SyncArray<T: Sendable>: @unchecked Sendable {
  private var arr: [T]
  private var lock = NSLock()

  init(repeating: T, count: Int) {
    arr = [T](repeating: repeating, count: count)
  }

  func set(_ value: T, at: Int) {
    lock.withLock { arr[at] = value }
  }

  var value: [T] {
    lock.withLock { arr }
  }
}
