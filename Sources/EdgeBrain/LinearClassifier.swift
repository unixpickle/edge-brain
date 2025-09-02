import HCBacktrace
import Honeycrisp

public class LinearClassifier: Trainable {
  @Param var weight: Tensor
  @Param var bias: Tensor

  public init(featureCount: Int, outputCount: Int) {
    super.init()
    weight = Tensor(randn: [featureCount, outputCount]) / Float(featureCount).squareRoot()
    bias = Tensor(zeros: [outputCount])
  }

  @recordCaller
  public func fit(
    program: Program,
    data: [Bitmap],
    labels: [Int],
    batchSize: Int = 128,
    lr: Float = 0.01,
    iters: Int = 10000
  ) async throws {
    let hiddenIDs = program.hiddenIDs()
    let opt = Adam(parameters, lr: lr)

    var epochData = data
    var epochLabels = labels

    for iter in 0..<iters {
      opt.lr = lr * Float(iters - iter) / Float(iters)

      while epochData.count < batchSize {
        let idxs = (0..<data.count).shuffled()
        epochData.append(contentsOf: idxs.map { data[$0] })
        epochLabels.append(contentsOf: idxs.map { labels[$0] })
      }
      let nextData = Array(epochData[..<batchSize])
      let nextLabels = Array(epochLabels[..<batchSize])
      epochData.removeFirst(batchSize)
      epochLabels.removeFirst(batchSize)

      let inputVector = Tensor(data: nextData.map { bmp in hiddenIDs.map { id in bmp[id] } }).cast(
        .float32)
      let targetVector = Tensor(data: nextLabels, dtype: .int64)

      let logits = (inputVector &* weight) + bias
      let logProbs = (-logits.logSoftmax(axis: -1)).gather(
        axis: 1, indices: targetVector[..., NewAxis()])
      let loss = logProbs.mean()
      loss.backward()

      opt.step()
      opt.clearGrads()
    }
  }

  @recordCaller
  public func loss(
    program: Program,
    data: [Bitmap],
    labels: [Int],
    bs: Int = 128
  ) async throws -> Float {
    let hiddenIDs = program.hiddenIDs()

    var losses = [Float]()

    for i in stride(from: 0, to: data.count, by: bs) {
      let mb = min(bs, data.count - i)
      let inputVector = Tensor(
        data: data[i..<(i + mb)].map { bmp in hiddenIDs.map { id in bmp[id] } }
      ).cast(.float32)
      let targetVector = Tensor(data: labels[i..<(i + mb)], dtype: .int64)

      let logits = (inputVector &* weight) + bias
      let logProbs = (-logits.logSoftmax(axis: -1)).gather(
        axis: 1, indices: targetVector[..., NewAxis()])
      losses.append(try await logProbs.sum().item())
    }

    return losses.reduce(0, +) / Float(data.count)
  }
}
