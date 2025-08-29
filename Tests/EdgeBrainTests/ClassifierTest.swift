import EdgeBrain
import Testing

@Test
func testClassifierPredictionLogits() {
  let pred = Classifier.Prediction(counts: [0, 3, 2], edgeWeight: 1.0)
  let actual = pred.logProbs
  let expected: [Float] = [-3.3490, -0.3490, -1.3490]
  for (a, x) in zip(actual, expected) {
    #expect(a.isFinite)
    #expect(abs(a - x) < 1e-4)
  }
  for (i, x) in pred.logProbs.enumerated() {
    #expect(abs(x - pred.logProb(label: i)) < 1e-4)
  }

  #expect(abs(pred.logProb(label: 0, withChange: 2, toLabel: 1) - (-5.0550)) < 1e-4)
  #expect(abs(pred.logProb(label: 1, withChange: 2, toLabel: 1) - (-0.0550)) < 1e-4)
  #expect(abs(pred.logProb(label: 2, withChange: 2, toLabel: 1) - (-3.0550)) < 1e-4)
}
