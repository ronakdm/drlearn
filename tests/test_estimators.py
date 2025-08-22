"""Tests for sklearn estimators in the drlearn package."""

from absl.testing import absltest

import numpy as np

import sys
sys.path.append(".")
from src import Ridge, BinaryLogisticRegression, MultinomialLogisticRegression

class TestEstimator(absltest.TestCase):

    def test_ridge(self):
        X = np.random.rand(100, 10)
        y = np.random.rand(100)

        model = Ridge()
        model.fit(X, y)
        predictions = model.predict(X)

        self.assertEqual(predictions.shape, (100,))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_binary_logistic_regression(self):
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, size=100)

        model = BinaryLogisticRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        self.assertEqual(predictions.shape, (100,))
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))

    def test_multinomial_logistic_regression(self):
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 3, size=100)

        model = MultinomialLogisticRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        self.assertEqual(predictions.shape, (100,))
        self.assertTrue(np.all(np.isin(predictions, [0, 1, 2])))

if __name__ == '__main__':
  absltest.main()