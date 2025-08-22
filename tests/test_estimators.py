"""Tests for sklearn estimators in the drlearn package."""

from absl.testing import absltest, parameterized

import numpy as np

# Public API
from drlearn import (
    make_esrm_spectrum,
    make_extremile_spectrum,
    make_superquantile_spectrum,
    Ridge, 
    BinaryLogisticRegression, 
    MultinomialLogisticRegression
)

# Internal API
from drlearn.loss import (
    squared_error_first_order_oracle,
    binary_cross_entropy_first_order_oracle,
    multinomial_cross_entropy_first_order_oracle
)
from drlearn.spectral_risk import make_spectral_risk_measure_oracle

class TestEstimator(parameterized.TestCase):

    @parameterized.product(optim=["prospect", "lsvrg"])
    def test_centering(self, optim):
        np.random.seed(123)
        X = np.random.rand(100, 10)
        y = np.random.rand(100)

        # Check if the model is fitted
        model = Ridge(optim=optim)
        model.fit(X, y)
        predictions = model.predict(X)
        self.assertEqual(model.coef_.shape, (10,))
        self.assertGreaterEqual(y.var(), ((y - predictions) ** 2).mean())

        # Check if the model is fitted (on centered data)
        X_ = X - X.mean(axis=0)
        y_ = y - y.mean()
        model = Ridge(optim=optim, fit_intercept=False)
        model.fit(X_, y_)
        predictions = model.predict(X_)
        self.assertEqual(model.coef_.shape, (10,))
        self.assertAlmostEqual(model.intercept_, 0.0, places=8)
        self.assertGreaterEqual(y_.var(), ((y_ - predictions) ** 2).mean())

    @parameterized.product(optim=["prospect", "lsvrg"], penalty=["chi2", "kl"])
    def test_ridge(self, optim, penalty):
        np.random.seed(123)
        n = 100
        X = np.random.normal(size=(n, 10))
        y = np.random.normal(size=(n,))
        shift_cost = 0.1
        weight_decay = 0.01

        spectrum1 = make_superquantile_spectrum(n, 0.5)
        max_oracle1 = make_spectral_risk_measure_oracle(spectrum1, penalty, shift_cost)
        spectrum2 = make_extremile_spectrum(n, 2.0)
        max_oracle2 = make_spectral_risk_measure_oracle(spectrum2, penalty, shift_cost)

        model1 = Ridge(
            spectrum=spectrum1, 
            optim=optim, 
            fit_intercept=False, 
            penalty=penalty, 
            shift_cost=shift_cost,
            weight_decay=weight_decay
        )
        model1.fit(X, y)
        losses1 = squared_error_first_order_oracle(model1.coef_, X, y)[0]

        model2 = Ridge(
            spectrum=spectrum2, 
            optim=optim, 
            fit_intercept=False, 
            penalty=penalty, 
            shift_cost=shift_cost,
            weight_decay=weight_decay
        )
        model2.fit(X, y)
        losses2 = squared_error_first_order_oracle(model2.coef_, X, y)[0]

        val11 = max_oracle1(losses1)[0] + 0.5 * weight_decay * np.sum(model1.coef_ ** 2)
        val12 = max_oracle1(losses2)[0] + 0.5 * weight_decay * np.sum(model2.coef_ ** 2)
        val21 = max_oracle2(losses1)[0] + 0.5 * weight_decay * np.sum(model1.coef_ ** 2)
        val22 = max_oracle2(losses2)[0] + 0.5 * weight_decay * np.sum(model2.coef_ ** 2)

        self.assertGreaterEqual(val12, val11)
        self.assertGreaterEqual(val21, val22)

    @parameterized.product(optim=["prospect", "lsvrg"], penalty=["chi2", "kl"])
    def test_binary_logistic_regression(self, optim, penalty):
        np.random.seed(456)
        n = 100
        X = np.random.normal(size=(n, 10))
        y = np.random.binomial(1, 0.5, size=(n,))
        shift_cost = 0.1
        weight_decay = 0.01

        spectrum1 = make_extremile_spectrum(n, 2.0)
        max_oracle1 = make_spectral_risk_measure_oracle(spectrum1, penalty, shift_cost)
        spectrum2 = make_esrm_spectrum(n, 1.0)
        max_oracle2 = make_spectral_risk_measure_oracle(spectrum2, penalty, shift_cost)

        model1 = BinaryLogisticRegression(
            spectrum=spectrum1, 
            optim=optim, 
            fit_intercept=False, 
            penalty=penalty, 
            shift_cost=shift_cost,
            weight_decay=weight_decay
        )
        model1.fit(X, y)
        losses1 = binary_cross_entropy_first_order_oracle(model1.coef_, X, y)[0]

        model2 = BinaryLogisticRegression(
            spectrum=spectrum2, 
            optim=optim, 
            fit_intercept=False, 
            penalty=penalty, 
            shift_cost=shift_cost,
            weight_decay=weight_decay
        )
        model2.fit(X, y)
        losses2 = binary_cross_entropy_first_order_oracle(model2.coef_, X, y)[0]

        val11 = max_oracle1(losses1)[0] + 0.5 * weight_decay * np.sum(model1.coef_ ** 2)
        val12 = max_oracle1(losses2)[0] + 0.5 * weight_decay * np.sum(model2.coef_ ** 2)
        val21 = max_oracle2(losses1)[0] + 0.5 * weight_decay * np.sum(model1.coef_ ** 2)
        val22 = max_oracle2(losses2)[0] + 0.5 * weight_decay * np.sum(model2.coef_ ** 2)

        self.assertGreaterEqual(val12, val11)
        self.assertGreaterEqual(val21, val22)

    @parameterized.product(optim=["prospect", "lsvrg"], penalty=["chi2", "kl"])
    def test_multinomial_logistic_regression(self, optim, penalty):
        np.random.seed(789)
        n, n_classes = 100, 5
        X = np.random.normal(size=(n, 10))
        y = np.random.choice(n_classes, size=(n,), replace=True)
        shift_cost = 0.1
        weight_decay = 0.01

        spectrum1 = make_esrm_spectrum(n, 1.0)
        max_oracle1 = make_spectral_risk_measure_oracle(spectrum1, penalty, shift_cost)
        spectrum2 = make_superquantile_spectrum(n, 0.5)
        max_oracle2 = make_spectral_risk_measure_oracle(spectrum2, penalty, shift_cost)

        model1 = MultinomialLogisticRegression(
            spectrum=spectrum1, 
            optim=optim, 
            fit_intercept=False, 
            penalty=penalty, 
            shift_cost=shift_cost,
            weight_decay=weight_decay
        )
        model1.fit(X, y)
        self.assertEqual(model1.coef_.shape, (10 * n_classes,))
        losses1 = multinomial_cross_entropy_first_order_oracle(n_classes, model1.coef_, X, y)[0]

        model2 = MultinomialLogisticRegression(
            spectrum=spectrum2, 
            optim=optim, 
            fit_intercept=False, 
            penalty=penalty, 
            shift_cost=shift_cost,
            weight_decay=weight_decay
        )
        model2.fit(X, y)
        losses2 = multinomial_cross_entropy_first_order_oracle(n_classes, model2.coef_, X, y)[0]

        val11 = max_oracle1(losses1)[0] + 0.5 * weight_decay * np.sum(model1.coef_ ** 2)
        val12 = max_oracle1(losses2)[0] + 0.5 * weight_decay * np.sum(model2.coef_ ** 2)
        val21 = max_oracle2(losses1)[0] + 0.5 * weight_decay * np.sum(model1.coef_ ** 2)
        val22 = max_oracle2(losses2)[0] + 0.5 * weight_decay * np.sum(model2.coef_ ** 2)

        self.assertGreaterEqual(val12, val11)
        self.assertGreaterEqual(val21, val22)

    def test_relabeling(self):
        np.random.seed(123)
        n, n_classes = 100, 5
        class_names = np.array([10, 3, 5, 7, 1])
        X = np.random.normal(size=(n, 10))
        y = class_names[np.random.choice(n_classes, size=(n,), replace=True)]
        shift_cost = 0.1
        weight_decay = 0.01
        penalty = "chi2"
        optim = "lsvrg"

        spectrum = make_esrm_spectrum(n, 1.0)

        model = MultinomialLogisticRegression(
            spectrum=spectrum, 
            optim=optim, 
            fit_intercept=False, 
            penalty=penalty, 
            shift_cost=shift_cost,
            weight_decay=weight_decay
        )
        model.fit(X, y)
        pred = model.predict(X)
        pred_proba = model.predict_proba(X)
        self.assertTrue(np.isin(pred, class_names).all())
        self.assertEqual(pred_proba.shape, (n, n_classes))
        self.assertAlmostEqual(pred_proba.sum(axis=1).min(), 1.0, places=6)
        self.assertAlmostEqual(pred_proba.sum(axis=1).max(), 1.0, places=6)
        self.assertGreaterEqual(pred_proba.min(), 0.0)

if __name__ == '__main__':
  absltest.main()