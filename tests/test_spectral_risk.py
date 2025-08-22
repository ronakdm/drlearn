"""Tests for sklearn estimators in the drlearn package."""

from absl.testing import absltest, parameterized

import numpy as np

# Public API
from drlearn import (
    make_esrm_spectrum,
    make_extremile_spectrum,
    make_superquantile_spectrum,
)

# Internal API
from drlearn.spectral_risk import make_spectral_risk_measure_oracle

class TestSpectralRisk(parameterized.TestCase):

    @parameterized.product(params=[
        (make_superquantile_spectrum, 1.0),
        (make_extremile_spectrum, 1.0),
        (make_esrm_spectrum, 0.0),
    ])
    def test_uniform_spectrum(self, params):
        # uniformity parameter recovers uniform spectrum
        batch_size = 30
        uniform = [1. / batch_size for _ in range(batch_size)]
        test_func, param = params
        self.assertSequenceAlmostEqual(test_func(batch_size, param), uniform, places=6)

    @parameterized.product(params=[
        (make_superquantile_spectrum, 0.2),
        (make_extremile_spectrum, 3.0),
        (make_esrm_spectrum, 0.0),
    ], penalty=["chi2", "kl"])
    def test_probability_weights(self, params, penalty):
        # spectra are valid probability weights
        batch_size = 30
        test_func, param = params
        spectrum = test_func(batch_size, param)
        self.assertGreaterEqual(spectrum.min(), -1e-12)
        self.assertAlmostEqual(spectrum.sum(), 1.0, places=6)

        max_oracle = make_spectral_risk_measure_oracle(spectrum, penalty, 1.0)
        np.random.seed(123)
        losses = np.random.normal(size=(batch_size,))
        val, weights = max_oracle(losses)
        self.assertGreaterEqual(weights.min().item(), -1e-12)
        self.assertAlmostEqual(weights.sum().item(), 1., places=6)

    @parameterized.product(params=[
        (make_superquantile_spectrum, 0.25, 0.75),
        (make_extremile_spectrum, 1.5, 2.0),
        (make_esrm_spectrum, 1.0, 2.0),
    ], penalty=["chi2", "kl"])
    def test_monotonicity(self, params, penalty):
        batch_size = 16
        test_func, param1, param2 = params

        # risk parameters exhibit correct monotonicity relations
        np.random.seed(459)
        losses = np.sort(np.random.normal(size=(batch_size,)))
        spectrum1 = test_func(batch_size, param1)
        spectrum2 = test_func(batch_size, param2)

        max_oracle1 = make_spectral_risk_measure_oracle(spectrum1, penalty, 0.1)
        max_oracle2 = make_spectral_risk_measure_oracle(spectrum2, penalty, 0.1)
        max_oracle3 = make_spectral_risk_measure_oracle(spectrum1, penalty, 0.2)

        val1, weights1 = max_oracle1(losses)
        val2, weights2 = max_oracle2(losses)
        val3, weights3 = max_oracle3(losses)

        self.assertGreaterEqual(val2, val1)
        self.assertGreaterEqual(val2, val3)

if __name__ == '__main__':
  absltest.main()