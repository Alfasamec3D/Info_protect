import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from puf_lab.calibration import (
    Curve, EQUAL_SPAN_PLANS, PLANS, SCENARIOS, TEST_LAGS,
    calibration_curve, candidate_estimate, inverse_ar, timing_path,
)
from puf_lab.calibration_cli import main


class CalibrationTheoryTests(unittest.TestCase):
    def test_exact_three_lag_inverse(self):
        for k in (1, 4):
            for eta in (0.0, 0.7, 0.98):
                ideal = Curve(0.1, ((0.26, eta),))
                fitted = inverse_ar(k, 9, ideal.disagreement(k, (0, 9, 18)))
                self.assertAlmostEqual(fitted.noise, ideal.noise)
                self.assertAlmostEqual(fitted.drift, ideal.drift)
                self.assertAlmostEqual(fitted.components[0][1], eta)

    def test_no_drift_is_unidentifiable_from_lag_differences(self):
        ideal = SCENARIOS["no_drift"]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            inverse_ar(4, 3, ideal.disagreement(4, (0, 3, 6)))

    def test_one_scale_recovers_noiseless_synthetic_aggregate(self):
        ideal = SCENARIOS["slow"]
        lags = PLANS["long"]
        fitted, ok = calibration_curve("single", 4, lags, ideal.disagreement(4, lags))
        self.assertTrue(ok)
        np.testing.assert_allclose(
            fitted.disagreement(4, TEST_LAGS), ideal.disagreement(4, TEST_LAGS),
            atol=1e-6,
        )

    def test_two_scale_recovers_unseen_lags_for_exact_observations(self):
        ideal = SCENARIOS["mixed"]
        lags = PLANS["long"]
        fitted, ok = calibration_curve("two_scale", 4, lags, ideal.disagreement(4, lags))
        self.assertTrue(ok)
        np.testing.assert_allclose(
            fitted.disagreement(4, TEST_LAGS), ideal.disagreement(4, TEST_LAGS),
            atol=1e-6,
        )

    def test_interpolation_holds_the_endpoints(self):
        measured = np.array([0.1, 0.2, 0.3])
        _, prediction, parameters = candidate_estimate(
            "interpolate", 4, (0, 10, 20), measured, (5, 15, 40),
        )
        np.testing.assert_allclose(prediction, [0.15, 0.25, 0.3])
        self.assertEqual(parameters["drift"], "")

    def test_irregular_session_sampling_preserves_mixed_covariance(self):
        ideal = SCENARIOS["mixed"]
        values = np.stack(list(timing_path(ideal, (0, 3, 12, 27), (100_000,),
                                            np.random.default_rng(71))))
        np.testing.assert_allclose(values.std(axis=1), ideal.drift, atol=0.003)
        actual = np.cov(values[0], values[-1])[0, 1]
        expected = sum(scale**2 * eta**27 for scale, eta in ideal.components)
        self.assertAlmostEqual(actual, expected, delta=0.0015)


class CalibrationSmokeTests(unittest.TestCase):
    def test_equal_span_plans_share_read_budget_and_horizon(self):
        plans = list(EQUAL_SPAN_PLANS.values())
        self.assertEqual({len(plan) for plan in plans}, {7})
        self.assertEqual({plan[-1] for plan in plans}, {72})
        for plan in plans:
            self.assertFalse(set(plan) & set(TEST_LAGS))

    def test_holdout_grid_and_budget(self):
        self.assertEqual(len(PLANS["short"]), len(PLANS["long"]))
        for lags in PLANS.values():
            self.assertEqual(set(lags) & set(TEST_LAGS), set())
        with tempfile.TemporaryDirectory() as folder:
            main(["--profile", "smoke", "--output", folder, "--quiet"])
            destination = Path(folder)
            with (destination / "calibration_predictions.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 240)
            self.assertEqual({int(row["lag"]) for row in rows}, set(TEST_LAGS))
            self.assertTrue((destination / "calibration.png").is_file())


if __name__ == "__main__":
    unittest.main()
