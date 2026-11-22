import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from puf_lab.forecast import (
    Cohort, FEATURES, PROFILES, compare, evaluate_transfer, simulate_device,
)
from puf_lab.forecast_cli import main


class ForecastTests(unittest.TestCase):
    def test_future_target_cannot_change_calibration_features(self):
        profile = PROFILES["smoke"]
        x, y = simulate_device(profile, 2, 2, 11)
        with patch("puf_lab.forecast.future_ber", return_value=1.0):
            changed_x, changed_y = simulate_device(profile, 2, 2, 11)
        self.assertNotEqual(y, changed_y)
        self.assertEqual(x, changed_x)

    def test_interval_uses_paired_device_differences(self):
        cohort = Cohort(("a", "b"), np.empty((2, len(FEATURES))), np.array([0.1, 0.2]))
        estimates = {"baseline": np.array([0.0, 0.5]), "ridge": np.array([0.1, 0.4])}
        rows = compare("test", cohort, estimates, np.array([[1, 1], [0, 0], [0, 1]]))
        self.assertAlmostEqual(rows[1]["mae_gain"], 0.1)
        self.assertAlmostEqual(rows[1]["gain_ci_low"], 0.1)
        self.assertAlmostEqual(rows[1]["gain_ci_high"], 0.1)

    def test_transfer_only_uses_frozen_models_and_target_devices(self):
        class ConstantModel:
            def __init__(self, value):
                self.value = value

            def predict(self, matrix):
                return np.full(len(matrix), self.value)

        models = {
            "stable": {"ridge": ConstantModel(0.1)},
            "volatile": {"ridge": ConstantModel(0.3)},
        }
        first = Cohort(("stable-test",), np.array([[0.1]]), np.array([0.1]))
        second = Cohort(("volatile-test",), np.array([[0.2]]), np.array([0.3]))
        draws = np.array([[0]])
        metrics, predictions = evaluate_transfer(
            models, {"stable": (first, draws), "volatile": (second, draws)},
        )
        self.assertEqual(len(metrics), 4)
        self.assertEqual(
            [(r["source_regime"], r["device_id"]) for r in predictions],
            [("stable", "volatile-test"), ("volatile", "stable-test")],
        )
        self.assertEqual([r["ridge"] for r in predictions], [0.1, 0.3])

    def test_smoke_device_split_and_baseline(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            main(["--profile", "smoke", "--output", temporary, "--quiet"])
            with (output / "partition_audit.csv").open() as f:
                partitions = list(csv.DictReader(f))
            devices = {row["device_id"] for row in partitions}
            self.assertEqual(len(devices), len(partitions))
            with (output / "device_predictions.csv").open() as f:
                predictions = list(csv.DictReader(f))
            tested = {row["device_id"] for row in partitions if row["split"] == "test"}
            self.assertEqual({row["device_id"] for row in predictions}, tested)
            self.assertEqual(len(predictions), 60)
            for row in predictions:
                self.assertEqual(row["baseline"], row["current_ber"])
                self.assertTrue(0 <= float(row["next_ber"]) <= 1)
            with (output / "metrics.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 9)
            with (output / "transfer_metrics.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 18)
            self.assertTrue((output / "forecast.png").is_file())
            self.assertTrue((output / "transfer.png").is_file())


if __name__ == "__main__":
    unittest.main()
