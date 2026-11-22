import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from puf_lab.cli import main
from puf_lab.config import Config
from puf_lab.experiment import trial_estimates
from puf_lab.model import (
    challenge_features,
    disagreement_probability,
    parity_error,
    population_error,
    read_error,
    sample_weights,
)

ROOT = Path(__file__).resolve().parents[1]


class DelayModelTests(unittest.TestCase):
    def test_suffix_products_and_affine_term(self):
        bits = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=np.uint8)
        expected = [[1, 1, 1, 1], [-1, 1, 1, 1], [1, 1, -1, 1]]
        np.testing.assert_array_equal(challenge_features(bits), expected)

    def test_two_chain_flip_requires_exactly_one_error(self):
        observed = parity_error(np.array([[0.1, 0.2]]))
        np.testing.assert_allclose(observed, [[0.1, 0.26]])

    def test_zero_noise_and_exact_tie(self):
        delays = np.array([[0.0, -0.25], [1.0, 0.0]])
        np.testing.assert_array_equal(read_error(delays, 0.0), np.zeros_like(delays))
        np.testing.assert_allclose(read_error(delays, 0.1), [
            [0.5, 0.5], [0.0, 0.5],
        ], atol=1e-15)

    def test_probability_agrees_with_direct_noisy_readout(self):
        rng = np.random.default_rng(42)
        clean_delays = np.array([0.04, -0.18, 0.3])
        noisy_delays = clean_delays + rng.normal(0, 0.1, size=(150_000, 3))
        changed = (noisy_delays < 0) != (clean_delays < 0)
        observed = np.logical_xor.accumulate(changed, axis=1).mean(axis=0)
        predicted = read_error(clean_delays[None, :], 0.1)[0]
        np.testing.assert_allclose(observed, predicted, atol=0.004)

    def test_population_p_equation_with_sampled_delay_weights(self):
        rng = np.random.default_rng(3)
        features = challenge_features(np.array([[0, 1, 1, 0]], dtype=np.uint8))[0]
        weights = sample_weights(rng, 160_000, 4).reshape(-1, 4, 5)
        conditional_error = read_error(weights @ features, 0.25).mean(axis=0)
        expected = [population_error(k, 0.25) for k in range(1, 5)]
        np.testing.assert_allclose(conditional_error, expected, atol=0.003)


class BudgetTests(unittest.TestCase):
    def test_majority_disagreement(self):
        p = np.array([0.0, 0.2, 0.5])
        np.testing.assert_allclose(disagreement_probability(p, 1), 2 * p * (1 - p))
        np.testing.assert_allclose(disagreement_probability(p, 3), [0.0, 0.2624, 0.5])

    def test_fixed_read_budget_simulation_moments(self):
        budget = 120
        trials = np.full((16_000, budget // 2), 0.2)
        for repeats, estimates in trial_estimates(
            trials, (1, 3, 5, 9), budget, np.random.default_rng(88)
        ):
            expected = float(disagreement_probability(0.2, repeats))
            variance = expected * (1 - expected) / (budget // (repeats + 1))
            self.assertAlmostEqual(float(estimates.mean()), expected, delta=0.002)
            self.assertAlmostEqual(float(estimates.var()), variance, delta=0.00035)

    def test_invalid_read_budget_is_rejected(self):
        config = Config.load(ROOT / "configs/smoke.json")
        with self.assertRaisesRegex(ValueError, "divisible"):
            replace(config, budgets=(121,)).validate()


class CommandLineTests(unittest.TestCase):
    def test_smoke_outputs_and_exact_reproduction(self):
        config = ROOT / "configs/smoke.json"
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "first"
            second = Path(temp) / "second"
            for output in (first, second):
                main(["--config", str(config), "--output", str(output), "--quiet"])
            tabular = [
                "reliability", "uniqueness", "budget_metrics", "budget_winners",
                "device_metrics", "pair_metrics", "budget_by_device",
            ]
            for name in tabular:
                self.assertEqual(
                    (first / f"{name}.csv").read_bytes(),
                    (second / f"{name}.csv").read_bytes(),
                )
            with (first / "budget_metrics.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 32)
            for row in rows:
                self.assertEqual(
                    int(row["challenges"]) * (int(row["enrollment_reads"]) + 1),
                    int(row["budget"]),
                )
            manifest = json.loads((first / "manifest.json").read_text())
            self.assertEqual(manifest["config"]["devices"], 4)
            with (first / "reliability.csv").open() as file:
                for row in csv.DictReader(file):
                    ber = float(row["ber"])
                    baseline = float(row["majority_accuracy"])
                    ceiling = float(row["noisy_read_ceiling"])
                    self.assertAlmostEqual(ceiling, 1 - ber)
                    self.assertLessEqual(0.5, baseline)
                    self.assertLessEqual(baseline, ceiling)
            for name in ("reliability", "measurement_budget", "reference_points"):
                self.assertGreater((first / f"{name}.png").stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
