import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from puf_lab.control import population_one_probability, run
from puf_lab.session import paired_error, session_trials
from puf_lab.session_cli import main

ROOT = Path(__file__).resolve().parents[1]


class SessionTests(unittest.TestCase):
    def test_no_drift_makes_protocols_identical_under_paired_draws(self):
        delays = np.random.default_rng(10).normal(size=(12, 60, 4))
        for _, values in session_trials(delays, 0.1, 0, (1, 3, 5), np.random.default_rng(5)):
            np.testing.assert_array_equal(values["same_session"], values["held_out"])
            np.testing.assert_array_equal(values["distributed"], values["held_out"])

    def test_single_reads_follow_bivariate_normal_formula(self):
        delays = np.random.default_rng(19).normal(size=(24_000, 24, 4))
        trials = session_trials(delays, 0.1, 0.2, (1,), np.random.default_rng(20))
        _, observed = next(trials)
        for name in ("same_session", "held_out"):
            expected = [paired_error(k, 0.1, 0.2, name == "same_session")
                        for k in range(1, 5)]
            np.testing.assert_allclose(observed[name].mean(axis=0), expected, atol=0.004)
        np.testing.assert_array_equal(observed["held_out"], observed["distributed"])

    def test_smoke_exact_budget_and_valid_intervals(self):
        with tempfile.TemporaryDirectory() as work:
            destination = Path(work)
            main(["--config", str(ROOT / "configs/session_smoke.json"),
                  "--output", work, "--quiet"])
            with (destination / "session_metrics.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 72)
            for row in rows:
                self.assertEqual(
                    int(row["challenges"]) * (int(row["reads"]) + 1), int(row["budget"]),
                )
                self.assertLessEqual(float(row["bias_ci_low"]), float(row["bias_ci_high"]))
            self.assertTrue((destination / "session_study.png").is_file())


class PopulationTests(unittest.TestCase):
    def test_centered_prior_has_equal_outcomes_for_any_parity(self):
        for k in range(1, 9):
            self.assertEqual(population_one_probability(k, 0.0, 0.1), 0.5)

    def test_uncentered_population_control(self):
        rows, _ = run(128238, devices=80, inputs=1024)
        for row in rows:
            self.assertAlmostEqual(row["fraction_ones"], row["theory_fraction_ones"], delta=0.015)
        self.assertGreater(population_one_probability(4, 0.6, 0.1),
                           population_one_probability(1, 0.6, 0.1))


if __name__ == "__main__":
    unittest.main()
