import csv
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from puf_lab.sensitivity import correlated_parity_probability
from puf_lab.sensitivity_cli import main
from puf_lab.session import paired_error, session_offsets, session_trials


class SpatialTests(unittest.TestCase):
    def test_odd_parity_is_balanced_under_global_sign_symmetry(self):
        for k in (1, 3, 5, 7):
            for rho in (0, 0.3, 0.99, 1):
                self.assertEqual(correlated_parity_probability(k, rho, 0.1), 0.5)

    def test_even_parity_matches_closed_form_for_two_chains(self):
        for rho in (0, 0.1, 0.6, 0.999, 1):
            noise = 0.12
            expected = math.acos(rho / (1 + noise**2)) / math.pi
            self.assertAlmostEqual(correlated_parity_probability(2, rho, noise), expected)

    def test_quadrature_resolves_near_perfect_dependence(self):
        rho = 1 - 1e-12
        self.assertAlmostEqual(
            correlated_parity_probability(2, rho, 0), math.acos(rho) / math.pi, places=12,
        )

    def test_independence_and_perfect_noiseless_dependence(self):
        for k in (2, 4, 6):
            self.assertEqual(correlated_parity_probability(k, 0, 0.0), 0.5)
            self.assertEqual(correlated_parity_probability(k, 1, 0.0), 0.0)


class TemporalTests(unittest.TestCase):
    def test_stationary_path_variance_and_lag_dependence(self):
        rng = np.random.default_rng(7642)
        paths = session_offsets(rng, 0.25, (10, 80_000), 0.9)
        np.testing.assert_allclose(paths.std(axis=1), 0.25, atol=0.003)
        self.assertAlmostEqual(np.corrcoef(paths[[0, 9]])[0, 1], 0.9**9, delta=0.012)

    def test_fully_persistent_drift_is_common_to_all_protocols(self):
        rng = np.random.default_rng(19)
        delays = rng.normal(size=(9, 60, 4))
        for _, m in session_trials(delays, 0.1, 0.3, (1, 3, 5), rng, persistence=1):
            np.testing.assert_array_equal(m["same_session"], m["held_out"])
            np.testing.assert_array_equal(m["held_out"], m["distributed"])
        self.assertEqual(paired_error(4, 0.1, 0.3, True),
                         paired_error(4, 0.1, 0.3, False, lag_correlation=1))

    def test_sensitivity_profile_exports_full_grids(self):
        with tempfile.TemporaryDirectory() as directory:
            main(["--profile", "smoke", "--output", directory, "--quiet"])
            output = Path(directory)
            for name, count in (("spatial_bias", 36), ("temporal_persistence", 192)):
                with (output / f"{name}.csv").open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(len(rows), count)
                if name == "temporal_persistence":
                    for row in rows:
                        for strategy in ("held", "distributed"):
                            self.assertAlmostEqual(
                                float(row[f"rmse_{strategy}"])**2,
                                float(row[f"bias_{strategy}"])**2
                                + float(row[f"error_sd_{strategy}"])**2,
                            )
            self.assertTrue((output / "sensitivity.png").is_file())


if __name__ == "__main__":
    unittest.main()
