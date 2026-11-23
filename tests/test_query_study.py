import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from puf_lab.query_study import PROFILES, repeated_noise, fit_mlp, stored_mlp_predict, majority_labels, run
from puf_lab.model import challenge_features


class QueryStudyTests(unittest.TestCase):
    def test_majority_votes_on_signed_bits(self):
        delays = np.array([[-1., 2.], [2., 3.], [-1., -3.]])
        eps = np.zeros((9, 3, 2))
        eps[:2, 0, 0] = 2
        np.testing.assert_array_equal(majority_labels(delays, 0., eps), [True, False, False])
        np.testing.assert_array_equal(majority_labels(delays, 1., eps[:3]), [False, False, False])
        np.testing.assert_array_equal(majority_labels(delays, 1., eps), [True, False, False])

    def test_noiseless_training_is_correct_and_single_arbiter_learns(self):
        profile = replace(PROFILES["smoke"], devices=1, bits=8, budgets=(144,), chains=(1,),
                          noises=(0.,), validation=32, test=64, maxiter=80)
        with TemporaryDirectory() as folder:
            rows, *_ = run(profile, Path(folder), lambda _: None, collect_only=True)
        self.assertTrue(all(r["train_label_error"] == 0 for r in rows))
        product = next(r for r in rows if r["model"] == "product")
        self.assertGreater(product["accuracy"], .9)

    def test_noise_preserves_marginals_and_requested_correlation(self):
        for rho in (0., .9):
            noise = repeated_noise(np.random.default_rng(12), 100000, 1, rho, repeats=3)[:, :, 0]
            np.testing.assert_allclose(noise.var(axis=1), 1, atol=.015)
            expected = np.full((3, 3), rho)
            np.fill_diagonal(expected, 1)
            np.testing.assert_allclose(np.corrcoef(noise), expected, atol=.012)

    def test_first_read_fixed_and_perfectly_correlated_repeats_identical(self):
        iid = repeated_noise(np.random.default_rng(3), 200, 4, 0.)
        common = repeated_noise(np.random.default_rng(3), 200, 4, 1.)
        np.testing.assert_array_equal(iid[0], common[0])
        for reading in common:
            np.testing.assert_array_equal(reading, common[0])

    def test_mlp_export_reproduces_predictions(self):
        rng = np.random.default_rng(7)
        x = np.column_stack((rng.normal(size=(128, 8)), np.ones(128)))
        y = x[:, 0] + x[:, 1] < 0
        model, trials = fit_mlp(x[:64], y[:64], x[64:96], y[64:96], rng, 2, 10)
        np.testing.assert_array_equal(model.predict(x[96:, :-1]),
                                      stored_mlp_predict(x[96:], model.coefs_, model.intercepts_))
        selected = [t for t in trials if t["selected"]]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["validation_accuracy"], max(t["validation_accuracy"] for t in trials))
        for label in (False, True):
            model, _ = fit_mlp(x[:64], np.full(64, label), x[64:96], np.full(32, label), rng, 1, 2)
            np.testing.assert_array_equal(model.predict(x[96:, :-1]),
                stored_mlp_predict(x[96:], model.coefs_, model.intercepts_, model.classes_))

    def test_features_match_independent_switch_recurrence(self):
        rng = np.random.default_rng(17)
        challenges = rng.integers(0, 2, (100, 64), dtype=np.uint8)
        weights = rng.normal(size=(4, 65))
        delay = np.zeros((100, 4))
        for stage in range(64):
            delay = (delay + weights[:, stage]) * (1 - 2 * challenges[:, stage].astype(float))[:, None]
        delay += weights[:, -1]
        np.testing.assert_allclose(delay, challenge_features(challenges) @ weights.T, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
