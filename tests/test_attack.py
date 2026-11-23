"""Checks of the attack's mathematical objective and experimental separation."""

from dataclasses import replace
import unittest

import numpy as np
from scipy.optimize import check_grad

from puf_lab.attack import (PROFILES, fit_product, objective, predict_product,
                            run, sample_device)


class AttackTests(unittest.TestCase):
    def test_analytic_gradient_including_zero_delays(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(30, 6))
        y = rng.choice([-1., 1.], 30)
        for chains in (1, 2, 4):
            w = rng.normal(size=(chains, 6)).ravel()
            error = check_grad(lambda v: objective(v, x, y, chains)[0],
                               lambda v: objective(v, x, y, chains)[1], w)
            self.assertLess(error, 2e-6)
            self.assertTrue(np.isfinite(objective(np.zeros_like(w), x, y, chains)[1]).all())

    def test_partitions_disjoint_and_repeat_independent(self):
        p = PROFILES["smoke"]
        raw, x, weights, slices = sample_device(p, 0)
        groups = [{row.tobytes() for row in raw[s]} for s in slices]
        self.assertTrue(groups[0].isdisjoint(groups[1]))
        self.assertTrue(groups[0].isdisjoint(groups[2]))
        self.assertTrue(groups[1].isdisjoint(groups[2]))
        self.assertEqual(sum(map(len, groups)), len(raw))
        np.testing.assert_array_equal(x, sample_device(p, 0)[1])
        self.assertFalse(np.array_equal(weights, sample_device(replace(p, seed=p.seed + 1), 0)[2]))
        with self.assertRaises(ValueError):
            sample_device(replace(p, bits=2), 0)

    def test_single_arbiter_learnable_on_unseen_inputs(self):
        p = replace(PROFILES["smoke"], budgets=(1024,), bits=16)
        _, x, weights, (tr, va, te) = sample_device(p, 0)
        labels = x @ weights[0] < 0
        model, candidates = fit_product(x[tr], labels[tr], x[va], labels[va], 1,
                                        np.random.default_rng(3), 2, 150)
        self.assertGreater(np.mean(predict_product(x[te], model) == labels[te]), .95)
        selected = [r for r in candidates if r["selected"]]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["validation_accuracy"],
                         max(r["validation_accuracy"] for r in candidates))

    def test_test_set_size_cannot_change_training(self):
        p = replace(PROFILES["smoke"], devices=1, chains=(1,), noises=(0.,), budgets=(64,))
        first, _ = run(p)
        second, _ = run(replace(p, test=512))
        self.assertEqual(first["validation"], second["validation"])
        again, _ = run(p)
        self.assertEqual(first, again)


if __name__ == "__main__":
    unittest.main()
