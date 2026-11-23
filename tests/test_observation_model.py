"""Checks of the observation likelihood, derivatives and isolation of evaluation."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.special import ndtr
from scipy.stats import binom

from puf_lab.observation_model import METHODS, fit, objective, vote_logs, xor_logs
from puf_lab.observation_study import Design, cohort
import puf_lab.observation_study as study


class ObservationTests(unittest.TestCase):
    def test_changed_frozen_protocol_is_rejected(self):
        with TemporaryDirectory() as folder:
            protocol = Path(folder) / 'protocol.json'
            protocol.write_text('{}')
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            lock = {'protocol_sha256': digest(protocol), 'scientific_source_sha256': {
                n: digest(Path(study.__file__).with_name(n)) for n in study.SOURCE_FILES}}
            protocol.with_name('observation_protocol_lock.json').write_text(json.dumps(lock))
            study.verify_lock(protocol)
            protocol.write_text('{"budget":1}')
            with self.assertRaises(ValueError): study.verify_lock(protocol)

    def test_probability_matches_independent_gaussian_parity(self):
        rng = np.random.default_rng(442)
        x, w = rng.normal(size=(30, 5)), rng.normal(size=(2, 5))
        q = ndtr(-(x @ w.T))
        expected = q[:, 0] + q[:, 1] - 2 * q[:, 0] * q[:, 1]
        a, b = xor_logs(x, w)
        np.testing.assert_allclose(np.exp(a), expected, atol=1e-14)
        np.testing.assert_allclose(np.exp(a) + np.exp(b), 1, atol=1e-14)

    def test_majority_probability_matches_binomial_tail(self):
        p = np.array([1e-8, .01, .2, .5, .9, 1 - 1e-8])
        for r in (1, 3, 9):
            a, b, _, _ = vote_logs(np.log(p), np.log1p(-p), r)
            np.testing.assert_allclose(a, binom.logsf(r // 2, r, p), atol=1e-12)
            np.testing.assert_allclose(b, binom.logcdf(r // 2, r, p), atol=1e-12)

    def test_all_gradients_match_finite_differences(self):
        rng = np.random.default_rng(551)
        x = rng.normal(size=(20, 5)); w = rng.normal(size=(2, 5)) * .3
        for method in METHODS:
            target = rng.integers(0, 10, 20) / 9 if method == 'fraction' else rng.integers(0, 2, 20)
            fun = lambda a: objective(a, x, target, method, 9, .01)[0]
            _, gradient = objective(w.ravel(), x, target, method, 9, .01)
            np.testing.assert_allclose(gradient, approx_derivative(fun, w.ravel()).ravel(), atol=2e-7, rtol=2e-6)

    def test_extreme_delays_have_finite_loss_and_gradient(self):
        x = np.array([[1., 0.], [0., 1.], [1., 1.], [1., -1.]])
        w = np.array([[50., -40.], [35., 45.]])
        for method in METHODS:
            loss, grad = objective(w.ravel(), x, np.array([0., 1., 0., 1.]), method, 9, .001)
            self.assertTrue(np.isfinite(loss)); self.assertTrue(np.isfinite(grad).all())

    def test_fraction_loss_equals_all_readings(self):
        rng = np.random.default_rng(88)
        x = rng.normal(size=(18, 4)); w = rng.normal(size=8) * .4
        bits = rng.integers(0, 2, (9, 18))
        expected = np.mean([objective(w, x, bit, 'single', 1, .01)[0] for bit in bits])
        self.assertAlmostEqual(expected, objective(w, x, bits.mean(axis=0), 'fraction', 9, .01)[0], places=12)

    def test_training_and_selection_do_not_depend_on_test_size(self):
        cfg = Design(seed=190009, devices=1, bits=12, budget=144, validation=48, test=48,
                     restarts=1, penalties=(.001,), maxiter=50, bootstrap=20)
        small, large = cohort(cfg, 0), cohort(replace(cfg, test=96), 0)
        for index in (0, 1, 2):
            np.testing.assert_array_equal(small[index][:192], large[index][:192])
        np.testing.assert_array_equal(small[3], large[3])
        initials = np.random.default_rng(4).normal(0, .2, (1, 2, 13))
        fits = []
        for data in (small, large):
            _, x, labels, bits, va, *_ = data
            fits.append(fit(x[:16], bits.mean(axis=0), x[va], labels[va], 'fraction', 9,
                            initials, cfg.penalties, cfg.maxiter))
        np.testing.assert_array_equal(fits[0][1][0], fits[1][1][0])
        self.assertEqual(fits[0][2], fits[1][2])


if __name__ == '__main__':
    unittest.main()
