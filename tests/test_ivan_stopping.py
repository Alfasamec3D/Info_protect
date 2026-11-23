"""Original theory checks plus integration tests for outcome-dependent sampling."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.optimize import check_grad

from puf_lab.ivan_studies import load_stopping, summarize_stopping, interval

method, _ = load_stopping()
spec = importlib.util.spec_from_file_location('ivan_theory_tests', Path('research/stopping/test_method.py'))
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
StoppingTests = original.StoppingTests


class IntegratedStoppingTests(unittest.TestCase):
    def test_variable_count_gradient(self):
        rng = np.random.default_rng(819)
        x = rng.normal(size=(8, 5)); w = rng.normal(size=5)
        counts = np.array([3, 15, 3, 15, 3, 15, 3, 15.])
        successes = np.array([0, 5, 3, 11, 0, 7, 3, 13.])
        for weighted in (True, False):
            def f(v): return method.probit_loss(v, x, successes, counts, weighted)
            self.assertLess(check_grad(lambda v: f(v)[0], lambda v: f(v)[1], w), 2e-6)

    def test_count_likelihood_matches_individual_observations(self):
        x = np.array([[1., 2.], [-1., .1], [.5, 1.]])
        n = np.array([3, 5, 15]); s = np.array([3, 2, 7]); w = np.array([.4, -.2])
        labels = np.concatenate([np.r_[np.ones(a), np.zeros(b-a)] for a, b in zip(s, n)])
        a = method.probit_loss(w, x, s, n, True)
        b = method.probit_loss(w, np.repeat(x, n, axis=0), labels, np.ones(len(labels)), True)
        self.assertAlmostEqual(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])
        self.assertNotAlmostEqual(a[0], method.probit_loss(w, x, s, n, False)[0])

    def test_familywise_interval_and_complete_cohorts(self):
        values = [.1, .3, -.2, .4, .1]
        regular, corrected = interval(values), interval(values, 8)
        self.assertLess(corrected['ci_low'], regular['ci_low'])
        self.assertGreater(corrected['ci_high'], regular['ci_high'])
        config = dict(seeds=[1, 2], sigmas=[.25], budgets=[50], caps=[5])
        with self.assertRaises(AssertionError):
            summarize_stopping([], config)

    def test_single_oracle_query_matches_gaussian_sign_probability(self):
        from puf_attacks.puf import ArbiterPUF
        puf = ArbiterPUF(np.array([[.3, -.2]]))
        challenges = np.array([[1], [-1]])
        probabilities = puf.probability_positive(challenges, .75)
        oracle = method.BernoulliOracle(probabilities, 191)
        for i, p in enumerate(probabilities):
            self.assertLess(abs(oracle.query(i, 100000)/100000-p), .006)


if __name__ == '__main__':
    unittest.main()
