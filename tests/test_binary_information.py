import itertools
import unittest

import numpy as np
from scipy.special import ndtr
from scipy.stats import binom

from puf_lab.binary_information import coefficient,majority_retention


class BinaryInformationTests(unittest.TestCase):
    def test_information_from_finite_difference_bernoulli_score(self):
        sigma=.75
        for delay in (-2.,-.3,0.,1.4):
            p=ndtr(-delay/sigma)
            eps=1e-6
            plus=ndtr(-(delay+eps)/sigma);minus=ndtr(-(delay-eps)/sigma)
            score1=(np.log(plus)-np.log(minus))/(2*eps)
            score0=(np.log1p(-plus)-np.log1p(-minus))/(2*eps)
            actual=p*score1**2+(1-p)*score0**2
            np.testing.assert_allclose(actual,coefficient(delay,sigma),rtol=1e-6)

    def test_count_sufficiency_and_majority_information_loss(self):
        for p in (.1,.3,.5,.8):
            for r in (1,3,9):
                counts=np.arange(r+1)
                score=(counts-r*p)/(p*(1-p))
                count_information=np.sum(binom.pmf(counts,r,p)*score**2)
                self.assertAlmostEqual(count_information,r/(p*(1-p)),places=10)
                ratio=float(majority_retention(p,r))
                self.assertLessEqual(ratio,1+1e-12)
                self.assertGreater(ratio,0)
                if r==1:self.assertAlmostEqual(ratio,1)
                else:self.assertLess(ratio,1)
        self.assertAlmostEqual(float(majority_retention(.5,3)),.75)

    def test_exact_random_design_moments_and_rank_on_finite_population(self):
        x=np.column_stack((np.array(list(itertools.product((-1.,1.),repeat=3))),np.ones(8)))
        a=coefficient(x@np.array([.4,-.2,.3,.1]),.75)
        matrices=a[:,None,None]*x[:,:,None]*x[:,None,:]
        # Two IID design points versus two reads at one point: T=2.
        single=np.array([u+v for u in matrices for v in matrices])
        repeat=2*matrices
        np.testing.assert_allclose(single.mean(axis=0),repeat.mean(axis=0),atol=1e-12)
        np.testing.assert_allclose(repeat.var(axis=0),2*single.var(axis=0),atol=1e-12)
        self.assertTrue(all(np.linalg.matrix_rank(m)==1 for m in repeat))


if __name__=='__main__':unittest.main()
