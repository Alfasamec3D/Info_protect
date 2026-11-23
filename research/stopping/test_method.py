import itertools
import unittest
import numpy as np
from scipy.optimize import check_grad
from scipy.special import ndtr
from method import stopped_mean, expected_count, probit_loss, BernoulliOracle, acquire


class StoppingTests(unittest.TestCase):
    def test_exhaustive_law_and_score(self):
        # Enumerate all stopped paths (not Monte Carlo; independent of formulas).
        for cap in [3,5,7]:
            for p in [.05,.2,.5,.7,.95]:
                mass=mean=length=score=0.
                for first in itertools.product([0,1],repeat=3):
                    s=sum(first)
                    suffixes=[()] if s in (0,3) else itertools.product([0,1],repeat=cap-3)
                    for tail in suffixes:
                        n=3+len(tail);total=s+sum(tail)
                        prob=p**total*(1-p)**(n-total)
                        mass+=prob;mean+=prob*total/n;length+=prob*n
                        score+=prob*(total-p*n)
                self.assertAlmostEqual(mass,1)
                self.assertAlmostEqual(mean,stopped_mean(p,cap))
                self.assertAlmostEqual(length,expected_count(p,cap))
                self.assertAlmostEqual(score,0)

    def test_gradients_and_expanded_likelihood(self):
        rng=np.random.default_rng(5)
        x=rng.normal(size=(6,4));s=np.array([0,1,2,3,4,5.]);n=np.full(6,5.)
        w=rng.normal(size=4)
        for weighted in [True,False]:
            obj=lambda v:probit_loss(v,x,s,n,weighted)
            self.assertLess(check_grad(lambda v:obj(v)[0],lambda v:obj(v)[1],w),1e-6)
        labels=np.concatenate([np.r_[np.ones(int(a)),np.zeros(int(b-a))] for a,b in zip(s,n)])
        expanded=probit_loss(w,np.repeat(x,5,axis=0),labels,np.ones(len(labels)),True)
        aggregate=probit_loss(w,x,s,n,True)
        self.assertAlmostEqual(expanded[0],aggregate[0])
        np.testing.assert_allclose(expanded[1],aggregate[1])
        # Fixed repeats: both objectives agree exactly, including regularization.
        fixed=probit_loss(w,x,s,n,False)
        self.assertAlmostEqual(fixed[0],aggregate[0])
        np.testing.assert_allclose(fixed[1],aggregate[1])

    def test_scale_derivative(self):
        # Direct expected loss using stopped-label mean, compared with theorem.
        z=np.array([-2.,-.8,-.2,.1,.7,1.7])
        p=ndtr(z)
        for cap in [5,15]:
            q=stopped_mean(p,cap)
            def risk(alpha):
                return probit_loss(np.array([alpha]),z[:,None],q,np.ones(len(q)),False,0.)[0]
            h=1e-5
            finite=(risk(1+h)-risk(1-h))/(2*h)
            exact=-(1-3/cap)*np.mean((2*p-1)*np.exp(-z*z/2)/np.sqrt(2*np.pi)*z)
            self.assertAlmostEqual(finite,exact,places=8)
            self.assertLess(exact,0.)

    def test_budget_and_policy(self):
        o=BernoulliOracle(np.full(1000,.5),1)
        s,n=acquire(o,1000,15)
        self.assertEqual(o.queries,int(n.sum()))
        self.assertLess(1000-o.queries,15)
        self.assertTrue(np.all(np.isin(n,[3,15])))
        self.assertTrue(np.all(np.isin(s[n==3],[0,3])))
        o=BernoulliOracle(np.ones(1000),1)
        s,n=acquire(o,100,15)
        self.assertTrue(np.all(n==3))
        self.assertTrue(np.all(s==3))

if __name__=='__main__':unittest.main()
