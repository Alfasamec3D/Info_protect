"""Outcome-dependent repeated-query acquisition and probit likelihood.

Independent research module: does not change historical experiment fingerprints.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr


def stopped_mean(p, cap=5):
    """Stop at 3 unanimous responses; otherwise finish at cap >= 3."""
    p = np.asarray(p)
    return p + (1-3/cap)*p*(1-p)*(2*p-1)


def expected_count(p, cap=5):
    p = np.asarray(p)
    return 3+(cap-3)*3*p*(1-p)


def probit_loss(w, x, successes, counts, weighted=True, l2=1e-6):
    z = x @ w
    lp, ln = log_ndtr(z), log_ndtr(-z)
    lphi = -.5*z*z-.5*np.log(2*np.pi)
    if weighted:
        positive, negative = successes, counts-successes
    else:
        positive, negative = successes/counts, 1-successes/counts
    norm = np.sum(positive+negative)
    value = -np.sum(positive*lp+negative*ln)/norm
    dz = (-positive*np.exp(lphi-lp)+negative*np.exp(lphi-ln))/norm
    return value+l2*np.dot(w,w)/2, x.T @ dz+l2*w


@dataclass
class Probit:
    weights: np.ndarray
    converged: bool
    iterations: int

    @classmethod
    def fit(cls,x,successes,counts,weighted):
        result = minimize(probit_loss,np.zeros(x.shape[1]),
                          args=(x,successes,counts,weighted),jac=True,
                          method='L-BFGS-B',options={'maxiter':300,'ftol':1e-12,'gtol':1e-7})
        if not np.isfinite(result.fun):
            raise RuntimeError('non-finite objective')
        return cls(result.x,bool(result.success),int(result.nit))

    def predict(self,x):
        return ndtr(x @ self.weights)


class BernoulliOracle:
    """Probability array is private to the simulator, never passed to fit."""
    def __init__(self,probabilities,seed):
        self._p=np.asarray(probabilities)
        self._rng=np.random.default_rng(seed)
        self.queries=0

    def query(self,index,count):
        self.queries+=count
        return int(np.sum(self._rng.random(count)<self._p[index]))


def acquire(oracle,budget,cap):
    """Acquire complete groups; reserve cap before starting, unused tail < cap."""
    if cap < 3 or budget < cap:
        raise ValueError('cap >= 3 and budget >= cap required')
    successes, counts = [], []
    while budget-oracle.queries >= cap:
        index=len(counts)
        s=oracle.query(index,3)
        n=3
        if s not in (0,3):
            s+=oracle.query(index,cap-3)
            n=cap
        successes.append(s)
        counts.append(n)
    return np.array(successes,float),np.array(counts,float)
