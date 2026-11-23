"""Stable likelihoods for noisy 2-XOR responses and observed majority votes.

The fitted parameters are delay weights divided by the unknown noise scale.
No circuit weights or noise realization are passed to the fitting routine.
"""
from functools import lru_cache
from math import comb

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf, log_ndtr, logsumexp

METHODS = ('single', 'majority_naive', 'majority_aware', 'fraction')


def xor_logs(x, weights, gradient=False):
    delay = x @ weights.T
    negative, positive = log_ndtr(-delay), log_ndtr(delay)
    one = np.logaddexp(negative[:, 0] + positive[:, 1], positive[:, 0] + negative[:, 1])
    zero = np.logaddexp(negative[:, 0] + negative[:, 1], positive[:, 0] + positive[:, 1])
    if not gradient:
        return one, zero
    density = -.5 * delay**2 - .5 * np.log(2 * np.pi)
    other = erf(delay[:, ::-1] / np.sqrt(2))
    return one, zero, -other * np.exp(density - one[:, None]), other * np.exp(density - zero[:, None])


@lru_cache(maxsize=None)
def count_terms(reads):
    if reads < 1 or reads % 2 == 0:
        raise ValueError('An odd positive number of readings is required')
    counts = np.arange(reads + 1)
    return counts, np.log([comb(reads, int(k)) for k in counts])


def vote_logs(one, zero, reads):
    counts, coefficients = count_terms(reads)
    masses = coefficients + one[:, None] * counts + zero[:, None] * (reads - counts)
    upper = counts > reads // 2
    a = logsumexp(masses[:, upper], axis=1)
    b = logsumexp(masses[:, ~upper], axis=1)
    ka = (np.exp(masses[:, upper] - a[:, None]) * counts[upper]).sum(axis=1)
    kb = (np.exp(masses[:, ~upper] - b[:, None]) * counts[~upper]).sum(axis=1)
    return a, b, ka, kb


def objective(flat, x, target, method, reads, penalty):
    weights = flat.reshape(2, x.shape[1])
    a, b, ga, gb = xor_logs(x, weights, gradient=True)
    if method == 'majority_aware':
        av, bv, ka, kb = vote_logs(a, b, reads)
        mean_count = target * ka + (1 - target) * kb
        loss = -(target * av + (1 - target) * bv).mean() / reads
        derivative = -(mean_count[:, None] * ga + (reads - mean_count[:, None]) * gb) / reads
    elif method in ('single', 'fraction', 'majority_naive'):
        scale = reads if method == 'majority_naive' else 1
        loss = -(target * a + (1 - target) * b).mean() / scale
        derivative = -(target[:, None] * ga + (1 - target[:, None]) * gb) / scale
    else:
        raise ValueError(method)
    grad = derivative.T @ x / len(x) + penalty * weights
    return float(loss + penalty * np.sum(weights**2) / 2), grad.ravel()


def fit(x, target, vx, vy, method, reads, initials, penalties, maxiter):
    """Select only by log loss on single-read validation responses."""
    models, attempts = [], []
    for penalty in penalties:
        for restart, initial in enumerate(initials):
            result = minimize(objective, initial.ravel(), args=(x, target, method, reads, penalty),
                              jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-10})
            model = result.x.reshape(2, x.shape[1])
            a, b = xor_logs(vx, model)
            models.append(model)
            attempts.append({'penalty': penalty, 'restart': restart,
                'validation_log_loss': float(-(vy * a + (1 - vy) * b).mean()),
                'validation_accuracy': float(np.mean((a > b) == vy)),
                'loss': float(result.fun), 'gradient_norm': float(np.linalg.norm(result.jac)),
                'iterations': int(result.nit), 'converged': bool(result.success)})
    chosen = int(np.argmin([r['validation_log_loss'] for r in attempts]))
    for i, row in enumerate(attempts):
        row['selected'] = i == chosen
    return chosen, models, attempts
