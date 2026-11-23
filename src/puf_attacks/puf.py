"""Bipolar convention: bit 0 -> +1, bit 1 -> -1; XOR becomes product."""
from dataclasses import dataclass
import numpy as np
from scipy.special import ndtr


def features(challenges):
    c = np.asarray(challenges)
    if c.ndim != 2 or c.shape[1] < 1 or not np.isin(c, [-1, 1]).all():
        raise ValueError("challenges must be a nonempty-width matrix of -1/+1")
    suffix = np.cumprod(c[:, ::-1], axis=1, dtype=np.int8)[:, ::-1]
    return np.column_stack((suffix, np.ones(len(c)))).astype(float)


def challenges(n, count, rng, exclude=None):
    if n < 1 or count < 1:
        raise ValueError("n and count must be positive")
    seen = set() if exclude is None else set(exclude)
    if n < 63 and count + len(seen) > 2**n:
        raise ValueError("not enough distinct challenges")
    out = []
    while len(out) < count:
        batch = rng.choice(np.array([-1, 1], dtype=np.int8), (count-len(out), n))
        for row in batch:
            key = row.tobytes()
            if key not in seen:
                out.append(row)
                seen.add(key)
    return np.asarray(out, dtype=np.int8)


@dataclass
class ArbiterPUF:
    # Simulator secrets. Attack code is passed only public CRP arrays.
    weights: np.ndarray

    @classmethod
    def random(cls, n=64, k=1, seed=0):
        if n < 1 or k < 1:
            raise ValueError("n and k must be positive")
        return cls(np.random.default_rng(seed).normal(size=(k, n+1))/np.sqrt(n+1))

    def margins(self, c):
        return features(c) @ self.weights.T

    def response(self, c, noise=0., repeats=1, rng=None):
        if noise < 0 or repeats < 1:
            raise ValueError("noise >= 0 and repeats >= 1 required")
        m = self.margins(c)
        z = np.broadcast_to(m[:, None, :], (len(c), repeats, len(self.weights)))
        if noise:
            if rng is None:
                raise ValueError("noisy evaluation needs an explicit random generator")
            z = z + rng.normal(scale=noise, size=z.shape)
        return np.prod(np.where(z >= 0, 1, -1), axis=2).astype(np.int8)

    def probability_positive(self, c, noise):
        """Exact noisy output probability; evaluator-only, never given to attacks."""
        m = self.margins(c)
        if noise == 0:
            return (np.prod(np.where(m >= 0, 1, -1), axis=1)+1)/2
        if noise < 0:
            raise ValueError("noise must be nonnegative")
        return (1+np.prod(2*ndtr(m/noise)-1, axis=1))/2


def split_challenges(n, sizes, seed):
    all_c = challenges(n, sum(sizes), np.random.default_rng(seed))
    return np.split(all_c, np.cumsum(sizes)[:-1])
