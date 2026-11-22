"""Normalized additive-delay model and readout error probabilities."""

import numpy as np
from scipy.special import ndtr
from scipy.stats import binom


def challenge_features(challenges):
    """Map binary rows to suffix products and append the affine coordinate."""
    signs = 1 - 2 * challenges.astype(np.float64)
    products = np.cumprod(signs[:, ::-1], axis=1)[:, ::-1]
    return np.column_stack((products, np.ones(len(products))))


def random_features(rng, count, bits):
    challenges = rng.integers(0, 2, size=(count, bits), dtype=np.uint8)
    return challenge_features(challenges)


def sample_weights(rng, chains, bits):
    """Sample independent normal weights with half variance at both ends."""
    weights = rng.normal(size=(chains, bits + 1)) / np.sqrt(bits)
    weights[:, [0, -1]] /= np.sqrt(2)
    return weights


def parity_error(component_errors):
    """Probability of an odd number of independent component errors."""
    return (1 - np.cumprod(1 - 2 * component_errors, axis=-1)) / 2


def read_error(delays, noise):
    """Error probability versus the noiseless response for each XOR prefix."""
    if noise == 0:
        return np.zeros_like(delays)
    return parity_error(ndtr(-np.abs(delays) / noise))


def population_error(chains, noise):
    """Exact BER averaged over normal circuit variation and readout noise."""
    single = np.arctan(noise) / np.pi
    return float((1 - (1 - 2 * single) ** chains) / 2)


def disagreement_probability(read_probability, repeats):
    """Probability that an independent read differs from majority enrollment."""
    reference_probability = binom.sf(repeats // 2, repeats, read_probability)
    return read_probability + (1 - 2 * read_probability) * reference_probability
