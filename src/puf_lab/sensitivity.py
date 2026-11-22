"""Independent sensitivity experiments for spatial and temporal dependence."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from .experiment import stream
from .model import random_features, read_error, sample_weights
from .session import paired_error, session_trials
from .summary import interval


SPATIAL_CORRELATIONS = (0.0, 0.1, 0.3, 0.6, 0.9, 1.0)
PERSISTENCE_LEVELS = (0.0, 0.7, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0)
DRIFT_LEVELS = (0.08, 0.3)
TEMPORAL_CHAINS = (1, 2, 4)
REPEATS = (1, 3, 5, 9)
READ_NOISE = 0.1
BUDGET = 2400


@dataclass(frozen=True)
class StudySize:
    seed: int
    spatial_devices: int
    spatial_inputs: int
    temporal_devices: int
    trials: int
    reference_inputs: int
    resamples: int


PROFILES = {
    "smoke": StudySize(764211, 12, 256, 4, 4, 512, 100),
    "main": StudySize(764211, 256, 8192, 48, 32, 32768, 4000),
    "repeat": StudySize(764212, 128, 8192, 24, 32, 32768, 4000),
}


def correlated_parity_probability(chains, correlation, noise):
    """Probability of an odd number of negative coordinates with a common factor."""
    if chains % 2 or correlation == 0:
        return 0.5
    residual = 1 - correlation + noise**2
    if residual == 0:
        return 0.0
    factor = math.sqrt(correlation / residual)

    # Near perfect dependence, integrate on the scale of sign disagreements.
    scale = 1 / factor if factor > 100 else 1.0

    def density(x):
        z = scale * x
        p = ndtr(-factor * z)
        parity = (1 - (1 - 2 * p)**chains) / 2
        return scale * parity * math.exp(-z**2 / 2) / math.sqrt(2 * math.pi)

    # For even parity the integrand is even. Integrate over both tails via symmetry.
    mass, _ = quad(density, 0, np.inf, epsabs=1e-12)
    return 2 * mass


def resampling_indices(seed, stream_id, devices, count):
    return stream(seed, stream_id).integers(devices, size=(count, devices))


def spatial_study(size):
    bits, chains = 64, 6
    features = random_features(stream(size.seed, 40), size.spatial_inputs, bits)
    values = np.zeros((len(SPATIAL_CORRELATIONS), size.spatial_devices, chains))
    for device in range(size.spatial_devices):
        weights = sample_weights(stream(size.seed, 41, device), chains + 1, bits)
        delays = features @ weights.T
        common, private = delays[:, :1], delays[:, 1:]
        noise = stream(size.seed, 42, device).normal(0, READ_NOISE, private.shape)
        for i, rho in enumerate(SPATIAL_CORRELATIONS):
            response = math.sqrt(rho) * common + math.sqrt(1 - rho) * private + noise
            values[i, device] = np.logical_xor.accumulate(response < 0, axis=1).mean(axis=0)

    draws = resampling_indices(size.seed, 43, size.spatial_devices, size.resamples)
    rows = []
    for i, rho in enumerate(SPATIAL_CORRELATIONS):
        for j in range(chains):
            fraction = values[i, :, j]
            theoretical = correlated_parity_probability(j + 1, rho, READ_NOISE)
            low, high = interval(fraction[draws].mean(axis=1))
            rows.append({
                "chains": j + 1, "correlation": rho, "read_noise": READ_NOISE,
                "fraction_ones": float(fraction.mean()), "ci_low": low, "ci_high": high,
                "theory_fraction_ones": theoretical,
                "theory_constant_accuracy": max(theoretical, 1 - theoretical),
                "difference": float(fraction.mean() - theoretical),
            })
    return rows


def temporal_device(size, device):
    chains = max(TEMPORAL_CHAINS)
    weights = sample_weights(stream(size.seed, 50, device), chains, 64)
    reference = random_features(
        stream(size.seed, 51, device), size.reference_inputs, 64,
    ) @ weights.T
    delays = random_features(
        stream(size.seed, 52, device), size.trials * (BUDGET // 2), 64,
    ) @ weights.T
    delays = delays.reshape(size.trials, BUDGET // 2, chains)
    for i, drift in enumerate(DRIFT_LEVELS):
        target = read_error(reference, math.hypot(READ_NOISE, drift)).mean(axis=0)
        for eta in PERSISTENCE_LEVELS:
            # Reuse the draws when varying persistence to keep the comparison paired.
            rng = stream(size.seed, 53, device, i)
            for count, measured in session_trials(
                delays, READ_NOISE, drift, REPEATS, rng, persistence=eta,
            ):
                same = measured["same_session"]
                held = measured["held_out"]
                distributed = measured["distributed"]
                for k in TEMPORAL_CHAINS:
                    j = k - 1
                    yield (k, drift, eta, count), (
                        float(np.mean(same[:, j])),
                        float(np.mean(held[:, j])),
                        float(np.mean(distributed[:, j])),
                        float(target[j]),
                        float(np.mean((held[:, j] - target[j])**2)),
                        float(np.mean((distributed[:, j] - target[j])**2)),
                    )


def temporal_study(size, progress=None):
    aggregation = {}
    for device in range(size.temporal_devices):
        for key, values in temporal_device(size, device):
            aggregation.setdefault(key, []).append(values)
        if progress is not None:
            progress(device + 1, size.temporal_devices)
    draws = resampling_indices(size.seed, 54, size.temporal_devices, size.resamples)

    rows = []
    for (k, drift, eta, reads), measurements in sorted(aggregation.items()):
        same, held, dist, target, mse_held, mse_dist = np.array(measurements).T
        bias_held, bias_dist = float(np.mean(held - target)), float(np.mean(dist - target))
        gaps = held - same
        rmse_held = float(np.sqrt(mse_held.mean()))
        rmse_dist = float(np.sqrt(mse_dist.mean()))
        gain = np.sqrt(mse_held[draws].mean(axis=1)) - np.sqrt(mse_dist[draws].mean(axis=1))
        low, high = interval(gaps[draws].mean(axis=1))
        gain_low, gain_high = interval(gain)
        endpoint_corr = eta**REPEATS[-1]
        expected_gap = (
            paired_error(k, READ_NOISE, drift, False, lag_correlation=endpoint_corr)
            - paired_error(k, READ_NOISE, drift, True)
        )
        rows.append({
            "chains": k, "drift": drift, "persistence": eta, "reads": reads,
            "budget": BUDGET, "read_noise": READ_NOISE, "lag": REPEATS[-1],
            "lag_correlation": endpoint_corr,
            "same_session": float(same.mean()), "held_out": float(held.mean()),
            "distributed": float(dist.mean()), "held_minus_same": float(gaps.mean()),
            "gap_ci_low": low, "gap_ci_high": high,
            "analytic_gap_r1": expected_gap if reads == 1 else "",
            "rmse_held": rmse_held, "rmse_distributed": rmse_dist,
            "target_ber": float(target.mean()),
            "bias_held": bias_held, "bias_distributed": bias_dist,
            "error_sd_held": math.sqrt(max(0, mse_held.mean() - bias_held**2)),
            "error_sd_distributed": math.sqrt(max(0, mse_dist.mean() - bias_dist**2)),
            "rmse_gain": rmse_held - rmse_dist, "rmse_gain_ci_low": gain_low,
            "rmse_gain_ci_high": gain_high,
        })
    return rows
