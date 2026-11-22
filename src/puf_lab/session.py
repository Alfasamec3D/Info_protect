"""Paired reliability experiments with persistent drift between measurement sessions."""

from dataclasses import dataclass
import json
import math

import numpy as np

from .experiment import stream
from .model import population_error, random_features, read_error, sample_weights
from .summary import group_rows, interval, vector

PROTOCOLS = ("same_session", "held_out", "distributed")


@dataclass(frozen=True)
class SessionConfig:
    seed: int
    challenge_bits: int
    chains: tuple
    devices: int
    trials: int
    budgets: tuple
    enrollment_reads: tuple
    read_noise: tuple
    session_drift: tuple
    reference_challenges: int
    bootstrap_resamples: int

    @classmethod
    def load(cls, path):
        with path.open(encoding="utf-8") as file:
            fields = json.load(file)
        for name in ("chains", "budgets", "enrollment_reads", "read_noise", "session_drift"):
            fields[name] = tuple(fields[name])
        config = cls(**fields)
        config.validate()
        return config

    def validate(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("challenge_bits", "devices", "trials", "reference_challenges",
                     "bootstrap_resamples"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 2:
                raise ValueError(f"{name} must be an integer >= 2")
        for name in ("chains", "budgets", "enrollment_reads"):
            values = getattr(self, name)
            if not values or any(type(k) is not int or k < 1 for k in values):
                raise ValueError(f"{name} must contain positive integers")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be strictly increasing")
        if self.enrollment_reads[0] != 1 or any(r % 2 == 0 for r in self.enrollment_reads):
            raise ValueError("enrollment_reads must start at 1 and contain only odd values")
        if any(b % (r + 1) for b in self.budgets for r in self.enrollment_reads):
            raise ValueError("each budget must be divisible by every enrollment_reads + 1")
        for name in ("read_noise", "session_drift"):
            values = getattr(self, name)
            if not values or any(type(v) not in (float, int) or not math.isfinite(v) or v < 0
                                 for v in values):
                raise ValueError(f"{name} must contain finite nonnegative numbers")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be strictly increasing")
        if self.read_noise[0] == 0:
            raise ValueError("read_noise must be positive")
        if self.session_drift[0] != 0 or len(self.session_drift) < 2:
            raise ValueError("session_drift must include zero and at least one positive value")


def paired_error(chains, noise, drift, same_session, *, lag_correlation=0.0):
    """Population discrepancy of two single reads via the bivariate normal law."""
    total_variance = 1 + noise**2 + drift**2
    covariance = 1 + drift**2 * (1 if same_session else lag_correlation)
    error = math.acos(covariance / total_variance) / math.pi
    return (1 - (1 - 2 * error)**chains) / 2


def read_signals(delays, offset, noise):
    """Read parity prefixes under a fixed device-session offset."""
    return np.logical_xor.accumulate(delays + offset + noise < 0, axis=-1)


def session_offsets(rng, drift, shape, persistence):
    """Sample a stationary normal AR(1) path along the session axis."""
    offsets = rng.normal(0, drift, size=shape)
    innovation_scale = math.sqrt(1 - persistence**2)
    for index in range(1, len(offsets)):
        offsets[index] = persistence * offsets[index - 1] + innovation_scale * offsets[index]
    return offsets


def session_trials(delays, noise, drift, repetitions, rng, *, persistence=0.0):
    """Evaluate three paired protocols with exactly R+1 reads per chosen input."""
    trials, largest_count, chains = delays.shape
    budget = 2 * largest_count
    offsets = session_offsets(
        rng, drift, (repetitions[-1] + 1, trials, 1, chains), persistence,
    )
    check_noise = rng.normal(0, noise, size=delays.shape)
    check_same = read_signals(delays, offsets[0], check_noise)
    check_held = read_signals(delays, offsets[-1], check_noise)
    counts_pooled = np.zeros(delays.shape, dtype=np.int32)
    counts_distributed = np.zeros_like(counts_pooled)
    previous = 0
    for repeats in repetitions:
        count = budget // (repeats + 1)
        local = delays[:, :count, :]
        for read in range(previous, repeats):
            epsilon = rng.normal(0, noise, size=local.shape)
            pooled_bits = read_signals(local, offsets[0], epsilon)
            distributed_bits = read_signals(local, offsets[read], epsilon)
            counts_pooled[:, :count, :] += pooled_bits
            counts_distributed[:, :count, :] += distributed_bits
        pooled_majority = counts_pooled[:, :count, :] > repeats // 2
        distributed_majority = counts_distributed[:, :count, :] > repeats // 2
        yield repeats, {
            "same_session": (pooled_majority != check_same[:, :count, :]).mean(axis=1),
            "held_out": (pooled_majority != check_held[:, :count, :]).mean(axis=1),
            "distributed": (distributed_majority != check_held[:, :count, :]).mean(axis=1),
        }
        previous = repeats


def device_study(config, device):
    bits = config.challenge_bits
    chains = max(config.chains)
    weights = sample_weights(stream(config.seed, 20, device), chains, bits)
    reference = random_features(
        stream(config.seed, 21, device), config.reference_challenges, bits,
    ) @ weights.T
    pairs = []
    observations = []
    for budget in config.budgets:
        count = budget // 2
        delays = random_features(
            stream(config.seed, 22, device, budget), count * config.trials, bits,
        ) @ weights.T
        delays = delays.reshape(config.trials, count, chains)
        for i, noise in enumerate(config.read_noise):
            for j, drift in enumerate(config.session_drift):
                target = read_error(reference, math.hypot(noise, drift)).mean(axis=0)
                rng = stream(config.seed, 23, device, budget, i, j)
                for repeats, estimates in session_trials(
                    delays, noise, drift, config.enrollment_reads, rng,
                ):
                    for k in config.chains:
                        p = target[k - 1]
                        for protocol, values in estimates.items():
                            measured = values[:, k - 1]
                            observations.append({
                                "device": device, "chains": k, "noise": noise,
                                "drift": drift, "budget": budget, "reads": repeats,
                                "protocol": protocol, "target": float(p),
                                "mean": float(measured.mean()),
                                "mse": float(np.mean((measured - p)**2)),
                            })
                        same, held, distributed = (estimates[name][:, k - 1] for name in PROTOCOLS)
                        pairs.append({
                            "device": device, "chains": k, "noise": noise,
                            "drift": drift, "budget": budget, "reads": repeats,
                            "gap": float(np.mean(held - same)),
                            "distributed_bias_delta": float(np.mean(distributed - held)),
                        })
    return observations, pairs


def collect(config, progress=None):
    config.validate()
    rows, differences = [], []
    for device in range(config.devices):
        measured, pairs = device_study(config, device)
        rows.extend(measured)
        differences.extend(pairs)
        if progress is not None:
            progress(device + 1, config.devices)
    return rows, differences


GROUP = ("chains", "noise", "drift", "budget", "reads")


def session_summary(config, rows, pairs):
    rng = stream(config.seed, 24)
    size = (config.bootstrap_resamples, config.devices)
    draws = rng.integers(config.devices, size=size)
    metrics, contrasts, checks = [], [], []
    bootstrap_rmse = {}
    for values, group in group_rows(rows, *GROUP, "protocol").items():
        record = dict(zip((*GROUP, "protocol"), values))
        k, noise, drift, budget, reads, protocol = values
        mean = vector(group, "mean")
        target = vector(group, "target")
        bias = mean - target
        mse = vector(group, "mse")
        low, high = interval(bias[draws].mean(axis=1))
        rmse_draws = np.sqrt(mse[draws].mean(axis=1))
        bootstrap_rmse[values] = (float(np.sqrt(mse.mean())), rmse_draws)
        rmse_low, rmse_high = interval(rmse_draws)
        metrics.append({
            **record, "challenges": budget // (reads + 1),
            "observed": float(mean.mean()), "target": float(target.mean()),
            "bias": float(bias.mean()), "bias_ci_low": low, "bias_ci_high": high,
            "rmse": float(np.sqrt(mse.mean())), "rmse_ci_low": rmse_low,
            "rmse_ci_high": rmse_high,
        })
        if reads == 1:
            expected = paired_error(k, noise, drift, protocol == "same_session")
            obs_low, obs_high = interval(mean[draws].mean(axis=1))
            checks.append({
                **record, "observed": float(mean.mean()), "analytic": expected,
                "difference": float(mean.mean() - expected),
                "ci_low": obs_low, "ci_high": obs_high,
                "population_ber": population_error(k, math.hypot(noise, drift)),
            })
    for values, group in group_rows(pairs, *GROUP).items():
        gap = vector(group, "gap")
        enrollment = vector(group, "distributed_bias_delta")
        low, high = interval(gap[draws].mean(axis=1))
        enroll_low, enroll_high = interval(enrollment[draws].mean(axis=1))
        r_distributed, draw_distributed = bootstrap_rmse[(*values, "distributed")]
        r_held, draw_held = bootstrap_rmse[(*values, "held_out")]
        rmse_low, rmse_high = interval(draw_held - draw_distributed)
        contrasts.append({
            **dict(zip(GROUP, values)),
            "held_minus_same": float(gap.mean()),
            "gap_ci_low": low, "gap_ci_high": high,
            "distributed_minus_held": float(enrollment.mean()),
            "distributed_ci_low": enroll_low, "distributed_ci_high": enroll_high,
            "rmse_gain": r_held - r_distributed, "rmse_gain_ci_low": rmse_low,
            "rmse_gain_ci_high": rmse_high,
        })
    return {"session_metrics": metrics, "session_contrasts": contrasts,
            "session_reference_checks": checks}
