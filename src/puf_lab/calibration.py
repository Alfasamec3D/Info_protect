"""Calibrate population-level PUF repeatability from aggregate disagreements."""

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import least_squares

from .experiment import stream
from .model import random_features, sample_weights
from .session import read_signals
from .summary import group_rows, interval, vector


PLANS = {
    "short": (0, 1, 2, 3, 4, 6, 9),
    "long": (0, 1, 3, 9, 18, 36, 72),
}
TEST_LAGS = (12, 27, 54, 108)
EQUAL_SPAN_PLANS = {
    "early": (0, 1, 2, 3, 4, 6, 72),
    "spread": PLANS["long"],
    "late": (0, 5, 10, 20, 32, 48, 72),
}
COMPARISONS = (
    ("long_vs_short_single", ("short", "single"), ("long", "single")),
    ("two_scale_vs_single_long", ("long", "single"), ("long", "two_scale")),
    ("two_scale_vs_interpolation_long", ("long", "interpolate"), ("long", "two_scale")),
)
EQUAL_SPAN_COMPARISONS = (
    ("spread_vs_early_two", ("early", "two_scale"), ("spread", "two_scale")),
    ("spread_vs_late_two", ("late", "two_scale"), ("spread", "two_scale")),
    ("spread_two_vs_interpolation", ("spread", "interpolate"), ("spread", "two_scale")),
)
CHAINS = (1, 4)


@dataclass(frozen=True)
class Curve:
    noise: float
    components: tuple[tuple[float, float], ...]

    @property
    def drift(self):
        return math.sqrt(sum(scale**2 for scale, _ in self.components))

    def disagreement(self, chains, lags):
        lags = np.asarray(lags)
        between = sum(
            (scale**2 * persistence**lags for scale, persistence in self.components),
            np.zeros_like(lags, dtype=float),
        )
        variance = 1 + self.noise**2 + self.drift**2
        single = np.arccos((1 + between) / variance) / np.pi
        return (1 - (1 - 2 * single)**chains) / 2


SCENARIOS = {
    "no_drift": Curve(0.10, ()),
    "weak": Curve(0.10, ((0.06, 0.95),)),
    "fast": Curve(0.10, ((0.20, 0.70),)),
    "slow": Curve(0.10, ((0.30, 0.98),)),
    "mixed": Curve(0.10, ((0.14, 0.55), (0.26, 0.985))),
}


@dataclass(frozen=True)
class CalibrationSize:
    seed: int
    cohorts: int
    devices: int
    series: int
    inputs: int
    resamples: int


PROFILES = {
    "smoke": CalibrationSize(381101, 4, 4, 3, 48, 100),
    "main": CalibrationSize(381101, 48, 16, 8, 256, 4000),
    "repeat": CalibrationSize(381102, 32, 16, 8, 256, 4000),
}


def timing_path(curve, lags, shape, rng):
    """Draw Gaussian session offsets at requested times without intermediate steps."""
    components = [
        rng.normal(0, scale, size=shape) for scale, _ in curve.components
    ]
    yield sum(components)
    for step in np.diff(lags):
        for j, (scale, persistence) in enumerate(curve.components):
            across = persistence**int(step)
            innovation = rng.normal(0, scale * math.sqrt(1 - across**2), size=shape)
            components[j] = across * components[j] + innovation
        yield sum(components)


def lag_disagreements(curve, delays, lags, rng):
    """Simulate pooled disagreement per lag for parity prefixes of the same readout."""
    shape = (delays.shape[0], 1, delays.shape[-1])
    requested = set(lags)
    times = tuple(sorted(requested | {0}))
    path = timing_path(curve, times, shape, rng)
    anchor_offset = next(path)
    anchor = read_signals(delays, anchor_offset, rng.normal(0, curve.noise, delays.shape))
    offsets = zip(times, [anchor_offset, *path])
    return np.array([
        (anchor != read_signals(delays, shift, rng.normal(0, curve.noise, delays.shape)))
        .mean(axis=(0, 1))
        for lag, shift in offsets if lag in requested
    ])


def measure_cohort(size, scenario_number, curve, cohort, lags, validation):
    estimates = np.empty((size.devices, len(lags), max(CHAINS)))
    for device in range(size.devices):
        # Training and evaluation hold out complete parameterizations and trajectories.
        rng = stream(size.seed, 61 + int(validation), scenario_number, cohort, device)
        weight = sample_weights(rng, max(CHAINS), 64)
        features = random_features(rng, size.series * size.inputs, 64)
        delays = (features @ weight.T).reshape(size.series, size.inputs, max(CHAINS))
        estimates[device] = lag_disagreements(curve, delays, lags, rng)
    return estimates.mean(axis=0)


def inverse_ar(chains, lag, measurements):
    """Recover one-scale parameters from exact probabilities at three equally spaced lags."""
    d0, d1, d2 = measurements
    if not (0 <= d0 < d1 <= d2 < 0.5) or lag <= 0:
        raise ValueError("drift parameters require strictly increasing disagreement")
    signs = (1 - (1 - 2 * np.array([d0, d1, d2]))**(1 / chains)) / 2
    r0, r1, r2 = np.cos(np.pi * signs)
    decay = (r1 - r2) / (r0 - r1)
    if not 0 <= decay < 1:
        raise ValueError("lag covariance does not identify a stable one-component curve")
    weight = (r0 - r1) / (1 - decay)
    baseline = r0 - weight
    if baseline <= 0 or r0 > 1:
        raise ValueError("invalid variance")
    return Curve(math.sqrt((1 - r0) / baseline),
                 ((math.sqrt(weight / baseline), decay**(1 / lag)),))


def calibration_curve(model, chains, lags, measured):
    """Fit a family of aggregate curves without any device-specific parameters."""
    if model == "single":
        initial = (0.1, 0.18, 0.8)
        bounds = ((0.001, 0, 0), (0.65, 0.9, 0.99999))
    elif model == "two_scale":
        initial = (0.1, 0.1, 0.5, 0.17, 0.97)
        bounds = ((0.001, 0, 0, 0, 0.9), (0.65, 0.9, 0.9, 0.9, 0.99999))
    else:
        raise ValueError(f"unknown curve model: {model}")

    def unpack(parameters):
        return Curve(parameters[0], tuple(zip(parameters[1::2], parameters[2::2])))

    def residual(parameters):
        return unpack(parameters).disagreement(chains, lags) - measured

    fit = least_squares(residual, initial, bounds=bounds, max_nfev=350)
    if not fit.success:
        fit = least_squares(residual, fit.x, bounds=bounds, max_nfev=750)
    return unpack(fit.x), bool(fit.success)


def candidate_estimate(model, chains, lags, observations, test_lags):
    if model == "interpolate":
        # Linear interpolation at observed times, with endpoints held outside the range.
        return observations, np.interp(test_lags, lags, observations), {
            "converged": True, "noise": "", "drift": "", "persistence": "",
        }
    fit, converged = calibration_curve(model, chains, lags, observations)
    return fit.disagreement(chains, lags), fit.disagreement(chains, test_lags), {
        "converged": converged, "noise": fit.noise, "drift": fit.drift,
        "persistence": fit.components[0][1] if model == "single" else "",
    }


def calibrate(size, progress=None, *, plans=PLANS):
    calibration_lags = tuple(sorted({lag for plan in plans.values() for lag in plan}))
    cohort_rows, lag_rows = [], []
    for scene, (name, curve) in enumerate(SCENARIOS.items()):
        for cohort in range(size.cohorts):
            train = measure_cohort(size, scene, curve, cohort, calibration_lags, False)
            test = measure_cohort(size, scene, curve, cohort, TEST_LAGS, True)
            for k in CHAINS:
                measured = test[:, k - 1]
                truth = curve.disagreement(k, TEST_LAGS)
                for plan, lags in plans.items():
                    used = [calibration_lags.index(lag) for lag in lags]
                    observations = train[used, k - 1]
                    for model in ("interpolate", "single", "two_scale"):
                        fitted, prediction, metadata = candidate_estimate(
                            model, k, lags, observations, TEST_LAGS,
                        )
                        record = {"scenario": name, "chains": k, "plan": plan,
                                  "model": model, "cohort": cohort}
                        cohort_rows.append({
                            **record, **metadata, "noise_truth": curve.noise,
                            "drift_truth": curve.drift,
                            "persistence_truth": (
                                curve.components[0][1] if len(curve.components) == 1 else ""
                            ),
                            "fit_mae": float(np.abs(fitted - observations).mean()),
                            "analytic_test_mae": float(np.abs(prediction - truth).mean()),
                            "cohort_test_mae": float(np.abs(prediction - measured).mean()),
                        })
                        lag_rows.extend({
                            **record, "lag": lag, "predicted": float(estimate),
                            "analytic": float(expected), "cohort_observed": float(observed),
                        } for lag, estimate, expected, observed in
                            zip(TEST_LAGS, prediction, truth, measured))
        if progress is not None:
            progress(scene + 1, len(SCENARIOS))
    return cohort_rows, lag_rows


CONDITION = ("scenario", "chains", "plan", "model")


def parameter_mae(group, name):
    if group[0][name] == "" or group[0][f"{name}_truth"] == "":
        return ""
    return float(np.abs(vector(group, name) - vector(group, f"{name}_truth")).mean())


def aggregate(size, cohorts, lags, *, comparisons=COMPARISONS):
    rng = stream(size.seed, 63)
    draws = rng.integers(size.cohorts, size=(size.resamples, size.cohorts))
    metrics, predictions, contrasts = [], [], []
    groups = group_rows(cohorts, *CONDITION)
    for key, group in groups.items():
        error = vector(group, "cohort_test_mae")
        lo, hi = interval(error[draws].mean(axis=1))
        metrics.append({
            **dict(zip(CONDITION, key)), "cohorts": len(group),
            "converged": int(vector(group, "converged").sum()),
            "noise_mae": parameter_mae(group, "noise"),
            "drift_mae": parameter_mae(group, "drift"),
            "persistence_mae": parameter_mae(group, "persistence"),
            "fit_mae": float(vector(group, "fit_mae").mean()),
            "analytic_test_mae": float(vector(group, "analytic_test_mae").mean()),
            "cohort_test_mae": float(error.mean()), "ci_low": lo, "ci_high": hi,
        })

    for key, group in group_rows(lags, *CONDITION, "lag").items():
        predictions.append({
            **dict(zip((*CONDITION, "lag"), key)),
            **{name: float(vector(group, name).mean()) for name in
               ("predicted", "analytic", "cohort_observed")},
        })

    for (scene, k), _ in group_rows(cohorts, "scenario", "chains").items():
        for label, a, b in comparisons:
            left, right = groups[(scene, k, *a)], groups[(scene, k, *b)]
            gain = vector(left, "cohort_test_mae") - vector(right, "cohort_test_mae")
            low, high = interval(gain[draws].mean(axis=1))
            converged = vector(left, "converged") & vector(right, "converged")
            contrasts.append({
                "scenario": scene, "chains": k, "comparison": label,
                "cohort_mae_gain": float(gain.mean()), "ci_low": low, "ci_high": high,
                "converged_pairs": int(converged.sum()),
                "gain_converged_pairs": float(gain[converged].mean()) if converged.any() else "",
                "analytic_mae_gain": float((vector(left, "analytic_test_mae") -
                                            vector(right, "analytic_test_mae")).mean()),
            })
    return {
        "calibration_metrics": metrics, "calibration_predictions": predictions,
        "calibration_contrasts": contrasts, "calibration_by_cohort": cohorts,
    }
