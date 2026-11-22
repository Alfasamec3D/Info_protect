"""Forecast only aggregate next-session BER on previously unseen simulated devices."""

from dataclasses import dataclass
import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .experiment import stream
from .model import random_features, sample_weights
from .summary import interval


FEATURES = (
    "current_ber", "within_session_disagreement", "block_ber_sd",
    "response_abs_bias", "chains",
)
PARTITIONS = ("train", "validation", "test")
REGIMES = {
    "stable": (0.0, 0.0),
    "persistent": (0.22, 0.995),
    "volatile": (0.32, 0.12),
}
CANDIDATES = {
    "ridge": (0.1, 1.0, 10.0, 50.0),
    "boosting": (7, 15, 31),
}


@dataclass(frozen=True)
class Profile:
    seed: int
    train_devices: int
    validation_devices: int
    test_devices: int
    calibration_inputs: int
    next_inputs: int
    bootstrap_resamples: int


PROFILES = {
    "smoke": Profile(481267, 56, 20, 20, 64, 128, 120),
    "main": Profile(481267, 1024, 256, 512, 512, 2048, 4000),
    "repeat": Profile(481268, 1024, 256, 512, 512, 2048, 4000),
}


@dataclass(frozen=True)
class Cohort:
    device_ids: tuple[str, ...]
    features: np.ndarray
    targets: np.ndarray


def parity(delays):
    return np.logical_xor.reduce(delays < 0, axis=-1)


def calibration_features(delays, offset, noise, rng):
    """Take two current-session reads and return session-level summaries."""
    clean_response = parity(delays)
    reads = parity(delays + offset + rng.normal(0, noise, size=(2, *delays.shape)))
    errors = (reads != clean_response).mean(axis=0)
    blocks = errors.reshape(8, -1).mean(axis=1)
    return (
        float(errors.mean()),
        float(np.mean(reads[0] != reads[1])),
        float(blocks.std(ddof=1)),
        float(abs(reads.mean() - 0.5)),
        delays.shape[1],
    )


def future_ber(delays, offset, noise, rng):
    clean_response = parity(delays)
    noisy_response = parity(delays + offset + rng.normal(0, noise, size=delays.shape))
    return float(np.mean(clean_response != noisy_response))


def simulate_device(profile, regime_no, partition_no, device):
    rng = stream(profile.seed, 80, regime_no, partition_no, device)
    drift, persistence = tuple(REGIMES.values())[regime_no]
    chains = int(rng.choice((1, 2, 4)))
    weights = sample_weights(rng, chains, 64)
    noise = rng.uniform(0.04, 0.14)
    offset = rng.normal(0, drift, size=chains)

    present = stream(profile.seed, 81, regime_no, partition_no, device)
    current = random_features(present, profile.calibration_inputs, 64) @ weights.T
    features = calibration_features(current, offset, noise, present)

    future = stream(profile.seed, 82, regime_no, partition_no, device)
    next_offset = persistence * offset + future.normal(
        0, drift * math.sqrt(1 - persistence**2), size=chains,
    )
    next_delays = random_features(future, profile.next_inputs, 64) @ weights.T
    return features, future_ber(next_delays, next_offset, noise, future)


def make_cohort(profile, regime_no, partition_no, count):
    prefix = f"{profile.seed}:{regime_no}:{partition_no}"
    ids = tuple(f"{prefix}:{i}" for i in range(count))
    observations = [simulate_device(profile, regime_no, partition_no, i) for i in range(count)]
    x, y = zip(*observations)
    return Cohort(ids, np.array(x), np.array(y))


def create_model(name, parameter, seed):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=parameter, solver="svd"))
    return HistGradientBoostingRegressor(
        max_leaf_nodes=parameter, max_iter=140, min_samples_leaf=24,
        l2_regularization=0.08, early_stopping=False, random_state=seed,
    )


def predict_rate(model, features):
    return np.clip(model.predict(features), 0, 1)


def train_models(train, validation, seed):
    settings, models = [], {}
    features = np.concatenate((train.features, validation.features))
    targets = np.concatenate((train.targets, validation.targets))
    for name, parameters in CANDIDATES.items():
        scores = []
        for parameter in parameters:
            estimator = create_model(name, parameter, seed)
            estimator.fit(train.features, train.targets)
            prediction = predict_rate(estimator, validation.features)
            scores.append(float(np.abs(prediction - validation.targets).mean()))
        selected = int(np.argmin(scores))
        models[name] = create_model(name, parameters[selected], seed).fit(features, targets)
        settings.extend({
            "model": name, "parameter": parameter, "validation_mae": score,
            "selected": i == selected,
        } for i, (parameter, score) in enumerate(zip(parameters, scores)))
    return models, settings


def compare(regime, cohort, predictions, draws):
    absolute_baseline = np.abs(predictions["baseline"] - cohort.targets)
    rows = []
    for name, prediction in predictions.items():
        errors = prediction - cohort.targets
        absolute = np.abs(errors)
        lows, highs = interval(absolute[draws].mean(axis=1))
        squared = errors**2
        rmse_lo, rmse_hi = interval(np.sqrt(squared[draws].mean(axis=1)))
        gain = absolute_baseline - absolute
        gain_lo, gain_hi = interval(gain[draws].mean(axis=1))
        rows.append({
            "regime": regime, "model": name, "devices": len(cohort.device_ids),
            "mae": float(absolute.mean()), "mae_ci_low": lows, "mae_ci_high": highs,
            "rmse": float(np.sqrt(squared.mean())),
            "rmse_ci_low": rmse_lo, "rmse_ci_high": rmse_hi,
            "bias": float(errors.mean()), "mae_gain": float(gain.mean()),
            "gain_ci_low": gain_lo, "gain_ci_high": gain_hi,
        })
    return rows


def estimate(models, cohort):
    return {
        "baseline": cohort.features[:, 0],
        **{name: predict_rate(model, cohort.features) for name, model in models.items()},
    }


def evaluate_transfer(models_by_regime, tests):
    """Evaluate frozen estimators on held-out devices under different conditions."""
    metrics, per_device = [], []
    for source, models in models_by_regime.items():
        for target, (cohort, draws) in tests.items():
            if source == target:
                continue
            predictions = estimate(models, cohort)
            metrics.extend(
                {"source_regime": source, **row}
                for row in compare(target, cohort, predictions, draws)
            )
            per_device.extend({
                "source_regime": source, "regime": target, "device_id": device,
                **{name: float(values[i]) for name, values in predictions.items()},
            } for i, device in enumerate(cohort.device_ids))
    return metrics, per_device


def run(profile, progress=None):
    metrics, predictions, partitions, validation = [], [], [], []
    models_by_regime, tests = {}, {}
    sizes = (profile.train_devices, profile.validation_devices, profile.test_devices)
    with threadpool_limits(limits=1):
        for regime_no, regime in enumerate(REGIMES):
            cohorts = tuple(make_cohort(profile, regime_no, i, count)
                            for i, count in enumerate(sizes))
            for name, cohort in zip(PARTITIONS, cohorts):
                partitions.extend({"device_id": ident, "regime": regime, "split": name}
                                  for ident in cohort.device_ids)
            train, valid, test = cohorts
            models, settings = train_models(train, valid, profile.seed)
            validation.extend({"regime": regime, **row} for row in settings)
            estimates = estimate(models, test)
            rng = stream(profile.seed, 83, regime_no)
            draws = rng.integers(sizes[-1], size=(profile.bootstrap_resamples, sizes[-1]))
            metrics.extend(compare(regime, test, estimates, draws))
            models_by_regime[regime] = models
            tests[regime] = (test, draws)
            predictions.extend({
                "device_id": ident, "regime": regime,
                **dict(zip(FEATURES, map(float, test.features[i]))),
                "next_ber": float(test.targets[i]),
                **{name: float(prediction[i]) for name, prediction in estimates.items()},
            } for i, ident in enumerate(test.device_ids))
            if progress is not None:
                progress(regime)
        transfer_metrics, transfer_predictions = evaluate_transfer(models_by_regime, tests)
    return {"metrics": metrics, "device_predictions": predictions,
            "partition_audit": partitions, "validation": validation,
            "transfer_metrics": transfer_metrics, "transfer_predictions": transfer_predictions}
