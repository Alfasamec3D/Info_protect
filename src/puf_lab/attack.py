"""CRP-only modeling attack on our simulated additive-delay PUFs.

Learners receive public challenge features and final XOR labels only.
Hidden delays, component labels and clean responses are evaluation-only.
"""

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .experiment import stream
from .model import challenge_features, read_error, sample_weights

MODELS = ("majority", "linear", "product", "shuffled")


@dataclass(frozen=True)
class Profile:
    seed: int = 729301
    devices: int = 12
    bits: int = 64
    budgets: tuple = (256, 1024, 4096)
    chains: tuple = (1, 2, 4)
    noises: tuple = (0., .1)
    validation: int = 512
    test: int = 2048
    restarts: int = 3
    maxiter: int = 250
    bootstrap: int = 4000


PROFILES = {
    "main": Profile(),
    "repeat": Profile(seed=729302),
    "smoke": Profile(devices=2, bits=16, budgets=(64, 256), validation=64,
                     test=256, restarts=2, maxiter=80, bootstrap=200),
}


def sample_device(profile, device):
    """Disjoint unique inputs, nested training budgets and fixed device weights."""
    rng = stream(profile.seed, 90, device)
    count = max(profile.budgets) + profile.validation + profile.test
    if count > 2 ** profile.bits:
        raise ValueError("More unique challenges requested than available")
    challenges, seen = [], set()
    while len(challenges) < count:
        row = rng.integers(0, 2, profile.bits, dtype=np.uint8)
        key = np.packbits(row).tobytes()
        if key not in seen:
            seen.add(key)
            challenges.append(row)
    raw = np.array(challenges)
    features = challenge_features(raw)
    weights = sample_weights(stream(profile.seed, 91, device), max(profile.chains), profile.bits)
    stop = max(profile.budgets)
    slices = (slice(0, stop), slice(stop, stop + profile.validation),
              slice(stop + profile.validation, None))
    return raw, features, weights, slices


def objective(flat, x, target_sign, chains, penalty=1e-4):
    """Regularized logistic loss of the product of chain delay estimates."""
    weights = flat.reshape(chains, x.shape[1])
    delays = x @ weights.T
    score = np.prod(delays, axis=1)
    loss = np.logaddexp(0., -target_sign * score).mean() + penalty * np.sum(weights**2) / 2
    derivative = -target_sign * expit(-target_sign * score) / len(x)
    gradient = np.empty_like(weights)
    for k in range(chains):
        # No division by a delay: the derivative stays defined at zero.
        others = np.prod(np.delete(delays, k, axis=1), axis=1)
        gradient[k] = x.T @ (derivative * others) + penalty * weights[k]
    return float(loss), gradient.ravel()


def fit_product(x, y, valid_x, valid_y, chains, rng, restarts, maxiter):
    candidates, best, best_accuracy = [], None, -1.
    for restart in range(restarts):
        initial = rng.normal(0, 1 / np.sqrt(x.shape[1]), (chains, x.shape[1]))
        fitted = minimize(objective, initial.ravel(), args=(x, 1 - 2 * y.astype(float), chains),
                          jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-9})
        weights = fitted.x.reshape(chains, x.shape[1])
        accuracy = float(np.mean(predict_product(valid_x, weights) == valid_y))
        candidates.append({"restart": restart, "validation_accuracy": accuracy,
                           "converged": bool(fitted.success), "iterations": int(fitted.nit),
                           "loss": float(fitted.fun)})
        if accuracy > best_accuracy:
            best, best_accuracy = weights, accuracy
    chosen = int(np.argmax([r["validation_accuracy"] for r in candidates]))
    for i, row in enumerate(candidates):
        row["selected"] = i == chosen
    return best, candidates


def predict_product(x, weights):
    return np.prod(x @ weights.T, axis=1) < 0


def fit_linear(x, y, test_x):
    if len(np.unique(y)) < 2:
        return np.full(len(test_x), y[0], dtype=bool)
    return LogisticRegression(C=1., fit_intercept=False, max_iter=1000,
                              solver="lbfgs", tol=1e-7).fit(x, y).predict(test_x)


def summarize(rows, profile):
    metrics = []
    for chains in profile.chains:
        for noise in profile.noises:
            for budget in profile.budgets:
                subset = [r for r in rows if (r["chains"], r["noise"], r["train_crps"])
                          == (chains, noise, budget)]
                baseline = np.array([r["accuracy"] for r in subset if r["model"] == "majority"])
                draws = stream(profile.seed, 95, chains, int(noise * 1000), budget).integers(
                    profile.devices, size=(profile.bootstrap, profile.devices))
                for model in MODELS:
                    records = [r for r in subset if r["model"] == model]
                    accuracy = np.array([r["accuracy"] for r in records])
                    gain = accuracy - baseline
                    low, high = np.quantile(accuracy[draws].mean(axis=1), [.025, .975])
                    gl, gh = np.quantile(gain[draws].mean(axis=1), [.025, .975])
                    metrics.append({"chains": chains, "noise": noise, "train_crps": budget,
                        "validation_crps": profile.validation if model in ("product", "shuffled") else 0,
                        "model": model, "devices": len(records), "accuracy": float(accuracy.mean()),
                        "ci_low": float(low), "ci_high": float(high), "gain": float(gain.mean()),
                        "gain_ci_low": float(gl), "gain_ci_high": float(gh),
                        **{key: float(np.mean([r[key] for r in records])) for key in
                           ("clean_accuracy", "balanced_accuracy", "oracle_accuracy")}})
    return metrics


def run(profile, progress=None):
    rows, audit, validation, arrays = [], [], [], {}
    with threadpool_limits(limits=1):
        for device in range(profile.devices):
            raw, x, weights, slices = sample_device(profile, device)
            train_slice, valid_slice, test_slice = slices
            ident = f"{profile.seed}:{device}"
            for name, part in zip(("train", "validation", "test"), slices):
                audit.append({"device_id": ident, "split": name, "count": len(raw[part]),
                              "challenge_sha256": hashlib.sha256(raw[part].tobytes()).hexdigest(),
                              "overlap": 0})
            arrays[f"d{device}_challenges"] = raw
            for chains in profile.chains:
                delays = x @ weights[:chains].T
                clean = np.logical_xor.reduce(delays < 0, axis=1)
                for noise_no, noise in enumerate(profile.noises):
                    noisy = delays + stream(profile.seed, 92, device, chains, noise_no).normal(
                        0, noise, delays.shape)
                    y = np.logical_xor.reduce(noisy < 0, axis=1)
                    truth = y[test_slice]
                    oracle = float((1 - read_error(delays[test_slice], noise)[:, -1]).mean())
                    prefix = f"d{device}_k{chains}_n{noise_no}"
                    arrays[f"{prefix}_labels"] = y
                    arrays[f"{prefix}_clean"] = clean[test_slice]
                    for budget in profile.budgets:
                        train_x, train_y = x[:budget], y[:budget]
                        fitted, candidates = fit_product(
                            train_x, train_y, x[valid_slice], y[valid_slice], chains,
                            stream(profile.seed, 93, device, chains, noise_no, budget),
                            profile.restarts, profile.maxiter)
                        shuffled = stream(profile.seed, 94, device, chains, noise_no, budget)
                        # Also shuffle validation labels: the negative control must not
                        # select restarts using the real challenge-label relationship.
                        null, null_candidates = fit_product(
                            train_x, shuffled.permutation(train_y), x[valid_slice],
                            shuffled.permutation(y[valid_slice]), chains, shuffled,
                            profile.restarts, profile.maxiter)
                        predictions = {
                            "majority": np.full(len(truth), train_y.mean() >= .5),
                            "linear": fit_linear(train_x, train_y, x[test_slice]),
                            "product": predict_product(x[test_slice], fitted),
                            "shuffled": predict_product(x[test_slice], null),
                        }
                        common = {"device_id": ident, "chains": chains, "noise": noise,
                                  "train_crps": budget}
                        for model, attempts in (("product", candidates), ("shuffled", null_candidates)):
                            validation.extend({**common, "model": model, **r} for r in attempts)
                        for model, pred in predictions.items():
                            correct = pred == truth
                            balanced = np.mean([correct[truth == b].mean() for b in (False, True)
                                                if np.any(truth == b)])
                            rows.append({**common, "model": model, "test_crps": len(truth),
                                         "accuracy": float(correct.mean()),
                                         "clean_accuracy": float(np.mean(pred == clean[test_slice])),
                                         "balanced_accuracy": float(balanced),
                                         "oracle_accuracy": oracle})
                            arrays[f"{prefix}_b{budget}_{model}"] = pred
            if progress:
                progress(device + 1)
    return {"per_device": rows, "metrics": summarize(rows, profile),
            "partition_audit": audit, "validation": validation}, arrays
