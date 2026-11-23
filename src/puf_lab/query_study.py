"""Fixed physical-query budgets, correlated repeats, and an MLP comparator."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
import sklearn

from .attack import Profile, fit_product, predict_product, sample_device
from .cli import write_csv
from .experiment import stream
from .model import read_error

PROFILES = {
    "main": Profile(seed=829301, devices=5, chains=(1, 2, 3, 4), budgets=(4608, 18432),
                    noises=(0., .25), restarts=2, maxiter=180),
    "repeat": Profile(seed=829302, devices=5, chains=(1, 2, 3, 4), budgets=(4608, 18432),
                      noises=(0., .25), restarts=2, maxiter=180),
    "smoke": Profile(seed=829301, devices=2, bits=16, chains=(1, 2, 3, 4), budgets=(144, 288),
                     noises=(0., .25), validation=32, test=64, restarts=1, maxiter=30, bootstrap=200),
}
GROUP = ("kind", "chains", "noise", "budget", "correlation", "reads", "model")


def repeated_noise(rng, count, chains, correlation, repeats=9):
    """Unit Gaussian equicorrelated errors; the first read is invariant in rho.

    For j>0: eps_j = rho*eps_0 + sqrt(rho*(1-rho))*g + sqrt(1-rho)*z_j.
    This gives variance 1 and covariance rho for every distinct pair.
    """
    first = rng.normal(size=(count, chains))
    common = rng.normal(size=(count, chains))
    other = rng.normal(size=(repeats - 1, count, chains))
    rest = (correlation * first + np.sqrt(correlation * (1 - correlation)) * common
            + np.sqrt(1 - correlation) * other)
    return np.concatenate((first[None], rest), axis=0)


def majority_labels(delays, noise, perturbations):
    """Vote on final response BITS, obtained from signs of perturbed delays."""
    responses = np.logical_xor.reduce(delays + noise * perturbations < 0, axis=-1)
    return responses.sum(axis=0) > len(perturbations) // 2


def stored_mlp_predict(x, coefs, intercepts, classes=(False, True)):
    if len(classes) == 1:
        return np.full(len(x), classes[0], dtype=bool)
    hidden = x[:, :-1]
    for coef, bias in zip(coefs[:-1], intercepts[:-1]):
        hidden = np.tanh(hidden @ coef + bias)
    return np.asarray(classes)[(expit(hidden @ coefs[-1] + intercepts[-1]).ravel() > .5).astype(int)]


def fit_mlp(x, y, vx, vy, rng, restarts, epochs):
    candidates, fitted = [], []
    for restart in range(restarts):
        model = MLPClassifier(hidden_layer_sizes=(64, 32), activation="tanh", solver="adam",
                              alpha=.001, batch_size=min(1024, len(x)), learning_rate_init=.001,
                              max_iter=epochs, early_stopping=False, n_iter_no_change=epochs + 1,
                              random_state=int(rng.integers(2**31 - 1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x[:, :-1], y)
        accuracy = float(np.mean(model.predict(vx[:, :-1]) == vy))
        candidates.append({"restart": restart, "validation_accuracy": accuracy,
                           "converged": model.n_iter_ < epochs, "iterations": model.n_iter_,
                           "loss": float(model.loss_)})
        fitted.append(model)
    chosen = int(np.argmax([r["validation_accuracy"] for r in candidates]))
    for i, row in enumerate(candidates):
        row["selected"] = i == chosen
    return fitted[chosen], candidates


def summarize(rows, profile):
    metrics = []
    keys = sorted({tuple(r[k] for k in GROUP) for r in rows})
    for number, key in enumerate(keys):
        subset = [r for r in rows if tuple(r[k] for k in GROUP) == key]
        draws = stream(profile.seed, 106, number).integers(len(subset), size=(profile.bootstrap, len(subset)))
        row = dict(zip(GROUP, key)) | {"devices": len(subset), "bootstrap_stream": number}
        for field in ("accuracy", "clean_accuracy", "gain_over_single", "train_label_error"):
            values = np.array([r[field] for r in subset])
            low, high = np.quantile(values[draws].mean(axis=1), [.025, .975])
            row.update({field: float(values.mean()), field + "_low": float(low), field + "_high": float(high)})
        row.update({k: float(np.mean([r[k] for r in subset]))
                    for k in ("fit_seconds", "oracle_accuracy", "majority_accuracy")})
        metrics.append(row)
    return metrics


def interactions(rows, profile):
    """Paired effect of repeat correlation, conditional on architecture and budget."""
    output = []
    for chains in (1, 2):
        for budget in profile.budgets:
            for reads in (3, 9):
                groups = [[r for r in rows if r["kind"] == "budget" and r["chains"] == chains
                           and r["budget"] == budget and r["reads"] == reads and r["correlation"] == rho]
                          for rho in (0., .9)]
                assert [r["device_id"] for r in groups[0]] == [r["device_id"] for r in groups[1]]
                delta = np.array([b["accuracy"] - a["accuracy"] for a, b in zip(*groups)])
                draws = stream(profile.seed, 107, chains, budget, reads).integers(
                    profile.devices, size=(profile.bootstrap, profile.devices))
                low, high = np.quantile(delta[draws].mean(axis=1), [.025, .975])
                output.append({"chains": chains, "budget": budget, "reads": reads,
                               "correlated_minus_iid": float(delta.mean()), "ci_low": float(low), "ci_high": float(high)})
    return output


def run(profile, output, progress=print, device_indices=None, collect_only=False):
    output.mkdir(parents=True, exist_ok=True)
    rows, selection, audits, thresholds, arrays, models = [], [], [], [], {}, {}
    epochs = 25 if profile.test == 64 else 40
    with threadpool_limits(limits=1):
        for device in (range(profile.devices) if device_indices is None else device_indices):
            first_record = len(rows)
            raw, x, weights, (tr, va, te) = sample_device(profile, device)
            ident = f"{profile.seed}:{device}"
            arrays[f"d{device}_challenges"] = raw
            for name, part in (("train", tr), ("validation", va), ("test", te)):
                audits.append({"device_id": ident, "split": name, "count": len(raw[part]),
                               "sha256": hashlib.sha256(raw[part].tobytes()).hexdigest()})
            for chains in profile.chains:
                delays = x @ weights[:chains].T
                clean = np.logical_xor.reduce(delays < 0, axis=1)
                for noise_no, noise in enumerate(profile.noises):
                    test_rng = stream(profile.seed, 100, device, chains, noise_no)
                    observed = np.logical_xor.reduce(delays + test_rng.normal(0, noise, delays.shape) < 0, axis=1)
                    for budget in profile.budgets:
                        base_accuracy = None
                        conditions = [("benchmark", 0., 1, "product"), ("benchmark", 0., 1, "mlp")]
                        if chains <= 2 and noise > 0:
                            conditions += [("budget", rho, r, "product") for rho in (0., .9) for r in (3, 9)]
                        for kind, rho, repeats, method in conditions:
                            count = budget // repeats
                            assert count * repeats == budget
                            perturbations = repeated_noise(stream(profile.seed, 101, device, chains, noise_no),
                                                           max(profile.budgets), chains, rho)
                            labels = majority_labels(delays[:count], noise, perturbations[:repeats, :count])
                            if noise == 0:
                                np.testing.assert_array_equal(labels, clean[:count])
                            rng = stream(profile.seed, 102, device, chains, noise_no, budget, repeats)
                            start = perf_counter()
                            if method == "product":
                                model, attempts = fit_product(x[:count], labels, x[va], observed[va], chains,
                                                              rng, profile.restarts, profile.maxiter)
                                pred = predict_product(x[te], model)
                            else:
                                model, attempts = fit_mlp(x[:count], labels, x[va], observed[va], rng,
                                                          profile.restarts, epochs)
                                pred = model.predict(x[te, :-1])
                            duration = perf_counter() - start
                            record_id = f"d{device}_r{len(rows) - first_record}"
                            if method == "product":
                                models[f"{record_id}_weights"] = model
                            else:
                                models[f"{record_id}_classes"] = model.classes_
                                for layer, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
                                    models[f"{record_id}_coef{layer}"] = coef
                                    models[f"{record_id}_bias{layer}"] = intercept
                                np.testing.assert_array_equal(pred, stored_mlp_predict(x[te], model.coefs_, model.intercepts_, model.classes_))
                            accuracy = float(np.mean(pred == observed[te]))
                            if kind == "benchmark" and method == "product":
                                base_accuracy = accuracy
                            arrays[f"{record_id}_prediction"] = pred
                            arrays[f"{record_id}_train_labels"] = labels
                            arrays[f"{record_id}_test_labels"] = observed[te]
                            arrays[f"{record_id}_clean"] = clean[te]
                            record = {"record_id": record_id, "device_id": ident, "device": device,
                                "kind": kind, "chains": chains, "noise": noise, "budget": budget,
                                "correlation": rho, "reads": repeats, "unique_challenges": count,
                                "validation_queries": profile.validation, "total_queries": budget + profile.validation,
                                "model": method, "accuracy": accuracy,
                                "clean_accuracy": float(np.mean(pred == clean[te])),
                                "majority_accuracy": float(np.mean((labels.mean() >= .5) == observed[te])),
                                "gain_over_single": accuracy - base_accuracy,
                                "train_label_error": float(np.mean(labels != clean[:count])),
                                "oracle_accuracy": float((1 - read_error(delays[te], noise)[:, -1]).mean()),
                                "fit_seconds": duration}
                            rows.append(record)
                            selection.extend({"record_id": record_id, **r} for r in attempts)
            for chains in profile.chains:
                for noise in profile.noises:
                    for method in ("product", "mlp"):
                        candidates = [r for r in rows if r["device_id"] == ident and r["kind"] == "benchmark"
                                      and r["chains"] == chains and r["noise"] == noise and r["model"] == method]
                        reached = [r["total_queries"] for r in candidates if r["accuracy"] >= .95]
                        thresholds.append({"device_id": ident, "chains": chains, "noise": noise, "model": method,
                            "first_tested_total_queries_at_95": min(reached) if reached else "not_reached",
                            "largest_tested_total_queries": max(r["total_queries"] for r in candidates)})
            progress(f"Query study seed {profile.seed}: device {device + 1}/{profile.devices}")
    if collect_only:
        return rows, selection, audits, thresholds, arrays, models
    save_results(profile, output, rows, selection, audits, thresholds, arrays, models, epochs)


def save_results(profile, output, rows, selection, audits, thresholds, arrays, models, epochs, workers=1):
    for name, table in (("per_device", rows), ("metrics", summarize(rows, profile)),
                        ("interactions", interactions(rows, profile)),
                        ("validation", selection), ("partition_audit", audits), ("thresholds", thresholds)):
        write_csv(output / f"{name}.csv", table)
    np.savez_compressed(output / "predictions.npz", **arrays)
    np.savez_compressed(output / "models.npz", **models)
    manifest = {**asdict(profile), "repeat_counts": [1, 3, 9], "repeat_correlations": [0., .9],
                "research_question": "distinct challenges versus majority repeats at equal physical-query budget",
                "correlation_model": "equicorrelated Gaussian read noise conditional on challenge and chain; independent across challenges",
                "validation": profile.validation, "test_queries_excluded_from_attacker_budget": profile.test,
                "mlp": {"hidden": [64, 32], "activation": "tanh", "epochs": epochs, "batch_size": 1024, "alpha": .001},
                "time_scope": "wall time of all restarts, validation selection and test prediction; excludes generation",
                "threshold_scope": "first observed grid point reaching 95%, not a certified minimum or confidence bound",
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
                "platform": platform.platform(), "machine": platform.machine(), "workers": workers}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    plot(summarize(rows, profile), output)


def device_worker(arguments):
    profile, output, device = arguments
    return run(profile, output, lambda message: print(message, flush=True), (device,), True)


def run_parallel(profile, output, workers):
    if workers == 1:
        return run(profile, output, lambda message: print(message, flush=True))
    rows, selection, audits, thresholds, arrays, models = [], [], [], [], {}, {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(device_worker, ((profile, output, i) for i in range(profile.devices))):
            for destination, source in zip((rows, selection, audits, thresholds), part[:4]):
                destination.extend(source)
            for destination, source in zip((arrays, models), part[4:]):
                assert not set(destination).intersection(source)
                destination.update(source)
    save_results(profile, output, rows, selection, audits, thresholds, arrays, models,
                 25 if profile.test == 64 else 40, workers)


def plot(metrics, output):
    from .plots import PALETTE, save, style
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), layout="constrained")
    budgets = sorted({r["budget"] for r in metrics})
    for ax, chains in zip(axes, (1, 2)):
        for color, (budget, rho) in zip(PALETTE, ((b, r) for b in budgets for r in (0., .9))):
            subset = [r for r in metrics if r["kind"] == "budget" and r["chains"] == chains
                      and r["budget"] == budget and r["correlation"] == rho]
            subset.sort(key=lambda r: r["reads"])
            ax.errorbar([1, *[r["reads"] for r in subset]], [0, *[r["gain_over_single"] for r in subset]],
                        yerr=[[0, *[max(0, r["gain_over_single"]-r["gain_over_single_low"]) for r in subset]],
                              [0, *[max(0, r["gain_over_single_high"]-r["gain_over_single"]) for r in subset]]],
                        marker="o", color=color, capsize=3, label=f"B={budget}, ρ={rho}")
        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.set(title=f"{chains}-XOR, σ = 0,25", xlabel="Чтений каждого запроса", xticks=[1,3,9])
        style(ax, "Изменение точности относительно одного чтения")
        ax.legend(frameon=False, fontsize=8)
    save(fig, output, "query_budget")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), layout="constrained")
    for ax, noise in zip(axes, (0., .25)):
        for color, (method, budget) in zip(PALETTE, ((m,b) for m in ("product","mlp") for b in budgets)):
            subset = sorted((r for r in metrics if r["kind"] == "benchmark" and r["noise"] == noise
                             and r["model"] == method and r["budget"] == budget), key=lambda r:r["chains"])
            ax.plot([r["chains"] for r in subset], [r["accuracy"] for r in subset], "o-", color=color,
                    label=f"{'Произведение' if method=='product' else 'MLP'}, B={budget}")
        ax.set(title=f"σ = {noise}", xlabel="Число XOR-цепочек", xticks=[1,2,3,4], ylim=(.4,1.01))
        style(ax,"Точность нового ответа")
        ax.legend(frameon=False, fontsize=8)
    save(fig, output, "architecture_comparison")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1, help="parallel independent devices")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    run_parallel(PROFILES[args.profile], args.output, args.workers)


if __name__ == "__main__":
    main()
