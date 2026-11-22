"""Aggregate results and bootstrap independent device blocks."""

from collections import defaultdict

import numpy as np

from .experiment import stream
from .model import population_error


def group_rows(rows, *keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return dict(sorted(groups.items()))


def vector(rows, name):
    return np.array([row[name] for row in rows])


def interval(estimates):
    return tuple(float(value) for value in np.quantile(estimates, [0.025, 0.975]))


def characterize(rows, pairs, device_draws, pair_draws):
    reliability = []
    for (k, noise), values in group_rows(rows, "chains", "noise").items():
        ber = vector(values, "ber")
        uniformity = vector(values, "uniformity")
        low, high = interval(ber[device_draws].mean(axis=1))
        reliability.append({
            "chains": k, "noise": noise, "devices": len(values),
            "ber": float(ber.mean()), "ber_ci_low": low, "ber_ci_high": high,
            "theory_ber": population_error(k, noise),
            "uniformity": float(uniformity.mean()),
            "majority_accuracy": float(vector(values, "majority_accuracy").mean()),
            "noisy_read_ceiling": float(1 - ber.mean()),
            "ceiling_ci_low": 1 - high, "ceiling_ci_high": 1 - low,
            "mean_abs_bias": float(np.abs(uniformity - 0.5).mean()),
            "mean_reference_se": float(vector(values, "reference_se").mean()),
        })
    uniqueness = []
    for (k,), values in group_rows(pairs, "chains").items():
        distance = vector(values, "hamming")
        low, high = interval(distance[pair_draws].mean(axis=1))
        uniqueness.append({
            "chains": k, "disjoint_pairs": len(values), "hamming": float(distance.mean()),
            "hamming_ci_low": low, "hamming_ci_high": high,
            "hamming_min": float(distance.min()), "hamming_max": float(distance.max()),
        })
    return reliability, uniqueness


def budget_summary(rows, draws):
    summary, winners = [], []
    groups = group_rows(rows, "chains", "noise", "budget")
    for (k, noise, budget), group in groups.items():
        strategies = group_rows(group, "enrollment_reads")
        comparisons = []
        for (repeats,), values in strategies.items():
            mse = vector(values, "analytic_mse")
            boot_rmse = np.sqrt(mse[draws].mean(axis=1))
            low, high = interval(boot_rmse)
            comparisons.append(({
                "chains": k, "noise": noise, "budget": budget,
                "enrollment_reads": repeats, "challenges": values[0]["challenges"],
                "target_ber": float(vector(values, "target_ber").mean()),
                "analytic_bias": float(vector(values, "analytic_bias").mean()),
                "analytic_rmse": float(np.sqrt(mse.mean())),
                "analytic_rmse_ci_low": low, "analytic_rmse_ci_high": high,
                "simulated_bias": float(vector(values, "simulated_bias").mean()),
                "simulated_rmse": float(np.sqrt(vector(values, "simulated_mse").mean())),
            }, boot_rmse))
        baseline, baseline_boot = comparisons[0]
        for row, boot in comparisons:
            row["rmse_delta_vs_r1"] = row["analytic_rmse"] - baseline["analytic_rmse"]
            row["delta_ci_low"], row["delta_ci_high"] = interval(boot - baseline_boot)
            summary.append(row)
        ordered = sorted(comparisons, key=lambda item: item[0]["analytic_rmse"])
        best, best_boot = ordered[0]
        winner = {
            "chains": k, "noise": noise, "budget": budget,
            "best_reads": best["enrollment_reads"], "best_rmse": best["analytic_rmse"],
            "baseline_rmse": baseline["analytic_rmse"],
            "rmse_gain_vs_r1": -best["rmse_delta_vs_r1"],
            "gain_ci_low": -best["delta_ci_high"], "gain_ci_high": -best["delta_ci_low"],
        }
        if len(ordered) > 1:
            second, second_boot = ordered[1]
            winner["next_reads"] = second["enrollment_reads"]
            winner["next_rmse_gap"] = second["analytic_rmse"] - best["analytic_rmse"]
            winner["gap_ci_low"], winner["gap_ci_high"] = interval(second_boot - best_boot)
        winners.append(winner)
    return summary, winners


def summarize(config, raw):
    rng = stream(config.seed, 5)
    size = (config.bootstrap_resamples, config.devices)
    device_draws = rng.integers(config.devices, size=size)
    pairs = config.devices // 2
    pair_draws = rng.integers(pairs, size=(config.bootstrap_resamples, pairs))
    devices, pair_rows, budgets = raw
    reliability, uniqueness = characterize(devices, pair_rows, device_draws, pair_draws)
    budget_metrics, winners = budget_summary(budgets, device_draws)
    return {
        "reliability": reliability, "uniqueness": uniqueness,
        "budget_metrics": budget_metrics, "budget_winners": winners,
        "device_metrics": devices, "pair_metrics": pair_rows, "budget_by_device": budgets,
    }
