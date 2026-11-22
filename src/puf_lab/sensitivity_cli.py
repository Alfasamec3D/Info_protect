"""Run and plot sensitivity to chain dependence and persistent drift."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

from .cli import write_csv
from .plots import PALETTE, save, style
from .sensitivity import (
    BUDGET, DRIFT_LEVELS, PERSISTENCE_LEVELS, PROFILES, READ_NOISE, REPEATS,
    SPATIAL_CORRELATIONS, TEMPORAL_CHAINS, spatial_study, temporal_study,
)
from .summary import group_rows

import matplotlib.pyplot as plt
import numpy as np


def graph(spatial, temporal, output):
    fig, axes = plt.subplots(1, 3, figsize=(12.7, 4.25), layout="constrained")
    spatial_ax, gap_ax, gain_ax = axes
    for i, k in enumerate((1, 2, 4, 6)):
        rows = [r for r in spatial if r["chains"] == k]
        x = [r["correlation"] for r in rows]
        y = np.array([r["fraction_ones"] for r in rows])
        low = [r["ci_low"] for r in rows]
        high = [r["ci_high"] for r in rows]
        spatial_ax.plot(x, [r["theory_fraction_ones"] for r in rows],
                        "--", c=PALETTE[i], alpha=0.75)
        spatial_ax.errorbar(x, y, yerr=np.maximum(0, [y - low, high - y]),
                            fmt="o", c=PALETTE[i], capsize=2, label=f"k = {k}")
    spatial_ax.set(xlabel="Корреляция задержек ρ", title="Проверка баланса ответа")
    style(spatial_ax, "Доля единиц XOR")
    spatial_ax.legend(frameon=False, fontsize=8, ncols=2)

    focus = [r for r in temporal if r["drift"] == 0.3 and r["chains"] == 4]
    for axis, reads, metric, lo, hi, label, title in (
        (gap_ax, 1, "held_minus_same", "gap_ci_low", "gap_ci_high",
         "Разница долей расхождений", "Проверка в другой сессии"),
        (gain_ax, 9, "rmse_gain", "rmse_gain_ci_low", "rmse_gain_ci_high",
         "Снижение RMSE (доля)", "Распределённый эталон"),
    ):
        rows = group_rows([r for r in focus if r["reads"] == reads], "persistence")
        ordered = [items[0] for items in rows.values()]
        x = [r["persistence"] for r in ordered]
        y = np.array([r[metric] for r in ordered])
        low = [r[lo] for r in ordered]
        high = [r[hi] for r in ordered]
        axis.axhline(0, color="#647680", linewidth=1, linestyle="--")
        axis.errorbar(x, y, yerr=np.maximum(0, [y - low, high - y]), fmt="-o",
                      capsize=3, c=PALETTE[0])
        axis.set(xlabel="Сохранение дрейфа между соседними сессиями η", title=title)
        style(axis, label)
    save(fig, output, "sensitivity")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    size = PROFILES[args.profile]

    def progress(done, total):
        print(f"Temporal sensitivity: {done}/{total}", file=sys.stderr, flush=True)

    spatial = spatial_study(size)
    temporal = temporal_study(size, None if args.quiet else progress)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "spatial_bias.csv", spatial)
    write_csv(args.output / "temporal_persistence.csv", temporal)
    metadata = {
        "profile": args.profile, **asdict(size), "budget": BUDGET,
        "read_noise": READ_NOISE, "drift": DRIFT_LEVELS, "repeats": REPEATS,
        "temporal_chains": TEMPORAL_CHAINS, "persistence": PERSISTENCE_LEVELS,
        "spatial_correlations": SPATIAL_CORRELATIONS, "verification_lag": REPEATS[-1],
    }
    (args.output / "sensitivity_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    graph(spatial, temporal, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
