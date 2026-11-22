"""Run and plot the aggregate BER forecasting experiment."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .cli import write_csv
from .forecast import FEATURES, PROFILES, REGIMES, run
from .plots import PALETTE, save, style

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import sklearn


def plot(rows, destination):
    x = np.arange(len(REGIMES))
    labels = ("Без дрейфа", "Устойчивый дрейф", "Изменчивый дрейф")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), layout="constrained")
    for axis, field, low_key, high_key in (
        (axes[0], "mae", "mae_ci_low", "mae_ci_high"),
        (axes[1], "mae_gain", "gain_ci_low", "gain_ci_high"),
    ):
        for i, (model, label) in enumerate((
            ("baseline", "Текущий BER"), ("ridge", "Ridge"), ("boosting", "Бустинг"),
        )):
            if field == "mae_gain" and model == "baseline":
                continue
            ordered = [next(r for r in rows if r["regime"] == regime and r["model"] == model)
                       for regime in REGIMES]
            values = np.array([r[field] * 100 for r in ordered])
            lo = [r[low_key] * 100 for r in ordered]
            hi = [r[high_key] * 100 for r in ordered]
            axis.errorbar(x + (i - 1) * 0.13, values, yerr=np.maximum(0, [values - lo, hi - values]),
                          fmt="o", c=PALETTE[i], capsize=3, label=label)
        axis.set_xticks(x, labels, rotation=13)
        style(axis, "MAE, п. п." if field == "mae" else "Снижение MAE, п. п.")
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        axis.legend(frameon=False, fontsize=9)
    axes[0].set_title("Ошибка прогноза следующей сессии")
    axes[1].axhline(0, color="#758994", linestyle="--", linewidth=1)
    axes[1].set_title("Парный выигрыш по экземплярам")
    save(fig, destination, "forecast")


def plot_transfer(local, cross, destination):
    regimes = tuple(REGIMES)
    labels = ("Без дрейфа", "Устойчивый", "Изменчивый")
    by_pair = {(row["source_regime"], row["regime"], row["model"]): row
               for row in cross}
    by_pair.update({(row["regime"], row["regime"], row["model"]): row for row in local})
    limit = max(abs(row["mae_gain"] * 100) for row in by_pair.values()) or 1
    fig, axes = plt.subplots(1, 2, figsize=(10.1, 4.6), layout="constrained")
    for axis, model, title in zip(axes, ("ridge", "boosting"), ("Ridge", "Бустинг")):
        cells = [[by_pair[src, dst, model] for dst in regimes] for src in regimes]
        gains = np.array([[row["mae_gain"] * 100 for row in line] for line in cells])
        image = axis.imshow(gains, cmap="RdBu",
                            norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
        for i, line in enumerate(cells):
            for j, row in enumerate(line):
                color = "white" if abs(gains[i, j]) > .67 * limit else "#142230"
                ci = f'{row["gain_ci_low"] * 100:.2f}…{row["gain_ci_high"] * 100:.2f}'
                axis.text(j, i, f"{gains[i, j]:+.2f}\n[{ci}]", color=color,
                          ha="center", va="center", fontsize=8)
        axis.set(xticks=np.arange(len(regimes)), xticklabels=labels,
                 yticks=np.arange(len(regimes)), yticklabels=labels,
                 xlabel="Режим проверки", title=title)
        axis.tick_params(axis="x", labelrotation=14)
    axes[0].set_ylabel("Режим обучения")
    fig.colorbar(image, ax=list(axes), label="Выигрыш в MAE, п. п.", shrink=.82)
    save(fig, destination, "transfer")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    profile = PROFILES[args.profile]

    def progress(regime):
        print(f"BER forecast: {regime}", file=sys.stderr, flush=True)

    tables = run(profile, None if args.quiet else progress)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(args.output / f"{name}.csv", rows)
    manifest = {
        "profile": args.profile, **asdict(profile),
        "chains": [1, 2, 4], "challenge_bits": 64, "read_noise_range": [0.04, 0.14],
        "regime_parameters": {
            name: {"drift": tau, "lag_1_persistence": eta}
            for name, (tau, eta) in REGIMES.items()
        },
        "features": FEATURES, "target": "next_ber",
        "cross_regime_protocol": "frozen models and target-regime held-out devices",
        "scikit_learn": sklearn.__version__,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    plot(tables["metrics"], args.output)
    plot_transfer(tables["metrics"], tables["transfer_metrics"], args.output)


if __name__ == "__main__":
    main()
