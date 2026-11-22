"""Run the held-out cohort experiment for population drift calibration."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

from .calibration import (
    COMPARISONS, EQUAL_SPAN_COMPARISONS, EQUAL_SPAN_PLANS,
    PLANS, PROFILES, SCENARIOS, TEST_LAGS, aggregate, calibrate,
)
from .cli import write_csv
from .plots import PALETTE, save, style

import matplotlib.pyplot as plt
import numpy as np


def plot(tables, destination):
    comparisons = tables["calibration_contrasts"]
    predictions = tables["calibration_predictions"]
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), layout="constrained")
    gain, curve = axes
    cases = ("no_drift", "weak", "fast", "slow", "mixed")
    labels = ("Нет дрейфа", "Слабый", "Быстрый", "Медленный", "Две скорости")
    x = np.arange(len(cases))
    for i, (k, offset) in enumerate(((1, -0.11), (4, 0.11))):
        rows = [next(r for r in comparisons if r["scenario"] == case and r["chains"] == k
                     and r["comparison"] == "long_vs_short_single") for case in cases]
        y = np.array([r["cohort_mae_gain"] for r in rows])
        low = [r["ci_low"] for r in rows]
        high = [r["ci_high"] for r in rows]
        gain.errorbar(x + offset, y, yerr=np.maximum(0, [y - low, high - y]), fmt="o", capsize=3,
                      c=PALETTE[i], label=f"k = {k}")
    gain.axhline(0, c="#758994", ls="--", lw=1)
    gain.set_title("Выигрыш длинного плана на новых устройствах")
    gain.set_xticks(x, labels, rotation=16)
    style(gain, "Отклонение MAE короткого и длинного планов")
    gain.legend(frameon=False)

    for i, (plan, model, label) in enumerate((
        ("short", "single", "Короткий план"),
        ("long", "single", "Длинный план"),
        ("long", "two_scale", "Длинный, две компоненты"),
        ("long", "interpolate", "Линейная интерполяция"),
    )):
        rows = [r for r in predictions if r["scenario"] == "mixed" and r["chains"] == 4
                and r["plan"] == plan and r["model"] == model]
        curve.plot([r["lag"] for r in rows], [r["predicted"] for r in rows], "o-",
                   c=PALETTE[i], label=label)
    curve.plot([r["lag"] for r in rows], [r["analytic"] for r in rows], "--",
               c="#1b2833", label="Аналитическое среднее")
    curve.set(xlabel="Лаг на контрольном измерении", title="Оценка расхождений при двух скоростях дрейфа")
    style(curve, "Доля расхождений · 4-XOR")
    curve.legend(frameon=False, fontsize=8)
    save(figure, destination, "calibration")


def plot_equal_span(tables, destination):
    contrasts = tables["calibration_contrasts"]
    figure, axis = plt.subplots(figsize=(8.4, 4.05), layout="constrained")
    cases = ("no_drift", "weak", "fast", "slow", "mixed")
    labels = ("Нет дрейфа", "Слабый", "Быстрый", "Медленный", "Две скорости")
    x = np.arange(len(cases))
    for i, (name, label) in enumerate((
        ("spread_vs_early_two", "Распределённый против раннего"),
        ("spread_vs_late_two", "Распределённый против позднего"),
    )):
        rows = [next(r for r in contrasts if r["scenario"] == case and r["chains"] == 4
                     and r["comparison"] == name) for case in cases]
        y = np.array([r["cohort_mae_gain"] for r in rows])
        lo = [r["ci_low"] for r in rows]
        hi = [r["ci_high"] for r in rows]
        axis.errorbar(x + (-0.1 if i == 0 else 0.1), y,
                      yerr=np.maximum(0, [y - lo, hi - y]), fmt="o",
                      capsize=3, c=PALETTE[i], label=label)
    axis.axhline(0, color="#758994", linewidth=1, ls="--")
    axis.set_title("Положение промежуточных измерений при одном горизонте · 4-XOR")
    axis.set_xticks(x, labels)
    style(axis, "Снижение MAE на новых устройствах")
    axis.legend(frameon=False, fontsize=9)
    save(figure, destination, "calibration")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=("window", "equal-span"), default="window")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    size = PROFILES[args.profile]

    def progress(done, total):
        print(f"Calibration scenarios: {done}/{total}", file=sys.stderr, flush=True)

    equal_span = args.family == "equal-span"
    plans = EQUAL_SPAN_PLANS if equal_span else PLANS
    comparisons = EQUAL_SPAN_COMPARISONS if equal_span else COMPARISONS
    measured = calibrate(size, None if args.quiet else progress, plans=plans)
    tables = aggregate(size, *measured, comparisons=comparisons)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(args.output / f"{name}.csv", rows)
    manifest = {"profile": args.profile, **asdict(size),
                "scenarios": {k: asdict(v) for k, v in SCENARIOS.items()},
                "plans": plans, "test_lags": TEST_LAGS, "reads_per_calibration_series": 8}
    (args.output / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    draw = plot_equal_span if equal_span else plot
    draw(tables, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
