"""Run the session-drift experiment and export aggregate reliability results."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from .cli import write_csv
from .plots import PALETTE, save, style
from .session import SessionConfig, collect, session_summary
from .summary import group_rows

PROTOCOL_LABELS = {
    "same_session": "Одна сессия",
    "held_out": "Отдельная проверка",
    "distributed": "Распределённый эталон",
}


def plots(tables, directory):
    metrics = tables["session_metrics"]
    key = max(row["budget"] for row in metrics)
    noise = max(row["noise"] for row in metrics)
    max_k = max(row["chains"] for row in metrics)
    high_drift = max(row["drift"] for row in metrics)
    focus = [row for row in metrics if row["budget"] == key and row["noise"] == noise]
    fig, (gap, rmse) = plt.subplots(1, 2, figsize=(10.7, 4.25), layout="constrained")
    contrasts = [row for row in tables["session_contrasts"]
                 if row["budget"] == key and row["noise"] == noise and row["reads"] == 1]
    for i, ((chains,), rows) in enumerate(group_rows(contrasts, "chains").items()):
        x = [r["drift"] for r in rows]
        y = np.array([r["held_minus_same"] for r in rows])
        low = [r["gap_ci_low"] for r in rows]
        high = [r["gap_ci_high"] for r in rows]
        gap.errorbar(x, y, yerr=np.maximum(0, [y - low, high - y]), fmt="-o",
                     capsize=3, color=PALETTE[i % len(PALETTE)], label=f"k = {chains}")
    gap.set(xlabel="Разброс между сессиями τ", title="Расхождения при отдельной проверке")
    style(gap, "Отдельная проверка − одна сессия")
    gap.legend(frameon=False, ncols=2)
    groups = group_rows([r for r in focus if r["chains"] == max_k and r["drift"] == high_drift],
                        "protocol")
    for i, ((protocol,), rows) in enumerate(groups.items()):
        x = [r["reads"] for r in rows]
        rmse.plot(x, [r["rmse"] for r in rows], "-o", color=PALETTE[i],
                  label=PROTOCOL_LABELS[protocol])
        rmse.set_xticks(x)
    rmse.set(xlabel="Чтений для эталона R", title=f"k = {max_k}, τ = {high_drift:g}")
    style(rmse, "RMSE оценки BER")
    rmse.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Эффект сессий измерения · σ = {noise:g}, B = {key}")
    save(fig, directory, "session_study")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = SessionConfig.load(args.config)
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    def progress(done, total):
        print(f"Session study: {done}/{total}", file=sys.stderr, flush=True)

    rows, pairs = collect(config, None if args.quiet else progress)
    tables = session_summary(config, rows, pairs)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        write_csv(args.output / f"{name}.csv", table)
    manifest = {"config": asdict(config), "devices_are_bootstrap_clusters": True}
    (args.output / "session_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    plots(tables, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
