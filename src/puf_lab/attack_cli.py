"""Run the reproducible synthetic CRP modeling experiment."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import scipy
import sklearn

from .attack import PROFILES, run
from .cli import write_csv
from .plots import PALETTE, save, style
import matplotlib.pyplot as plt


def plot(rows, destination):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout="constrained", sharey=True)
    for i, noise in enumerate((0., .1)):
        for j, chains in enumerate((1, 2, 4)):
            axis = axes[i, j]
            for model, label, color in zip(
                ("majority", "linear", "product", "shuffled"),
                ("Константа", "Линейная LR", "Произведение задержек", "Перемешанные метки"),
                PALETTE):
                subset = [r for r in rows if (r["chains"], r["noise"], r["model"]) == (chains, noise, model)]
                xx = [r["train_crps"] for r in subset]
                yy = np.array([r["accuracy"] for r in subset])
                axis.plot(xx, yy, "o-", label=label, color=color, markersize=3)
                axis.fill_between(xx, [r["ci_low"] for r in subset],
                                  [r["ci_high"] for r in subset], color=color, alpha=.1)
            oracle = next(r["oracle_accuracy"] for r in rows if r["chains"] == chains and r["noise"] == noise)
            axis.axhline(oracle, linestyle="--", color="#73828c", linewidth=1)
            axis.set_title(f"{chains}-XOR, σ = {noise:.1f}")
            axis.set_xscale("log", base=2)
            axis.set_ylim(.38, 1.02)
            if i == 1:
                axis.set_xlabel("Обучающие CRP (+ валидация для рестартов)")
            style(axis, "Точность на новых запросах" if j == 0 else None)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    save(fig, destination, "learning_curves")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    profile = PROFILES[args.profile]
    tables, arrays = run(profile, None if args.quiet else lambda i: print(
        f"CRP attack {args.profile}: device {i}/{profile.devices}", flush=True))
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(args.output / f"{name}.csv", rows)
    np.savez_compressed(args.output / "predictions.npz", **arrays)
    manifest = {"profile": args.profile, **asdict(profile), "data_origin": "synthetic additive-delay model",
        "target": "final XOR bit on unseen challenges of the SAME device",
        "selection": "restart chosen by validation accuracy; no test tuning or retraining",
        "regularization": 1e-4, "linear_C": 1.,
        "intervals": "pointwise paired percentile bootstrap over independent devices; not simultaneous",
        "independent_repeat": "new device weights, challenges, noise and optimizer starts",
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("attack.py", "model.py")},
        "python": platform.python_version(), "numpy": np.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    plot(tables["metrics"], args.output)


if __name__ == "__main__":
    main()
