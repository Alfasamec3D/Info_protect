"""Static publication figures created from aggregate metrics."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from .summary import group_rows

PALETTE = ["#116b67", "#315cb0", "#a34720", "#743980"]
INK = "#172c3a"
GRID = "#dce6ec"


def style(axis, ylabel=None):
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("#b1c0ca")
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(colors=INK)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    if ylabel:
        axis.set_ylabel(ylabel, color=INK)


def save(figure, directory, name):
    figure.savefig(directory / f"{name}.png", dpi=180, facecolor="white")
    plt.close(figure)


def reliability_plot(tables, directory):
    fig, (ax, distance) = plt.subplots(
        1, 2, figsize=(10.5, 4.1), gridspec_kw={"width_ratios": [1.7, 1]},
        layout="constrained",
    )
    for i, ((k,), rows) in enumerate(group_rows(tables["reliability"], "chains").items()):
        color = PALETTE[i % len(PALETTE)]
        noise = [row["noise"] for row in rows]
        ax.plot(noise, [row["theory_ber"] for row in rows], "--", c=color, alpha=0.68)
        y = np.array([row["ber"] for row in rows])
        low = [row["ber_ci_low"] for row in rows]
        high = [row["ber_ci_high"] for row in rows]
        ax.errorbar(
            noise, y, yerr=np.maximum(0, [y - low, high - y]),
            fmt="o", c=color, capsize=3, label=f"k = {k}",
        )
    ax.set_title("Ошибки при чтении")
    ax.set_xlabel("Стандартное отклонение шума σ")
    style(ax, "BER относительно чистого ответа")
    ax.legend(frameon=False, ncols=2, title="Число цепочек")
    rows = tables["uniqueness"]
    x = [row["chains"] for row in rows]
    y = np.array([row["hamming"] for row in rows])
    low = [row["hamming_ci_low"] for row in rows]
    high = [row["hamming_ci_high"] for row in rows]
    distance.axhline(0.5, color="#647680", lw=1, ls="--")
    distance.errorbar(x, y, yerr=np.maximum(0, [y - low, high - y]), fmt="o",
                      color=PALETTE[0], capsize=4)
    distance.set_title("Межэкземплярное различие")
    distance.set_xlabel("Число цепочек k")
    distance.set_xticks(x)
    style(distance, "Нормированное расстояние Хэмминга")
    save(fig, directory, "reliability")


def budget_plot(tables, directory):
    rows = tables["budget_metrics"]
    groups = group_rows(rows, "budget", "noise")
    budgets = sorted({row["budget"] for row in rows})
    noises = sorted({row["noise"] for row in rows})
    fig, axes = plt.subplots(
        len(budgets), len(noises), figsize=(max(5, 3.35 * len(noises)), 2.7 * len(budgets)),
        squeeze=False, layout="constrained",
    )
    for i, budget in enumerate(budgets):
        for j, noise in enumerate(noises):
            ax = axes[i, j]
            by_chain = group_rows(groups[budget, noise], "chains")
            for color_index, ((k,), records) in enumerate(by_chain.items()):
                color = PALETTE[color_index % len(PALETTE)]
                x = [row["enrollment_reads"] for row in records]
                ax.plot(x, [row["analytic_rmse"] for row in records],
                        "-o", color=color, linewidth=1.3, markersize=4, label=f"k = {k}")
                ax.set_xticks(x)
            ax.set_title(f"B = {budget}  ·  σ = {noise:g}", color=INK, fontsize=10)
            style(ax, "RMSE" if j == 0 else None)
            if i == len(budgets) - 1:
                ax.set_xlabel("Чтений для эталона R")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=min(6, len(labels)),
               frameon=False)
    save(fig, directory, "measurement_budget")


def reference_points_plot(tables, directory):
    rows = tables["reliability"]
    noise = min({row["noise"] for row in rows}, key=lambda value: abs(value - 0.1))
    rows = [row for row in rows if row["noise"] == noise]
    x = np.array([row["chains"] for row in rows])
    fig, ax = plt.subplots(figsize=(8.5, 3.7), layout="constrained")
    ax.axhline(0.5, color="#647680", linestyle="--", label="Случайное угадывание")
    ax.plot(x, [row["majority_accuracy"] for row in rows], "-o",
            color=PALETTE[1], label="Константа по частому ответу")
    y = np.array([row["noisy_read_ceiling"] for row in rows])
    low = [row["ceiling_ci_low"] for row in rows]
    high = [row["ceiling_ci_high"] for row in rows]
    ax.errorbar(x, y, yerr=np.maximum(0, [y - low, high - y]), fmt="-o",
                capsize=3, color=PALETTE[0], label="При известном чистом ответе")
    ax.set_title(f"Ориентиры точности на шумном чтении · σ = {noise:g}")
    ax.set(xlabel="Число цепочек k", ylim=(0.44, 1.03), xticks=x)
    style(ax, "Доля совпадающих ответов")
    ax.legend(frameon=False, loc="center right")
    save(fig, directory, "reference_points")


def plot(tables, directory):
    reliability_plot(tables, directory)
    budget_plot(tables, directory)
    reference_points_plot(tables, directory)
