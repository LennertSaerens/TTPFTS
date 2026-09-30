#!/usr/bin/env python3
"""Render the two matched environments and six condition-specific metric figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import FixedLocator, FuncFormatter

from .analyze import METRICS
from .config import ALGORITHMS, CONDITIONS, STD_VALUES, default_output_root, load_config
from .instances import FrozenCondition, load_conditions


FONT_SIZE = 12
PERFORMANCE_FIGSIZE = (7.16, 4.35)
ENVIRONMENT_FIGSIZE = (7.16, 5.1)
XTICKS = np.linspace(0.0, 1.0, 5)
DISPLAY = {
    "EGE_SH": "EGE-SH",
    "EGE_SR": "EGE-SR",
    "Uniform": "Uniform",
    "APE": "APE",
    "TTPFTS_UNKNOWN_VAR": "TTPFTS",
}
METRIC_LABELS = {
    "jaccard": "Jaccard Metric",
    "bernoulli": "Bernoulli Metric",
    "misidentification": "Misidentification Metric",
}


def _configure() -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": FONT_SIZE})


def render_environment(item: FrozenCondition, output_dir: Path) -> tuple[Path, Path]:
    """Render a fixed-limit mean-space view of one variance assignment."""
    _configure()
    fig, axis = plt.subplots(figsize=ENVIRONMENT_FIGSIZE)
    cmap = ListedColormap(plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(STD_VALUES))))
    bounds = np.asarray([0.05, 0.15, 0.25, 0.35, 0.45, 0.55])
    norm = BoundaryNorm(bounds, cmap.N)
    pareto = np.zeros(item.num_arms, dtype=bool)
    pareto[item.pareto_indices] = True
    for mask, marker, label, size in (
        (~pareto, "o", "Dominated", 56),
        (pareto, "D", "True Pareto", 76),
    ):
        scatter = axis.scatter(
            item.means[mask, 0], item.means[mask, 1],
            c=item.arm_stds[mask], cmap=cmap, norm=norm, marker=marker, s=size,
            edgecolors="black", linewidths=0.65, label=label, zorder=3 if marker == "D" else 2,
        )
    epsilon_offsets = ((-10, 7), (0, 12), (10, 5))
    for arm, (x_value, y_value) in enumerate(item.means):
        if arm < 5:
            offset = (7, 7)
        else:
            # The three epsilon perturbations around each parent/gap mean
            # point are intentionally very close; fixed distinct offsets keep
            # all arm IDs readable without data-dependent text repulsion.
            offset = epsilon_offsets[(arm - 5) % 3]
        axis.annotate(
            str(arm), (x_value, y_value), xytext=offset, textcoords="offset points",
            fontsize=6.2, zorder=4,
        )
    axis.set(
        xlim=(0.15, 0.95), ylim=(0.10, 0.95),
        xlabel="Objective 1 mean", ylabel="Objective 2 mean",
        title=item.condition.replace("_", " ").title(),
    )
    axis.tick_params(labelsize=FONT_SIZE)
    axis.grid(alpha=0.22, linewidth=0.55)
    axis.legend(loc="lower left", frameon=True, fontsize=FONT_SIZE)
    colorbar = fig.colorbar(scatter, ax=axis, ticks=STD_VALUES, pad=0.02)
    colorbar.set_label("Per-arm standard deviation", fontsize=FONT_SIZE)
    colorbar.ax.tick_params(labelsize=FONT_SIZE)
    inset = axis.inset_axes([0.60, 0.08, 0.35, 0.27])
    inset.scatter(item.noise_ranks, item.gaps, c=item.arm_stds, cmap=cmap, norm=norm, s=20, edgecolors="none")
    inset.set(xlabel="Gap rank", ylabel="True gap", xlim=(-1, 50))
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.18, linewidth=0.45)
    fig.subplots_adjust(left=0.11, right=0.91, bottom=0.13, top=0.92)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(
        {
            "condition": item.condition,
            "arm": np.arange(item.num_arms),
            "objective_1_mean": item.means[:, 0],
            "objective_2_mean": item.means[:, 1],
            "std": item.arm_stds,
            "true_gap": item.gaps,
            "noise_rank": item.noise_ranks,
            "is_true_pareto": pareto.astype(int),
        }
    )
    table.to_csv(output_dir / f"{item.condition}_environment_plotted_values.csv", index=False)
    png, pdf = output_dir / f"{item.condition}_environment.png", output_dir / f"{item.condition}_environment.pdf"
    fig.savefig(png, dpi=500, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def render_performance(condition: str, metric: str, summary_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Render one requested condition-specific metric curve figure."""
    if condition not in CONDITIONS or metric not in METRICS:
        raise ValueError("unsupported condition or metric")
    table = pd.read_csv(summary_path)
    required = {
        "condition", "algorithm", "grid_index", "normalized_budget", "metric",
        "mean", "lower", "upper", "n_repetitions", "bootstrap_replicates",
    }
    if required - set(table):
        raise ValueError(f"{summary_path}: missing plotting columns {sorted(required - set(table))}")
    table = table[table.condition == condition].copy()
    if set(table.metric) != {metric} or set(table.algorithm) != set(ALGORITHMS):
        raise ValueError(f"{summary_path}: incomplete condition/metric algorithm coverage")
    if (table.n_repetitions != 100).any() or (table.bootstrap_replicates != 2000).any():
        raise ValueError("publication figures require 100 repetitions and 2,000 bootstrap replicates")
    _configure()
    fig, axis = plt.subplots(figsize=PERFORMANCE_FIGSIZE)
    for algorithm in ALGORITHMS:
        curve = table[table.algorithm == algorithm].sort_values("grid_index")
        if curve.grid_index.tolist() != list(range(21)):
            raise ValueError(f"{condition}/{algorithm}: incomplete grid")
        x_values = curve.normalized_budget.to_numpy(float)
        # Raw checkpoints are round(kB/20), so their normalized locations
        # retain the intentionally tiny rounding deviations from k/20.
        if x_values[0] != 0.0 or x_values[-1] != 1.0 or np.any(np.diff(x_values) <= 0.0):
            raise ValueError(f"{condition}/{algorithm}: invalid endpoint-inclusive normalized grid")
        mean, lower, upper = (curve[column].to_numpy(float) for column in ("mean", "lower", "upper"))
        if metric == "misidentification":
            visible = mean > 0.0
            mean, lower, upper = tuple(np.where(visible, values, np.nan) for values in (mean, lower, upper))
        line, = axis.plot(x_values, mean, label=DISPLAY[algorithm], linewidth=1.7)
        axis.fill_between(x_values, lower, upper, color=line.get_color(), alpha=0.17, linewidth=0)
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_locator(FixedLocator(XTICKS))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}".rstrip("0").rstrip(".")))
    if metric in {"jaccard", "bernoulli"}:
        axis.set_ylim(0, 1)
    else:
        positive = table.loc[table["mean"] > 0.0, "mean"]
        if positive.empty:
            raise ValueError(f"{condition}/{metric}: no positive value for log plotting")
        axis.set_yscale("log")
        axis.set_ylim(bottom=float(positive.min()) / 2.0)
    axis.set_xlabel("Normalized Budget", fontsize=FONT_SIZE)
    axis.set_ylabel(METRIC_LABELS[metric], fontsize=FONT_SIZE)
    axis.tick_params(axis="both", labelsize=FONT_SIZE)
    axis.grid(alpha=0.22, linewidth=0.55)
    axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.23), ncol=5, frameon=True,
        fontsize=FONT_SIZE, columnspacing=0.9, handlelength=1.5,
    )
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.16, top=0.78)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.sort_values(["algorithm", "grid_index"]).to_csv(
        output_dir / f"{condition}_{metric}_plotted_values.csv", index=False
    )
    png, pdf = output_dir / f"{condition}_{metric}.png", output_dir / f"{condition}_{metric}.pdf"
    fig.savefig(png, dpi=500, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = default_output_root(load_config())
    analysis_dir, output_dir = args.analysis_dir or root / "analysis", args.output_dir or root / "figures"
    for item in load_conditions():
        render_environment(item, output_dir)
    for condition in CONDITIONS:
        for metric in METRICS:
            render_performance(condition, metric, analysis_dir / f"{metric}_condition_summary.csv", output_dir)


if __name__ == "__main__":
    main()
