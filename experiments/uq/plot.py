#!/usr/bin/env python3
"""Render the six approved hardness-budget UQ publication figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import MEASURES, correlation_summary, evolution_summary, load_results, trajectory_correlations, write_analysis


LABELS = {
    "bhattacharyya": "Bhattacharyya Overlap",
    "symmetric_kl": "Symmetric KL Divergence",
    "entropy": "Posterior Entropy (nats)",
}


def _environment_sort_key(name: str) -> int:
    return int(name.removeprefix("EgeExp"))


def plot_evolution(summary: pd.DataFrame, measure: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for environment in sorted(summary["environment"].unique(), key=_environment_sort_key):
        curve = summary[summary["environment"] == environment].sort_values("normalized_budget")
        valid = curve["n_valid"] > 0
        line, = ax.plot(
            curve.loc[valid, "normalized_budget"], curve.loc[valid, "mean"],
            linewidth=1.8, label=environment.removeprefix("EgeExp"),
        )
        ax.fill_between(
            curve.loc[valid, "normalized_budget"], curve.loc[valid, "lower"], curve.loc[valid, "upper"],
            color=line.get_color(), alpha=0.16, linewidth=0,
        )
    ax.set(
        xlim=(0, 1),
        xlabel="Normalized Budget",
        ylabel=LABELS[measure],
    )
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(True, alpha=0.25)
    ax.legend(title="Environment", ncol=4, frameon=True, loc="best")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=500, bbox_inches="tight")
    plt.close(fig)


def plot_correlations(correlations: pd.DataFrame, measure: str, output: Path) -> None:
    environments = sorted(correlations["environment"].unique(), key=_environment_sort_key)
    values = [
        correlations.loc[correlations["environment"] == environment, "pearson_r"].dropna().to_numpy()
        for environment in environments
    ]
    summary = correlation_summary(correlations).set_index("environment")
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    boxes = ax.boxplot(values, labels=[
        f"{environment.removeprefix('EgeExp')}\n({int(summary.loc[environment, 'n_valid'])}/{int(summary.loc[environment, 'n_total'])})"
        for environment in environments
    ], patch_artist=True, showfliers=True)
    for patch, color in zip(boxes["boxes"], plt.rcParams["axes.prop_cycle"].by_key()["color"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(
        ylim=(-1, 1),
        xlabel="Environment (valid trajectories / total)",
        ylabel=f"Pearson r: {LABELS[measure]} vs Jaccard",
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=500, bbox_inches="tight")
    plt.close(fig)


def render(results_dir: Path, output_dir: Path) -> None:
    """Write analysis tables and exactly six UQ figures."""
    write_analysis(results_dir, output_dir)
    rows = load_results(results_dir)
    for measure in MEASURES:
        evolution = evolution_summary(rows, measure)
        correlations = trajectory_correlations(rows, measure)
        plot_evolution(evolution, measure, output_dir / f"{measure}_evolution.png")
        plot_correlations(correlations, measure, output_dir / f"{measure}_jaccard_correlation.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
