#!/usr/bin/env python3
"""Render the publication-ready figures for Variant 1 and Variant 2 (500 DPI, serif, fontsize 14)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analyze import (
    evolution_summary,
    load_kv_results,
    trajectory_correlations,
    write_all_analyses,
)


PAPER_FONT_SIZE = 14
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
PAPER_WIDTH_INCHES = 7.16

METRIC_LABELS = {
    "bhattacharyya": r"Baseline Eq. (10) ($\mathcal{U}_\beta$)",
    "uq_v1": r"Variant 1 ($\mathcal{U}_{\mathrm{V1}}$)",
    "uq_v2": r"Variant 2 ($\mathcal{U}_{\mathrm{V2}}$)",
    "uq_mc": r"MC Reference ($q_t$)",
}

GT_LABELS = {
    "jaccard": "Jaccard Similarity",
    "bernoulli": "Exact Recovery (Bernoulli)",
    "misidentification": "Misidentification Error",
}

FAMILY_DISPLAY = {
    "Unif_K10_d2": "Unif (K=10, d=2)",
    "Norm_K10_d2": "Norm (K=10, d=2)",
    "Unif_K20_d2": "Unif (K=20, d=2)",
    "Norm_K20_d2": "Norm (K=20, d=2)",
    "Unif_K50_d2": "Unif (K=50, d=2)",
    "Unif_K10_d4": "Unif (K=10, d=4)",
    "Unif_K20_d4": "Unif (K=20, d=4)",
    "Unif_K10_d2_s05": r"Unif (K=10, d=2, $\sigma$=0.5)",
}


def _apply_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": PAPER_FONT_SIZE,
        "axes.titlesize": PAPER_FONT_SIZE,
        "axes.labelsize": PAPER_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
    })


def _env_key(name: str) -> int:
    return int(name.removeprefix("EgeExp"))


def _plot_single_evolution(summary: pd.DataFrame, measure: str, output: Path) -> None:
    _apply_paper_style()
    fig, ax = plt.subplots(figsize=(PAPER_WIDTH_INCHES, 4.6))
    for env in sorted(summary["environment"].unique(), key=_env_key):
        curve = summary[summary["environment"] == env].sort_values("normalized_budget")
        valid = curve["n_valid"] > 0
        line, = ax.plot(
            curve.loc[valid, "normalized_budget"],
            curve.loc[valid, "mean"],
            linewidth=2.0,
            label=env.removeprefix("EgeExp"),
        )
        ax.fill_between(
            curve.loc[valid, "normalized_budget"],
            curve.loc[valid, "lower"],
            curve.loc[valid, "upper"],
            color=line.get_color(),
            alpha=0.16,
            linewidth=0,
        )
    ax.set(xlim=(0, 1), ylim=(0, 1.03), xlabel="Normalized Budget", ylabel=METRIC_LABELS[measure])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(True, alpha=0.25)
    ax.legend(title="Environment", ncol=4, frameon=True, loc="best", fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=500, bbox_inches="tight")
    plt.close(fig)


def plot_variants_evolution_combined(rows: pd.DataFrame, output: Path) -> None:
    """Figure 1: Decrease in uncertainty over normalized budget in 8 environments for Variant 1 and Variant 2."""
    _apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    handles, labels = None, None
    panels = [("uq_v1", "(a) Variant 1: Directional BC Overlap"), ("uq_v2", "(b) Variant 2: Analytic Dominance Risk")]

    for ax, (measure, title) in zip(axes, panels):
        summary = evolution_summary(rows, "environment", measure)
        for env in sorted(summary["environment"].unique(), key=_env_key):
            curve = summary[summary["environment"] == env].sort_values("normalized_budget")
            valid = curve["n_valid"] > 0
            line, = ax.plot(
                curve.loc[valid, "normalized_budget"],
                curve.loc[valid, "mean"],
                linewidth=2.0,
                label=f"Env {env.removeprefix('EgeExp')}",
            )
            ax.fill_between(
                curve.loc[valid, "normalized_budget"],
                curve.loc[valid, "lower"],
                curve.loc[valid, "upper"],
                color=line.get_color(),
                alpha=0.16,
                linewidth=0,
            )
        ax.set(xlim=(0, 1), ylim=(0, 1.03), xlabel="Normalized Budget", title=title)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(True, alpha=0.25)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    axes[0].set_ylabel("Uncertainty Score")
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07),
        ncol=8, frameon=True, fontsize=LEGEND_FONT_SIZE,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=500, bbox_inches="tight")
    plt.close(fig)


def plot_variant_correlations_8envs(corr_df: pd.DataFrame, measure: str, output: Path) -> None:
    """Figures 2 & 3: Correlation boxplots of one UQ variant against all three ground-truth metrics across 8 envs."""
    _apply_paper_style()
    envs = sorted(corr_df["environment"].unique(), key=_env_key)
    env_labels = [e.removeprefix("EgeExp") for e in envs]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    gt_order = ("jaccard", "bernoulli", "misidentification")
    panel_titles = (
        "(a) Pearson r vs. Jaccard",
        "(b) Pearson r vs. Bernoulli (Exact)",
        "(c) Pearson r vs. Misidentification",
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6), sharey=True)
    for ax, gt, title in zip(axes, gt_order, panel_titles):
        sub = corr_df[(corr_df["measure"] == measure) & (corr_df["gt_metric"] == gt)]
        values = [sub.loc[sub["environment"] == env, "pearson_r"].dropna().to_numpy() for env in envs]
        boxes = ax.boxplot(
            values,
            labels=env_labels,
            vert=False,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.5},
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set(xlim=(-1.02, 1.02), xlabel="Pearson Correlation (r)", title=title)
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.grid(axis="x", alpha=0.25)

    axes[0].set_ylabel("Environment")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=500, bbox_inches="tight")
    plt.close(fig)


def plot_single_jaccard_correlation(corr_df: pd.DataFrame, measure: str, output: Path) -> None:
    _apply_paper_style()
    envs = sorted(corr_df["environment"].unique(), key=_env_key)
    sub = corr_df[(corr_df["measure"] == measure) & (corr_df["gt_metric"] == "jaccard")]
    values = [sub.loc[sub["environment"] == env, "pearson_r"].dropna().to_numpy() for env in envs]
    fig, ax = plt.subplots(figsize=(PAPER_WIDTH_INCHES, 4.6))
    boxes = ax.boxplot(
        values,
        labels=[e.removeprefix("EgeExp") for e in envs],
        vert=False,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, color in zip(boxes["boxes"], plt.rcParams["axes.prop_cycle"].by_key()["color"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set(xlim=(-1, 1), xlabel=f"Pearson r: {METRIC_LABELS[measure]} vs Jaccard", ylabel="Environment")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=500, bbox_inches="tight")
    plt.close(fig)


def plot_random_instances_correlation(random_corr_df: pd.DataFrame, output: Path) -> None:
    """Figure 4: Correlation boxplots across random instances comparing Eq. (10), Variant 1, and Variant 2."""
    _apply_paper_style()
    families = list(FAMILY_DISPLAY.keys())
    families = [f for f in families if f in set(random_corr_df["family"].unique())]
    measures = ("bhattacharyya", "uq_v1", "uq_v2")
    measure_colors = {"bhattacharyya": "#7f7f7f", "uq_v1": "#1f77b4", "uq_v2": "#d62728"}
    measure_names = {
        "bhattacharyya": r"Baseline Eq. (10) ($\mathcal{U}_\beta$)",
        "uq_v1": r"Variant 1 ($\mathcal{U}_{\mathrm{V1}}$)",
        "uq_v2": r"Variant 2 ($\mathcal{U}_{\mathrm{V2}}$)",
    }

    gt_order = ("jaccard", "bernoulli", "misidentification")
    panel_titles = (
        "(a) Pearson r vs. Jaccard",
        "(b) Pearson r vs. Bernoulli (Exact)",
        "(c) Pearson r vs. Misidentification",
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.6), sharey=True)
    base_y = np.arange(len(families), dtype=float)
    offsets = {"bhattacharyya": -0.25, "uq_v1": 0.0, "uq_v2": 0.25}
    legend_handles = []

    for ax, gt, title in zip(axes, gt_order, panel_titles):
        for m in measures:
            sub = random_corr_df[(random_corr_df["measure"] == m) & (random_corr_df["gt_metric"] == gt)]
            vals = [sub.loc[sub["family"] == fam, "pearson_r"].dropna().to_numpy() for fam in families]
            pos = base_y + offsets[m]
            boxes = ax.boxplot(
                vals,
                positions=pos,
                widths=0.22,
                vert=False,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.3},
            )
            for patch in boxes["boxes"]:
                patch.set_facecolor(measure_colors[m])
                patch.set_alpha(0.75)
            if gt == gt_order[0]:
                legend_handles.append(boxes["boxes"][0])
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_yticks(base_y)
        ax.set_yticklabels([FAMILY_DISPLAY.get(f, f) for f in families])
        ax.set(xlim=(-1.02, 1.02), xlabel="Pearson Correlation (r)", title=title)
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.grid(axis="x", alpha=0.25)

    fig.legend(
        legend_handles,
        [measure_names[m] for m in measures],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=True,
        fontsize=LEGEND_FONT_SIZE,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=500, bbox_inches="tight")
    plt.close(fig)


def render_all(results_dir: Path, random_trajectories: Path | None, output_dir: Path) -> None:
    write_all_analyses(results_dir, random_trajectories, output_dir)
    rows = load_kv_results(results_dir)
    corr_8env = trajectory_correlations(rows, "environment")

    # Individual & combined 8-env evolution figures (Figure 1)
    plot_variants_evolution_combined(rows, output_dir / "uq_variants_evolution_8envs.png")
    for m in ("bhattacharyya", "uq_v1", "uq_v2"):
        _plot_single_evolution(evolution_summary(rows, "environment", m), m, output_dir / f"{m}_evolution_kv.png")

    # Correlation boxplots with 3 GT metrics in 8 envs (Figures 2 & 3 + single Jaccard companions)
    plot_variant_correlations_8envs(corr_8env, "uq_v1", output_dir / "uq_v1_correlation_8envs.png")
    plot_variant_correlations_8envs(corr_8env, "uq_v2", output_dir / "uq_v2_correlation_8envs.png")
    plot_single_jaccard_correlation(corr_8env, "uq_v1", output_dir / "uq_v1_jaccard_correlation_kv.png")
    plot_single_jaccard_correlation(corr_8env, "uq_v2", output_dir / "uq_v2_jaccard_correlation_kv.png")

    # Random instances correlation boxplots (Figure 4)
    if random_trajectories is not None and random_trajectories.is_file():
        r_rows = pd.read_csv(random_trajectories, keep_default_na=True)
        r_corr = trajectory_correlations(r_rows, "family")
        plot_random_instances_correlation(r_corr, output_dir / "uq_random_instances_correlation.png")
    print(f"Rendered figures to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--random-trajectories", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render_all(args.results_dir, args.random_trajectories, args.output_dir)


if __name__ == "__main__":
    main()
