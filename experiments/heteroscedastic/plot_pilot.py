#!/usr/bin/env python3
"""Create primary and paired-effect heteroscedastic pilot figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .analyze import METRICS, load_pilot_rows, paired_contrasts, summarize_primary


DISPLAY = {"EGE_SH": "EGE-SH", "EGE_SR": "EGE-SR", "TTPFTS_UNKNOWN_VAR": "TTPFTS"}
ORDER = ["EGE_SH", "EGE_SR", "Uniform", "APE", "TTPFTS_UNKNOWN_VAR"]
LABELS = {
    "bernoulli": "Bernoulli Metric",
    "jaccard": "Jaccard Metric",
    "misidentification": "Misidentification Metric",
}


def _plot_primary(summary, metric: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=metric != "misidentification")
    for ax, condition in zip(axes, ("heterogeneous", "homoscedastic_rms")):
        for algorithm in ORDER:
            curve = summary[(summary.condition == condition) & (summary.algorithm == algorithm)]
            line, = ax.plot(curve.normalized_budget, curve["mean"], label=DISPLAY.get(algorithm, algorithm))
            ax.fill_between(curve.normalized_budget, curve.lower, curve.upper, color=line.get_color(), alpha=0.18)
        ax.set_title("Heterogeneous" if condition == "heterogeneous" else "RMS-matched control")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.grid(True, alpha=0.22)
        if metric in {"bernoulli", "jaccard"}:
            ax.set_ylim(0, 1)
    fig.supxlabel("Normalized Budget")
    fig.supylabel(LABELS[metric])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True)
    fig.tight_layout(rect=[0.04, 0.06, 1, 0.84])
    fig.savefig(out, dpi=500, bbox_inches="tight")
    plt.close(fig)


def _plot_paired(contrasts, metric: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for algorithm in ORDER:
        curve = contrasts[contrasts.algorithm == algorithm]
        line, = ax.plot(curve.normalized_budget, curve.heterogeneous_minus_control,
                        label=DISPLAY.get(algorithm, algorithm))
        ax.fill_between(curve.normalized_budget, curve.lower, curve.upper, color=line.get_color(), alpha=0.18)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlim=(0, 1), xlabel="Normalized Budget",
           ylabel=f"Heterogeneous − control {LABELS[metric]}")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=500, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/heteroscedastic_pilot"))
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    rows = load_pilot_rows(args.results_dir)
    for metric in METRICS:
        primary = summarize_primary(rows, metric)
        paired = paired_contrasts(rows, metric)
        primary.to_csv(args.figures_dir / f"{metric}_primary_table.csv", index=False)
        paired.to_csv(args.figures_dir / f"{metric}_paired_table.csv", index=False)
        _plot_primary(primary, metric, args.figures_dir / f"{metric}_primary.png")
        _plot_paired(paired, metric, args.figures_dir / f"{metric}_paired.png")


if __name__ == "__main__":
    main()
