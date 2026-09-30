#!/usr/bin/env python3
"""Render three paper-style metric figures from exact bootstrap summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

from .analyze import METRICS
from .config import ALGORITHMS, default_output_root, load_config


FIGSIZE = (7.16, 4.35)
FONT_SIZE = 12
XTICKS = np.linspace(0.0, 1.0, 5)
DISPLAY = {
    "EGE_SH": "EGE-SH",
    "EGE_SR": "EGE-SR",
    "Uniform": "Uniform",
    "APE": "APE",
    "TTPFTS_UNKNOWN_VAR": "TTPFTS",
}
LABELS = {
    "jaccard": "Jaccard Metric",
    "bernoulli": "Bernoulli Metric",
    "misidentification": "Misidentification Metric",
}


def render(metric: str, summary_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Render a metric average over the exact 1,000-instance frozen cohort."""
    if metric not in METRICS:
        raise ValueError(f"unsupported metric {metric}")
    table = pd.read_csv(summary_path)
    required = {
        "algorithm", "grid_index", "arm_pull_normalized", "mean", "lower", "upper",
        "n_instances", "bootstrap_replicates", "metric",
    }
    missing = required - set(table)
    if missing:
        raise ValueError(f"{summary_path}: missing plotting columns {sorted(missing)}")
    if set(table["metric"]) != {metric}:
        raise ValueError(f"{summary_path}: expected only {metric} rows")
    if set(table["algorithm"]) != set(ALGORITHMS):
        raise ValueError(f"{summary_path}: unexpected algorithms")
    if (table["n_instances"] != 1000).any() or (table["bootstrap_replicates"] != 2000).any():
        raise ValueError("publication plot requires all 1,000 instances and 2,000 bootstrap replicates")
    plt.rcParams.update({"font.family": "serif", "font.size": FONT_SIZE})
    fig, axis = plt.subplots(figsize=FIGSIZE)
    for algorithm in ALGORITHMS:
        curve = table[table["algorithm"] == algorithm].sort_values("grid_index")
        if curve["grid_index"].tolist() != list(range(21)):
            raise ValueError(f"{algorithm}: incomplete checkpoint curve")
        x = curve["arm_pull_normalized"].to_numpy(dtype=float)
        if not np.allclose(x, np.linspace(0.0, 1.0, 21), rtol=0.0, atol=1e-12):
            raise ValueError(f"{algorithm}: normalized grid is not the approved endpoint-inclusive grid")
        mean = curve["mean"].to_numpy(dtype=float)
        lower = curve["lower"].to_numpy(dtype=float)
        upper = curve["upper"].to_numpy(dtype=float)
        if metric == "misidentification":
            visible = mean > 0.0
            mean, lower, upper = (
                np.where(visible, mean, np.nan),
                np.where(visible, lower, np.nan),
                np.where(visible, upper, np.nan),
            )
        line, = axis.plot(x, mean, label=DISPLAY[algorithm], linewidth=1.7)
        axis.fill_between(x, lower, upper, color=line.get_color(), alpha=0.17, linewidth=0)
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_locator(FixedLocator(XTICKS))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}".rstrip("0").rstrip(".")))
    if metric in {"jaccard", "bernoulli"}:
        axis.set_ylim(0, 1)
    else:
        positive = table.loc[table["mean"] > 0, "mean"]
        if positive.empty:
            raise ValueError("cannot log-plot all-zero misidentification")
        axis.set_yscale("log")
        axis.set_ylim(bottom=float(positive.min()) / 2.0)
    axis.set_xlabel("Normalized Budget", fontsize=FONT_SIZE)
    axis.set_ylabel(LABELS[metric], fontsize=FONT_SIZE)
    axis.tick_params(axis="both", labelsize=FONT_SIZE)
    axis.grid(alpha=0.22, linewidth=0.55)
    axis.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.23), ncol=5,
        fontsize=FONT_SIZE, frameon=True, columnspacing=0.9, handlelength=1.5,
    )
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.16, top=0.78)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.sort_values(["algorithm", "grid_index"]).to_csv(
        output_dir / f"{metric}_plotted_values.csv", index=False
    )
    png = output_dir / f"heterogeneous_ensemble_{metric}.png"
    pdf = output_dir / f"heterogeneous_ensemble_{metric}.pdf"
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
    analysis_dir = args.analysis_dir or root / "analysis"
    output_dir = args.output_dir or root / "figures"
    for metric in METRICS:
        png, pdf = render(metric, analysis_dir / f"{metric}_summary.csv", output_dir)
        print(f"Wrote {png} and {pdf}")


if __name__ == "__main__":
    main()
