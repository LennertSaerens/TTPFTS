#!/usr/bin/env python3
"""Render the publication-style 2x4 Jaccard figure for the validated p sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import default_output_root, load_config


PAPER_WIDTH_INCHES = 7.16
FONT_SIZE = 12
XTICKS = [0, 0.25, 0.5, 0.75, 1]


def _environment_key(name: str) -> int:
    return int(name.removeprefix("EgeExp"))


def render(jaccard_summary: Path, output_dir: Path) -> Path:
    """Render the exact Jaccard summary table as a double-column p-sweep figure."""
    table = pd.read_csv(jaccard_summary)
    required = {
        "environment", "p", "normalized_budget", "mean", "lower", "upper", "n", "metric"
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Jaccard summary is missing columns: {sorted(missing)}")
    if set(table["metric"]) != {"jaccard"}:
        raise ValueError("plot input must contain only Jaccard summaries")
    environments = sorted(table["environment"].unique(), key=_environment_key)
    if len(environments) != 8:
        raise ValueError(f"expected exactly eight environments, found {len(environments)}")
    p_values = sorted(table["p"].unique())
    if len(p_values) != 11 or not np.allclose(p_values, np.linspace(0.0, 1.0, 11)):
        raise ValueError("expected p=0.0,0.1,...,1.0")
    if (table.groupby(["environment", "p"])["n"].min() != 100).any():
        raise ValueError("all plotted checkpoints must contain 100 repetitions")

    plt.rcParams.update({"font.family": "serif", "font.size": FONT_SIZE})
    fig, axes = plt.subplots(2, 4, figsize=(PAPER_WIDTH_INCHES, 5.45), sharex=True, sharey=True)
    colors = plt.cm.coolwarm(np.linspace(0.05, 0.95, len(p_values)))
    legend_handles, legend_labels = [], []
    for axis, environment in zip(axes.flat, environments):
        environment_rows = table[table["environment"] == environment]
        for index, p in enumerate(p_values):
            curve = environment_rows[environment_rows["p"] == p].sort_values("normalized_budget")
            if curve["normalized_budget"].iloc[0] != 0.0 or curve["normalized_budget"].iloc[-1] != 1.0:
                raise ValueError(f"{environment}/p={p}: missing normalized endpoint")
            emphasized = np.isclose(p, 0.5)
            line, = axis.plot(
                curve["normalized_budget"],
                curve["mean"],
                color="black" if emphasized else colors[index],
                linewidth=2.1 if emphasized else 1.35,
                marker="^" if emphasized else None,
                markersize=3.5,
                markevery=4 if emphasized else None,
                label=fr"$p={p:.1f}$",
                zorder=3 if emphasized else 2,
            )
            axis.fill_between(
                curve["normalized_budget"], curve["lower"], curve["upper"],
                color=line.get_color(), alpha=0.13 if emphasized else 0.08, linewidth=0,
            )
            if environment == environments[0]:
                legend_handles.append(line)
                legend_labels.append(fr"$p={p:.1f}$")
        axis.text(
            0.96, 0.075, str(_environment_key(environment)),
            transform=axis.transAxes, ha="right", va="bottom", fontsize=FONT_SIZE,
        )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xticks(XTICKS)
        axis.set_yticks(XTICKS)
        axis.grid(alpha=0.22, linewidth=0.6)
        axis.tick_params(labelsize=FONT_SIZE)
    fig.supxlabel("Normalized Budget", fontsize=FONT_SIZE, y=0.055)
    fig.supylabel("Jaccard Metric", fontsize=FONT_SIZE, x=0.035)
    fig.legend(
        legend_handles, legend_labels, ncol=6, loc="upper center",
        bbox_to_anchor=(0.5, 0.99), frameon=True, fontsize=FONT_SIZE,
        columnspacing=0.9, handlelength=1.55,
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.14, top=0.78, wspace=0.22, hspace=0.26)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.sort_values(["environment", "p", "normalized_budget"]).to_csv(
        output_dir / "jaccard_plotted_values.csv", index=False
    )
    output = output_dir / "ttpfts_p_sweep_jaccard.png"
    fig.savefig(output, dpi=500, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_root = default_output_root(load_config())
    output = render(
        (args.analysis_dir or output_root / "analysis") / "jaccard_summary.csv",
        args.output_dir or output_root / "figures",
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
