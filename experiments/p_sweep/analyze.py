#!/usr/bin/env python3
"""Create reproducible metric summaries for a validated TTPFTS p sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from .config import config_hash, default_output_root, expected_environments, load_config


METRICS = ("bernoulli", "jaccard", "misidentification")


def load_results(results_dir: Path, environments: list[str] | None = None) -> pd.DataFrame:
    """Load config-matching merged results without silently accepting stale files."""
    config = load_config()
    environments = (
        [item["name"] for item in expected_environments()]
        if environments is None
        else environments
    )
    frames = []
    for environment in environments:
        path = results_dir / f"{environment}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {
            "environment", "repetition", "arm_pull", "max_budget", "rho",
            "normalized_budget", "sweep_config_hash", *METRICS,
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        if set(frame["sweep_config_hash"]) != {config_hash(config)}:
            raise ValueError(f"{path}: p-sweep config hash mismatch")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def metric_summary(rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute one deterministic 95% repetition-level interval per curve point."""
    if metric not in METRICS:
        raise ValueError(f"unsupported metric {metric}")
    z = NormalDist().inv_cdf(0.975)
    records = []
    groups = rows.groupby(
        ["environment", "algorithm", "rho", "arm_pull", "max_budget", "normalized_budget"],
        sort=True,
    )
    for keys, group in groups:
        environment, algorithm, p, arm_pull, budget, normalized = keys
        values = group[metric].astype(float).to_numpy()
        n = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        se = std / n ** 0.5
        records.append(
            {
                "environment": environment,
                "algorithm": algorithm,
                "p": float(p),
                "arm_pull": int(arm_pull),
                "max_budget": int(budget),
                "normalized_budget": float(normalized),
                "metric": metric,
                "mean": mean,
                "std": std,
                "se": se,
                "lower": max(0.0, mean - z * se),
                "upper": min(1.0, mean + z * se),
                "n": n,
            }
        )
    return pd.DataFrame(records).sort_values(
        ["environment", "p", "normalized_budget"]
    ).reset_index(drop=True)


def endpoint_auc_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-trajectory endpoint values and normalized-budget AUCs."""
    records = []
    for metric in METRICS:
        for (environment, algorithm, p, repetition), group in rows.groupby(
            ["environment", "algorithm", "rho", "repetition"], sort=True
        ):
            curve = group.sort_values("normalized_budget")
            x = curve["normalized_budget"].astype(float).to_numpy()
            y = curve[metric].astype(float).to_numpy()
            if len(x) < 2 or x[0] != 0.0 or x[-1] != 1.0:
                raise ValueError(f"{environment}/p={p}/rep={repetition}: invalid normalized endpoint grid")
            records.append(
                {
                    "environment": environment,
                    "algorithm": algorithm,
                    "p": float(p),
                    "repetition": int(repetition),
                    "metric": metric,
                    "endpoint": float(y[-1]),
                    "normalized_budget_auc": float(np.trapz(y, x)),
                }
            )
    trajectory = pd.DataFrame(records)
    summary = (
        trajectory.groupby(["environment", "algorithm", "p", "metric"], sort=True)
        .agg(
            n=("repetition", "size"),
            endpoint_mean=("endpoint", "mean"),
            endpoint_std=("endpoint", "std"),
            auc_mean=("normalized_budget_auc", "mean"),
            auc_std=("normalized_budget_auc", "std"),
        )
        .reset_index()
    )
    comparison = (
        summary.groupby(["p", "metric"], sort=True)
        .agg(
            environments=("environment", "size"),
            equal_environment_endpoint_mean=("endpoint_mean", "mean"),
            equal_environment_auc_mean=("auc_mean", "mean"),
        )
        .reset_index()
    )
    return trajectory, summary, comparison


def write_analysis(results_dir: Path, output_dir: Path) -> None:
    """Write all raw/checkpoint/endpoint analysis tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_results(results_dir)
    rows.to_csv(output_dir / "trajectory_checkpoints.csv", index=False)
    for metric in METRICS:
        metric_summary(rows, metric).to_csv(output_dir / f"{metric}_summary.csv", index=False)
    trajectory, summary, comparison = endpoint_auc_summary(rows)
    trajectory.to_csv(output_dir / "trajectory_endpoint_auc.csv", index=False)
    summary.to_csv(output_dir / "endpoint_auc_summary.csv", index=False)
    comparison.to_csv(output_dir / "equal_environment_comparison.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_root = default_output_root(load_config())
    write_analysis(args.results_dir or output_root / "results", args.output_dir or output_root / "analysis")


if __name__ == "__main__":
    main()
