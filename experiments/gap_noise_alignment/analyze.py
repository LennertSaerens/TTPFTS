#!/usr/bin/env python3
"""Create repetition-bootstrap metric, contrast, and EGE-trace summaries."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ALGORITHMS, CONDITIONS, config_hash, default_output_root, load_config, sha256_file
from .merge import RESULT_FILE, TRACE_FILE
from .validate import validate


METRICS = ("jaccard", "bernoulli", "misidentification")


def bootstrap_indices(n: int, replicates: int, seed: int) -> np.ndarray:
    if n < 2 or replicates < 1:
        raise ValueError("bootstrap requires at least two observations and one replicate")
    return np.random.default_rng(seed).integers(0, n, size=(replicates, n), endpoint=False)


def interval(values: np.ndarray, indices: np.ndarray, confidence: float) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size != indices.shape[1]:
        raise ValueError("bootstrap values must be a one-dimensional complete repetition vector")
    draws = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(values.mean()), float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))


def condition_metric_summary(rows: pd.DataFrame, metric: str, indices: np.ndarray, confidence: float) -> pd.DataFrame:
    """Summarize one metric per condition/algorithm/checkpoint over paired repetitions."""
    records = []
    for condition in CONDITIONS:
        for algorithm in ALGORITHMS:
            for grid_index in range(21):
                point = rows[
                    (rows.condition == condition)
                    & (rows.algorithm == algorithm)
                    & (rows.grid_index == grid_index)
                ].sort_values("repetition")
                if point.repetition.tolist() != list(range(indices.shape[1])):
                    raise ValueError(f"{condition}/{algorithm}/{grid_index}: incomplete repetition coverage")
                mean, lower, upper = interval(point[metric].to_numpy(float), indices, confidence)
                records.append(
                    {
                        "condition": condition,
                        "algorithm": algorithm,
                        "grid_index": grid_index,
                        "normalized_budget": float(point.normalized_budget.iloc[0]),
                        "metric": metric,
                        "mean": mean,
                        "lower": lower,
                        "upper": upper,
                        "n_repetitions": indices.shape[1],
                        "bootstrap_replicates": indices.shape[0],
                    }
                )
    return pd.DataFrame(records)


def paired_condition_contrasts(rows: pd.DataFrame, metric: str, indices: np.ndarray, confidence: float) -> pd.DataFrame:
    """Compute small-gap-noisy minus large-gap-noisy contrasts paired by repetition."""
    records = []
    for algorithm in ALGORITHMS:
        for grid_index in range(21):
            selected = rows[(rows.algorithm == algorithm) & (rows.grid_index == grid_index)]
            pivot = selected.pivot(index="repetition", columns="condition", values=metric).reindex(range(indices.shape[1]))
            if list(pivot.columns) != list(CONDITIONS) or pivot.isna().any().any():
                raise ValueError(f"{algorithm}/{grid_index}: incomplete paired condition coverage")
            values = (pivot["small_gap_noisy"] - pivot["large_gap_noisy"]).to_numpy(float)
            mean, lower, upper = interval(values, indices, confidence)
            records.append(
                {
                    "algorithm": algorithm,
                    "grid_index": grid_index,
                    "normalized_budget": float(selected.normalized_budget.iloc[0]),
                    "metric": metric,
                    "small_gap_noisy_minus_large_gap_noisy": mean,
                    "lower": lower,
                    "upper": upper,
                    "n_paired_repetitions": indices.shape[1],
                    "bootstrap_replicates": indices.shape[0],
                }
            )
    return pd.DataFrame(records)


def trace_summary(traces: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe EGE classification order/error by condition, noise, and true gap."""
    traces = traces.copy()
    traces["is_final_accept"] = traces["decision"].eq("accept_final")
    traces["gap_rank_quintile"] = pd.qcut(
        traces["noise_rank"], q=5, labels=False, duplicates="drop"
    )
    grouped = (
        traces.groupby(
            ["condition", "algorithm", "grid_index", "noise_rank", "std", "true_pareto"],
            sort=True,
        )
        .agg(
            classifications=("arm", "size"),
            decision_accuracy=("decision_correct", "mean"),
            mean_classification_round=("round", "mean"),
            mean_samples_at_decision=("samples_at_decision", "mean"),
        )
        .reset_index()
    )
    order = []
    for keys, group in traces.groupby(["condition", "algorithm", "repetition", "grid_index"], sort=True):
        correlation = group["true_gap"].corr(group["round"], method="spearman")
        order.append(
            {
                "condition": keys[0],
                "algorithm": keys[1],
                "repetition": keys[2],
                "grid_index": keys[3],
                "true_gap_vs_classification_round_spearman": correlation,
            }
        )
    return grouped, pd.DataFrame(order)


def write_analysis(run_dir: Path, results_dir: Path, output_dir: Path) -> None:
    """Validate full output and write all reproducible numerical summaries."""
    problems = validate(run_dir, results_dir)
    if problems:
        raise ValueError("refusing to analyze invalid results:\n- " + "\n- ".join(problems[:20]))
    config = load_config()
    rows = pd.read_csv(results_dir / RESULT_FILE)
    traces = pd.read_csv(results_dir / TRACE_FILE)
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = bootstrap_indices(config["repetitions"], config["bootstrap"]["replicates"], config["bootstrap"]["seed"])
    np.save(output_dir / "repetition_bootstrap_indices.npy", indices)
    rows.to_csv(output_dir / "validated_metric_rows.csv", index=False)
    traces.to_csv(output_dir / "validated_ege_trace_rows.csv", index=False)
    for metric in METRICS:
        condition_metric_summary(rows, metric, indices, config["bootstrap"]["confidence"]).to_csv(
            output_dir / f"{metric}_condition_summary.csv", index=False
        )
        paired_condition_contrasts(rows, metric, indices, config["bootstrap"]["confidence"]).to_csv(
            output_dir / f"{metric}_paired_condition_contrast.csv", index=False
        )
    classification, ordering = trace_summary(traces)
    classification.to_csv(output_dir / "ege_classification_summary.csv", index=False)
    ordering.to_csv(output_dir / "ege_ordering_summary.csv", index=False)
    manifest = {
        "schema_version": "gap-noise-alignment-analysis-v1",
        "config_hash": config_hash(config),
        "metric_results_sha256": sha256_file(results_dir / RESULT_FILE),
        "trace_results_sha256": sha256_file(results_dir / TRACE_FILE),
        "bootstrap": config["bootstrap"],
        "conditions": list(CONDITIONS),
        "algorithms": list(ALGORITHMS),
        "metrics": list(METRICS),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = default_output_root(load_config())
    write_analysis(args.run_dir or root / "run", args.results_dir or root / "results", args.output_dir or root / "analysis")


if __name__ == "__main__":
    main()
