#!/usr/bin/env python3
"""Create deterministic equal-instance bootstrap summaries for ensemble results."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ALGORITHMS, config_hash, default_output_root, load_config, sha256_file
from .merge import RESULT_FILE
from .validate import validate


METRICS = ("jaccard", "bernoulli", "misidentification")


def _bootstrap_indices(n_instances: int, replicates: int, seed: int) -> np.ndarray:
    if n_instances < 1 or replicates < 1:
        raise ValueError("bootstrap requires positive instance and replicate counts")
    return np.random.default_rng(seed).integers(0, n_instances, size=(replicates, n_instances), endpoint=False)


def _interval(values: np.ndarray, indices: np.ndarray, confidence: float) -> tuple[float, float, float]:
    """Return exact mean and percentile whole-instance bootstrap interval."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size != indices.shape[1]:
        raise ValueError("values must have one entry for each bootstrap instance")
    replicates = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return (
        float(values.mean()),
        float(np.quantile(replicates, tail)),
        float(np.quantile(replicates, 1.0 - tail)),
    )


def _matrix(
    rows: pd.DataFrame,
    metric: str,
    algorithm: str,
    grid_index: int,
    instance_ids: np.ndarray,
) -> np.ndarray:
    selected = rows[
        (rows["algorithm"] == algorithm)
        & (rows["grid_index"] == grid_index)
    ][["instance_id", metric]]
    by_instance = selected.set_index("instance_id")[metric]
    missing = set(instance_ids) - set(by_instance.index)
    if missing:
        raise ValueError(f"{algorithm}/grid={grid_index}: missing instances, first={sorted(missing)[:3]}")
    if by_instance.index.duplicated().any():
        raise ValueError(f"{algorithm}/grid={grid_index}: duplicate instance rows")
    return by_instance.loc[instance_ids].to_numpy(dtype=float)


def metric_summary(
    rows: pd.DataFrame,
    metric: str,
    *,
    instance_ids: np.ndarray,
    bootstrap_indices: np.ndarray,
    confidence: float,
) -> pd.DataFrame:
    """Summarize every curve point, weighting each frozen instance once."""
    if metric not in METRICS:
        raise ValueError(f"unsupported metric {metric}")
    records = []
    grid = rows[["grid_index", "arm_pull", "normalized_budget"]].drop_duplicates().sort_values("grid_index")
    if len(grid) != 21 or grid["grid_index"].tolist() != list(range(21)):
        raise ValueError("results must contain the exact 21-point normalized grid")
    for algorithm in ALGORITHMS:
        for grid_row in grid.itertuples(index=False):
            values = _matrix(rows, metric, algorithm, int(grid_row.grid_index), instance_ids)
            mean, lower, upper = _interval(values, bootstrap_indices, confidence)
            records.append(
                {
                    "algorithm": algorithm,
                    "grid_index": int(grid_row.grid_index),
                    "arm_pull_normalized": float(grid_row.normalized_budget),
                    "mean": mean,
                    "lower": lower,
                    "upper": upper,
                    "n_instances": len(instance_ids),
                    "bootstrap_replicates": bootstrap_indices.shape[0],
                    "metric": metric,
                }
            )
    return pd.DataFrame(records).sort_values(["algorithm", "grid_index"]).reset_index(drop=True)


def endpoint_auc_summary(
    rows: pd.DataFrame,
    *,
    instance_ids: np.ndarray,
    bootstrap_indices: np.ndarray,
    confidence: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write per-instance endpoints/AUCs and algorithm-level bootstrap summaries."""
    per_instance: list[dict[str, float | int | str]] = []
    aggregate: list[dict[str, float | int | str]] = []
    for metric in METRICS:
        for algorithm in ALGORITHMS:
            ordered = rows[rows["algorithm"] == algorithm].pivot(
                index="instance_id", columns="grid_index", values=metric
            ).reindex(index=instance_ids, columns=range(21))
            if ordered.isna().any().any():
                raise ValueError(f"{algorithm}/{metric}: cannot calculate endpoint/AUC on incomplete curve")
            normalized = rows[rows["algorithm"] == algorithm].drop_duplicates("grid_index").set_index("grid_index")
            x = normalized.loc[range(21), "normalized_budget"].to_numpy(dtype=float)
            values = ordered.to_numpy(dtype=float)
            endpoints = values[:, -1]
            aucs = np.trapezoid(values, x=x, axis=1)
            for instance_id, endpoint, auc in zip(instance_ids, endpoints, aucs):
                per_instance.append(
                    {
                        "instance_id": int(instance_id),
                        "algorithm": algorithm,
                        "metric": metric,
                        "endpoint": float(endpoint),
                        "normalized_budget_auc": float(auc),
                    }
                )
            for statistic, statistic_values in (("endpoint", endpoints), ("normalized_budget_auc", aucs)):
                mean, lower, upper = _interval(statistic_values, bootstrap_indices, confidence)
                aggregate.append(
                    {
                        "algorithm": algorithm,
                        "metric": metric,
                        "statistic": statistic,
                        "mean": mean,
                        "lower": lower,
                        "upper": upper,
                        "n_instances": len(instance_ids),
                        "bootstrap_replicates": bootstrap_indices.shape[0],
                    }
                )
    return pd.DataFrame(per_instance), pd.DataFrame(aggregate)


def paired_contrasts(
    rows: pd.DataFrame,
    metric: str,
    *,
    instance_ids: np.ndarray,
    bootstrap_indices: np.ndarray,
    confidence: float,
) -> pd.DataFrame:
    """Compute A-minus-B contrasts paired by frozen instance at every checkpoint."""
    records = []
    for algorithm_a, algorithm_b in itertools.combinations(ALGORITHMS, 2):
        for grid_index in range(21):
            values = _matrix(rows, metric, algorithm_a, grid_index, instance_ids) - _matrix(
                rows, metric, algorithm_b, grid_index, instance_ids
            )
            mean, lower, upper = _interval(values, bootstrap_indices, confidence)
            normalized = rows.loc[
                (rows["algorithm"] == algorithm_a) & (rows["grid_index"] == grid_index),
                "normalized_budget",
            ].iloc[0]
            records.append(
                {
                    "algorithm_a": algorithm_a,
                    "algorithm_b": algorithm_b,
                    "grid_index": grid_index,
                    "normalized_budget": float(normalized),
                    "metric": metric,
                    "mean_a_minus_b": mean,
                    "lower": lower,
                    "upper": upper,
                    "n_instances": len(instance_ids),
                    "bootstrap_replicates": bootstrap_indices.shape[0],
                }
            )
    return pd.DataFrame(records)


def _validate_result_columns(rows: pd.DataFrame) -> None:
    required = {
        "environment", "algorithm", "grid_index", "normalized_budget", "instance_hash",
        "ensemble_config_hash", "jaccard", "bernoulli", "misidentification",
    }
    missing = required - set(rows)
    if missing:
        raise ValueError(f"merged results miss analysis columns: {sorted(missing)}")
    if set(rows["algorithm"]) != set(ALGORITHMS):
        raise ValueError("merged results contain an unexpected algorithm set")
    if rows["environment"].duplicated().all():
        raise ValueError("merged result rows must retain environment names")


def write_analysis(run_dir: Path, results_dir: Path, output_dir: Path) -> None:
    """Validate, analyze, and persist every table needed to reproduce plots."""
    problems = validate(run_dir, results_dir)
    if problems:
        raise ValueError("refusing to analyze invalid results:\n- " + "\n- ".join(problems[:20]))
    config = load_config()
    rows = pd.read_csv(results_dir / RESULT_FILE)
    _validate_result_columns(rows)
    rows["instance_id"] = rows["environment"].str.removeprefix("HeterogeneousInstance").astype(int)
    if rows["instance_id"].nunique() != config["cohort"]["accepted_instances"]:
        raise ValueError("merged results do not contain exactly the approved instance cohort")
    instance_ids = np.sort(rows["instance_id"].unique())
    bootstrap = _bootstrap_indices(
        len(instance_ids), config["bootstrap"]["replicates"], config["bootstrap"]["seed"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "whole_instance_bootstrap_indices.npy", bootstrap)
    rows.to_csv(output_dir / "validated_merged_results.csv", index=False)
    for metric in METRICS:
        metric_summary(
            rows, metric, instance_ids=instance_ids, bootstrap_indices=bootstrap,
            confidence=config["bootstrap"]["confidence"],
        ).to_csv(output_dir / f"{metric}_summary.csv", index=False)
        paired_contrasts(
            rows, metric, instance_ids=instance_ids, bootstrap_indices=bootstrap,
            confidence=config["bootstrap"]["confidence"],
        ).to_csv(output_dir / f"{metric}_paired_contrasts.csv", index=False)
    per_instance, endpoint = endpoint_auc_summary(
        rows, instance_ids=instance_ids, bootstrap_indices=bootstrap,
        confidence=config["bootstrap"]["confidence"],
    )
    per_instance.to_csv(output_dir / "per_instance_endpoint_auc.csv", index=False)
    endpoint.to_csv(output_dir / "endpoint_auc_summary.csv", index=False)
    manifest = {
        "schema_version": "heterogeneous-ensemble-analysis-v1",
        "ensemble_config_hash": config_hash(config),
        "result_sha256": sha256_file(results_dir / RESULT_FILE),
        "instance_count": len(instance_ids),
        "bootstrap": config["bootstrap"],
        "metrics": list(METRICS),
        "algorithm_order": list(ALGORITHMS),
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
    write_analysis(
        args.run_dir or root / "run",
        args.results_dir or root / "results",
        args.output_dir or root / "analysis",
    )


if __name__ == "__main__":
    main()
