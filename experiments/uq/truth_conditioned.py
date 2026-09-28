#!/usr/bin/env python3
"""Generate truth-conditioned boundary-UQ supplementary analyses from saved states."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.final_config_loader import load_final_config
from momab_main import ENVIRONMENTS, _construct_environment
from runner.instance import compute_instance_hash

from .analyze import MEASURES, correlation_summary, evolution_summary, trajectory_correlations
from .config import config_hash, environment_config, load_config
from .metrics import (
    true_pareto_boundary_pairs,
    truth_conditioned_uncertainty_from_tposterior_state,
)
from .plot import plot_correlations, plot_evolution
from .replay import shard_paths


COLUMNS = [
    "environment", "instance_hash", "repetition", "arm_pull", "max_budget",
    "normalized_budget", "jaccard", "uq_available", "uq_reason",
    "bhattacharyya", "symmetric_kl", "entropy", "true_pareto_boundary_size",
    "true_challenger_size", "boundary_pairs", "uq_config_hash",
]


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-truth-uq-", suffix=".csv")
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def _environment_boundary(environment_name: str, config: dict) -> tuple[tuple[tuple[int, int], ...], str]:
    final_config = load_final_config()
    cls, _budget = ENVIRONMENTS[environment_name]
    environment = _construct_environment(cls, final_config["reward_model"]["distribution"])
    details = environment_config(environment_name, config)
    instance_hash = compute_instance_hash(environment)
    if instance_hash != details["instance_hash"]:
        raise ValueError(f"{environment_name}: frozen instance hash mismatch")
    return true_pareto_boundary_pairs(environment.arms, environment.pareto_indices), instance_hash


def derive_rows(results_dir: Path, run_dir: Path) -> tuple[pd.DataFrame, dict[str, list[list[int]]]]:
    """Recompute fixed-boundary UQ at every saved checkpoint."""
    config = load_config()
    digest = config_hash(config)
    environments = [item["name"] for item in load_final_config()["environments"]]
    all_rows: list[dict] = []
    boundaries: dict[str, list[list[int]]] = {}
    for environment_name in environments:
        details = environment_config(environment_name, config)
        boundary_pairs, instance_hash = _environment_boundary(environment_name, config)
        boundaries[environment_name] = [list(pair) for pair in boundary_pairs]
        current = pd.read_csv(results_dir / f"{environment_name}.csv")
        if set(current["uq_config_hash"].astype(str)) != {digest}:
            raise ValueError(f"{environment_name}: primary UQ config hash mismatch")
        if set(current["instance_hash"].astype(str)) != {instance_hash}:
            raise ValueError(f"{environment_name}: primary UQ instance hash mismatch")
        if len(current) != config["num_runs"] * len(details["grid"]):
            raise ValueError(f"{environment_name}: incomplete primary UQ result table")
        for repetition in range(config["num_runs"]):
            checkpoint_rows = current[current["repetition"] == repetition].sort_values("arm_pull")
            if checkpoint_rows["arm_pull"].tolist() != details["grid"]:
                raise ValueError(f"{environment_name}/rep{repetition}: grid mismatch")
            _csv_path, state_path = shard_paths(run_dir, environment_name, repetition)
            with np.load(state_path, allow_pickle=False) as state:
                if state["grid"].tolist() != details["grid"]:
                    raise ValueError(f"{environment_name}/rep{repetition}: state grid mismatch")
                if str(state["uq_config_hash"].item()) != digest:
                    raise ValueError(f"{environment_name}/rep{repetition}: state config hash mismatch")
                counts, means, m2 = state["n"], state["means"], state["m2"]
            for index, primary_row in enumerate(checkpoint_rows.itertuples(index=False)):
                pull = int(primary_row.arm_pull)
                if pull == 0:
                    available, reason, result = False, "budget_zero", None
                else:
                    result = truth_conditioned_uncertainty_from_tposterior_state(
                        counts[index], means[index], m2[index], boundary_pairs, alpha=0.0
                    )
                    available, reason = result.available, result.reason
                all_rows.append(
                    {
                        "environment": environment_name,
                        "instance_hash": instance_hash,
                        "repetition": repetition,
                        "arm_pull": pull,
                        "max_budget": details["budget"],
                        "normalized_budget": pull / details["budget"],
                        "jaccard": float(primary_row.jaccard),
                        "uq_available": int(available),
                        "uq_reason": reason,
                        "bhattacharyya": "" if not available else result.bhattacharyya,
                        "symmetric_kl": "" if not available else result.symmetric_kl,
                        "entropy": "" if not available else result.entropy,
                        "true_pareto_boundary_size": 0 if result is None else result.first_front_size,
                        "true_challenger_size": 0 if result is None else result.second_front_size,
                        "boundary_pairs": json.dumps(boundaries[environment_name], separators=(",", ":")),
                        "uq_config_hash": digest,
                    }
                )
    frame = pd.DataFrame(all_rows, columns=COLUMNS)
    for measure in MEASURES:
        frame[measure] = pd.to_numeric(frame[measure], errors="coerce")
    return frame, boundaries


def write_supplement(results_dir: Path, run_dir: Path, output_dir: Path) -> None:
    """Write supplemental tables and six fixed-boundary UQ figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, boundaries = derive_rows(results_dir, run_dir)
    _atomic_csv(output_dir / "truth_conditioned_checkpoints.csv", rows.to_dict("records"))
    (output_dir / "truth_conditioned_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "truth-conditioned-boundary-uq-v1",
                "uq_config_hash": config_hash(load_config()),
                "source_results_dir": str(results_dir),
                "source_run_dir": str(run_dir),
                "definition": (
                    "For every true Pareto arm, average posterior-moment UQ "
                    "against each direct-gap-minimizing true dominated challenger."
                ),
                "boundary_pairs": boundaries,
                "unavailable_environments": {
                    environment: "no_true_dominated_boundary"
                    for environment, pairs in boundaries.items() if not pairs
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for measure in MEASURES:
        evolution = evolution_summary(rows, measure)
        correlations = trajectory_correlations(rows, measure)
        evolution.to_csv(output_dir / f"{measure}_evolution.csv", index=False)
        correlations.to_csv(output_dir / f"{measure}_trajectory_correlations.csv", index=False)
        correlation_summary(correlations).to_csv(output_dir / f"{measure}_correlation_summary.csv", index=False)
        plot_evolution(evolution, measure, output_dir / f"{measure}_evolution.png")
        plot_correlations(correlations, measure, output_dir / f"{measure}_jaccard_correlation.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_supplement(args.results_dir, args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
