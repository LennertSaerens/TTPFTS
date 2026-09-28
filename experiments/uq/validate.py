#!/usr/bin/env python3
"""Independently validate UQ replay state, rows, coverage, and final-run identity."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from .config import config_hash, environment_config, load_config
from .metrics import uncertainty_from_tposterior_state
from .replay import UQ_COLUMNS, shard_is_complete, shard_paths


IDENTITY_COLUMNS = (
    "environment", "instance_hash", "algorithm", "posterior_model", "repetition",
    "arm_pull", "max_budget", "evaluation_grid", "rho", "master_seed", "bernoulli",
    "jaccard", "misidentification", "recommendation", "true_pareto",
)
MEASURES = ("bhattacharyya", "symmetric_kl", "entropy")


def _add(problems: list[str], message: str) -> None:
    problems.append(message)


def _load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _matches_float(actual: str, expected: float, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)
    except ValueError:
        return False


def _validate_row_state(row: dict, n: np.ndarray, means: np.ndarray, m2: np.ndarray, problems: list[str], label: str) -> None:
    pull = int(row["arm_pull"])
    if pull == 0:
        if row["uq_available"] != "0" or row["uq_reason"] != "budget_zero":
            _add(problems, f"{label}: budget zero must be unavailable")
        if any(row[measure] != "" for measure in MEASURES):
            _add(problems, f"{label}: budget zero has UQ values")
        return
    result = uncertainty_from_tposterior_state(n, means, m2, alpha=0.0)
    if int(row["uq_available"]) != int(result.available) or row["uq_reason"] != result.reason:
        _add(problems, f"{label}: UQ availability/reason mismatch")
    if int(row["first_front_size"]) != result.first_front_size or int(row["second_front_size"]) != result.second_front_size:
        _add(problems, f"{label}: front sizes mismatch")
    if result.available:
        expected_values = {
            "bhattacharyya": result.bhattacharyya,
            "symmetric_kl": result.symmetric_kl,
            "entropy": result.entropy,
        }
        for measure, expected in expected_values.items():
            if not _matches_float(row[measure], expected):
                _add(problems, f"{label}: {measure} mismatch")
    elif any(row[measure] != "" for measure in MEASURES):
        _add(problems, f"{label}: unavailable UQ row has a value")


def validate(
    run_dir: Path,
    final_results_dir: Path,
    environments: list[str] | None = None,
    repetitions: list[int] | None = None,
) -> list[str]:
    """Return all violations; an empty return means full validation passed."""
    config = load_config()
    digest = config_hash(config)
    if environments is None:
        from experiments.final_config_loader import load_final_config

        environments = [item["name"] for item in load_final_config()["environments"]]
    problems: list[str] = []
    repetitions = list(range(config["num_runs"])) if repetitions is None else repetitions
    for environment in environments:
        details = environment_config(environment, config)
        grid = details["grid"]
        final_path = final_results_dir / f"{environment}.csv"
        if not final_path.is_file():
            _add(problems, f"{environment}: missing validated final results")
            continue
        final_rows = {
            (int(row["repetition"]), int(row["arm_pull"])): row
            for row in _load_rows(final_path)
            if row["algorithm"] == "TTPFTS_UNKNOWN_VAR"
        }
        expected_keys = {(rep, pull) for rep in repetitions for pull in grid}
        if not expected_keys.issubset(final_rows):
            _add(problems, f"{environment}: final TTPFTS lacks requested replay coverage")
            continue
        for repetition in repetitions:
            label = f"{environment}/rep{repetition:05d}"
            if not shard_is_complete(run_dir, environment, repetition, grid, digest):
                _add(problems, f"{label}: incomplete/stale shard")
                continue
            csv_path, state_path = shard_paths(run_dir, environment, repetition)
            rows = _load_rows(csv_path)
            if len(rows) != len(grid) or (rows and set(rows[0]) != set(UQ_COLUMNS)):
                _add(problems, f"{label}: CSV schema or row count mismatch")
                continue
            try:
                with np.load(state_path, allow_pickle=False) as state:
                    n, means, m2 = state["n"], state["means"], state["m2"]
                    saved_grid = state["grid"].tolist()
            except (OSError, ValueError, KeyError) as exc:
                _add(problems, f"{label}: unreadable state shard: {exc}")
                continue
            if saved_grid != grid or n.shape != means.shape or means.shape != m2.shape:
                _add(problems, f"{label}: state shape/grid mismatch")
                continue
            if n.shape[0] != len(grid) or n.shape[1:] != (details["num_arms"], details["num_objectives"]):
                _add(problems, f"{label}: state dimension mismatch")
                continue
            if not np.all(np.isfinite(means)) or not np.all(np.isfinite(m2)) or np.any(m2 < 0) or np.any(n < 0):
                _add(problems, f"{label}: invalid posterior sufficient statistics")
                continue
            for index, (row, pull) in enumerate(zip(rows, grid)):
                row_label = f"{label}/t={pull}"
                if int(row["arm_pull"]) != pull or row["uq_config_hash"] != digest:
                    _add(problems, f"{row_label}: grid/config mismatch")
                    continue
                if not _matches_float(row["normalized_budget"], pull / details["budget"]):
                    _add(problems, f"{row_label}: normalized budget mismatch")
                expected_total = pull * details["num_objectives"]
                if int(n[index].sum()) != expected_total:
                    _add(problems, f"{row_label}: posterior count does not equal pull count")
                final = final_rows[(repetition, pull)]
                for column in IDENTITY_COLUMNS:
                    if row[column] != final[column]:
                        _add(problems, f"{row_label}: final identity mismatch in {column}")
                _validate_row_state(row, n[index], means[index], m2[index], problems, row_label)
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--final-results-dir", type=Path, required=True)
    parser.add_argument("--environments", default=None)
    parser.add_argument("--repetitions", default=None, help="Comma-separated repetition subset for a smoke run.")
    args = parser.parse_args()
    environments = args.environments.split(",") if args.environments else None
    repetitions = None if args.repetitions is None else [int(value) for value in args.repetitions.split(",")]
    problems = validate(args.run_dir, args.final_results_dir, environments, repetitions)
    if problems:
        print("UQ VALIDATION FAILED")
        for problem in problems[:100]:
            print(f"- {problem}")
        return 1
    print("UQ VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
