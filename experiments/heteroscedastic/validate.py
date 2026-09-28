#!/usr/bin/env python3
"""Independent integrity validation for heteroscedastic pilot results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from runner.schema import SCHEMA_COLUMNS, decode_int_set, recompute_metrics
from runner.instance import compute_instance_hash

from .instances import FrozenHeteroscedasticEnvironment
from .pilot import environment_key, load_pilot_config, pilot_candidates, pilot_config_hash


def _fail(problems: list[str], message: str) -> None:
    problems.append(message)


def validate(results_dir: Path) -> list[str]:
    """Return all detected integrity violations; an empty list is a pass."""
    problems: list[str] = []
    config = load_pilot_config()
    candidates = pilot_candidates(config)
    expected = {environment_key(candidate): candidate for candidate in candidates}
    manifest_path = results_dir / "run" / "pilot_manifest.json"
    if not manifest_path.exists():
        return [f"missing {manifest_path}"]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("pilot_config_hash") != pilot_config_hash(config):
        _fail(problems, "pilot config hash mismatch")
    if manifest.get("algorithms") != config["algorithms"]:
        _fail(problems, "algorithm list mismatch")
    if set(manifest.get("instances", {})) != set(expected):
        _fail(problems, "pilot manifest instance coverage differs from approved cohort")

    total_rows = 0
    for key, candidate in expected.items():
        metadata = manifest.get("instances", {}).get(key, {})
        environment = FrozenHeteroscedasticEnvironment(candidate.means, candidate.stds)
        if metadata.get("budget") != candidate.budget or metadata.get("grid") != list(candidate.grid):
            _fail(problems, f"{key}: manifest budget/grid mismatch")
        if metadata.get("hardness") != candidate.hardness.hardness:
            _fail(problems, f"{key}: manifest hardness mismatch")
        path = results_dir / f"{key}.csv"
        if not path.exists():
            _fail(problems, f"{key}: missing merged CSV")
            continue
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != SCHEMA_COLUMNS:
                _fail(problems, f"{key}: schema mismatch")
                continue
            rows = list(reader)
        total_rows += len(rows)
        grid = list(candidate.grid)
        expected_count = len(config["algorithms"]) * len(grid)
        if len(rows) != expected_count:
            _fail(problems, f"{key}: row count {len(rows)}, expected {expected_count}")
        seen = set()
        expected_hash = compute_instance_hash(environment)
        for row in rows:
            row_key = (row["algorithm"], int(row["repetition"]), int(row["arm_pull"]))
            if row_key in seen:
                _fail(problems, f"{key}: duplicate row {row_key}")
            seen.add(row_key)
            if row["environment"] != key or row["algorithm"] not in config["algorithms"]:
                _fail(problems, f"{key}: unexpected row identity {row_key}")
            if int(row["repetition"]) != 0 or int(row["max_budget"]) != candidate.budget:
                _fail(problems, f"{key}: repetition/budget mismatch at {row_key}")
            if json.loads(row["evaluation_grid"]) != grid:
                _fail(problems, f"{key}: evaluation grid mismatch at {row_key}")
            if row["instance_hash"] != expected_hash:
                _fail(problems, f"{key}: instance hash mismatch at {row_key}")
            recommendation = decode_int_set(row["recommendation"])
            true_pareto = decode_int_set(row["true_pareto"])
            if true_pareto != set(candidate.pareto_indices.tolist()):
                _fail(problems, f"{key}: true Pareto mismatch at {row_key}")
            if any(arm < 0 or arm >= candidate.means.shape[0] for arm in recommendation):
                _fail(problems, f"{key}: invalid recommendation arm at {row_key}")
            metrics = recompute_metrics(recommendation, true_pareto, candidate.means.shape[0])
            for metric, value in metrics.items():
                if not math.isclose(float(row[metric]), value, rel_tol=0.0, abs_tol=1e-12):
                    _fail(problems, f"{key}: {metric} mismatch at {row_key}")
        for algorithm in config["algorithms"]:
            pulls = sorted(pull for alg, repetition, pull in seen if alg == algorithm and repetition == 0)
            if pulls != grid:
                _fail(problems, f"{key}/{algorithm}: grid {pulls}, expected {grid}")
    expected_total = len(expected) * len(config["algorithms"]) * 11
    if total_rows != expected_total:
        _fail(problems, f"total rows {total_rows}, expected {expected_total}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    problems = validate(args.results_dir)
    if problems:
        print("VALIDATION FAILED")
        for problem in problems[:50]:
            print(f"- {problem}")
        return 1
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
