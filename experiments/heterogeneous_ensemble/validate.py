#!/usr/bin/env python3
"""Independently validate heterogeneous-ensemble rows, tasks, metrics, and provenance."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from runner.ege_allocation import expected_ege_draws
from runner.schema import decode_int_set, encode_int_set, validate_row_metrics
from runner.seeding import derive_seed

from .config import ALGORITHMS, config_hash, default_output_root, load_config, sha256_file
from .instances import MANIFEST_PATH, FrozenInstance, load_frozen_instances
from .merge import RESULT_FILE, RUN_MANIFEST_FILE
from .run import (
    ANYTIME_ALGORITHMS,
    EGE_ALGORITHMS,
    RESULT_COLUMNS,
    anytime_is_complete,
    anytime_paths,
    ege_is_complete,
    ege_path,
)


def _add(problems: list[str], message: str) -> None:
    problems.append(message)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _expected_noise_metadata(instance: FrozenInstance, algorithm: str) -> tuple[str, str, str, int]:
    if algorithm == "APE":
        return ("", "global_max_standard_deviation_bound", str(float(np.max(instance.stds))), 0)
    if algorithm == "TTPFTS_UNKNOWN_VAR":
        return ("TPosterior", "unknown_variance_tposterior", "", 4)
    if algorithm == "Uniform":
        return ("", "none", "", 0)
    if algorithm in EGE_ALGORITHMS:
        return ("", "none", "", 0)
    raise ValueError(algorithm)


def _validate_row(
    row: dict[str, str],
    instance: FrozenInstance,
    algorithm: str,
    arm_pull: int,
    grid_index: int,
    digest: str,
    config: dict,
    problems: list[str],
) -> None:
    label = f"{instance.name}/{algorithm}/t={arm_pull}"
    if set(row) != set(RESULT_COLUMNS):
        _add(problems, f"{label}: schema mismatch")
        return
    execution_kind = "anytime" if algorithm in ANYTIME_ALGORITHMS else "ege"
    expected = {
        "environment": instance.name,
        "instance_hash": instance.instance_hash,
        "algorithm": algorithm,
        "repetition": "0",
        "arm_pull": str(arm_pull),
        "max_budget": str(instance.budget),
        "evaluation_grid": encode_int_set(instance.grid),
        "master_seed": str(config["master_seed"]),
        "proposal_id": str(instance.proposal_id),
        "num_arms": str(instance.num_arms),
        "num_objectives": str(instance.num_objectives),
        "pareto_size": str(len(instance.pareto_indices)),
        "grid_index": str(grid_index),
        "execution_kind": execution_kind,
        "ensemble_config_hash": digest,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            _add(problems, f"{label}: {key}={row.get(key)!r}, expected {value!r}")
    try:
        if not math.isclose(float(row["hardness"]), instance.hardness.hardness, rel_tol=0.0, abs_tol=0.0):
            _add(problems, f"{label}: hardness mismatch")
        if not math.isclose(float(row["normalized_budget"]), arm_pull / instance.budget, rel_tol=0.0, abs_tol=1e-12):
            _add(problems, f"{label}: normalized budget mismatch")
    except (KeyError, ValueError):
        _add(problems, f"{label}: malformed numeric metadata")
    posterior, knowledge, ape_bound, warmup = _expected_noise_metadata(instance, algorithm)
    if row.get("posterior_model") != posterior:
        _add(problems, f"{label}: posterior model mismatch")
    if row.get("noise_knowledge") != knowledge:
        _add(problems, f"{label}: noise-knowledge metadata mismatch")
    if row.get("ape_sigma_bound") != ape_bound:
        _add(problems, f"{label}: APE sigma-bound metadata mismatch")
    if row.get("warmup_pulls_per_arm") != str(warmup):
        _add(problems, f"{label}: warm-up metadata mismatch")
    if algorithm == "TTPFTS_UNKNOWN_VAR" and row.get("rho") != "0.5":
        _add(problems, f"{label}: TTPFTS rho mismatch")
    if algorithm != "TTPFTS_UNKNOWN_VAR" and row.get("rho") != "":
        _add(problems, f"{label}: non-TTPFTS rho must be empty")
    expected_seed_parts = (
        (config["master_seed"], "heterogeneous-ensemble", instance.instance_hash, algorithm)
        if algorithm in ANYTIME_ALGORITHMS
        else (config["master_seed"], "heterogeneous-ensemble", instance.instance_hash, algorithm, "budget", arm_pull)
    )
    if row.get("algorithm_seed") != str(derive_seed(*expected_seed_parts)):
        _add(problems, f"{label}: algorithm seed mismatch")
    try:
        recommendation = decode_int_set(row["recommendation"])
        true_pareto = decode_int_set(row["true_pareto"])
        if not recommendation <= set(range(instance.num_arms)):
            _add(problems, f"{label}: recommendation contains invalid arm")
        if true_pareto != set(instance.pareto_indices.tolist()):
            _add(problems, f"{label}: true Pareto set mismatch")
        for problem in validate_row_metrics(row, instance.num_arms, atol=1e-12):
            _add(problems, f"{label}: {problem}")
    except (KeyError, ValueError, TypeError) as exc:
        _add(problems, f"{label}: invalid metric/set encoding: {exc}")
    try:
        draws = int(row["actual_reward_draws"])
        expected_draws = arm_pull if algorithm in ANYTIME_ALGORITHMS else expected_ege_draws(
            algorithm, arm_pull, instance.num_arms
        )
        if draws != expected_draws:
            _add(problems, f"{label}: actual reward draws={draws}, expected {expected_draws}")
    except (KeyError, ValueError):
        _add(problems, f"{label}: malformed actual reward draws")


def _expected_shard_paths(run_dir: Path, instances: list[FrozenInstance]) -> set[Path]:
    paths = set()
    for instance in instances:
        for algorithm in ALGORITHMS:
            if algorithm in ANYTIME_ALGORITHMS:
                csv_path, _state_path = anytime_paths(run_dir, instance, algorithm)
                paths.add(csv_path)
            else:
                paths.update(ege_path(run_dir, instance, algorithm, budget) for budget in instance.grid)
    return paths


def validate(
    run_dir: Path,
    results_dir: Path,
    *,
    instance_ids: list[int] | None = None,
    require_manifest: bool = True,
) -> list[str]:
    """Return all result/candidate violations. Empty means full validation passed."""
    config = load_config()
    digest = config_hash(config)
    instances = load_frozen_instances(instance_ids)
    problems: list[str] = []
    result_path = results_dir / RESULT_FILE
    if not result_path.is_file():
        return [f"missing merged result file {result_path}"]
    rows = _read_rows(result_path)
    expected_keys = {
        (instance.instance_id, algorithm, index)
        for instance in instances
        for algorithm in ALGORITHMS
        for index in range(len(instance.grid))
    }
    observed: dict[tuple[int, str, int], dict[str, str]] = {}
    instance_by_name = {instance.name: instance for instance in instances}
    for row in rows:
        try:
            instance = instance_by_name[row["environment"]]
            key = (instance.instance_id, row["algorithm"], int(row["grid_index"]))
        except (KeyError, ValueError):
            _add(problems, "merged results contain an unrecognized row identity")
            continue
        if key in observed:
            _add(problems, f"duplicate merged row {key}")
        observed[key] = row
    if set(observed) != expected_keys:
        missing, extra = expected_keys - set(observed), set(observed) - expected_keys
        if missing:
            _add(problems, f"missing merged rows, first={sorted(missing)[:3]}")
        if extra:
            _add(problems, f"unexpected merged rows, first={sorted(extra)[:3]}")

    for instance in instances:
        for algorithm in ALGORITHMS:
            if algorithm in ANYTIME_ALGORITHMS:
                if not anytime_is_complete(run_dir, instance, algorithm, digest):
                    _add(problems, f"{instance.name}/{algorithm}: incomplete or stale anytime shard")
            else:
                for budget in instance.grid:
                    if not ege_is_complete(run_dir, instance, algorithm, budget, digest):
                        _add(problems, f"{instance.name}/{algorithm}/budget={budget}: incomplete or stale EGE shard")
            for index, arm_pull in enumerate(instance.grid):
                row = observed.get((instance.instance_id, algorithm, index))
                if row is not None:
                    _validate_row(row, instance, algorithm, arm_pull, index, digest, config, problems)
    expected_paths = _expected_shard_paths(run_dir, instances)
    shard_root = run_dir / "shards"
    actual_paths = {
        path
        for path in shard_root.rglob("*.csv")
        if not path.name.startswith(".tmp-")
    } if shard_root.is_dir() else set()
    unexpected = actual_paths - expected_paths
    missing_paths = expected_paths - actual_paths
    if unexpected:
        _add(problems, f"unmanifested shard(s), first={sorted(map(str, unexpected))[:3]}")
    if missing_paths:
        _add(problems, f"missing shard(s), first={sorted(map(str, missing_paths))[:3]}")
    states = list(shard_root.rglob("*.state.pkl")) if shard_root.is_dir() else []
    if states:
        _add(problems, f"unfinished anytime state shard(s), first={sorted(map(str, states))[:3]}")

    if require_manifest:
        manifest_path = run_dir / RUN_MANIFEST_FILE
        if not manifest_path.is_file():
            _add(problems, f"missing run manifest {manifest_path}")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("ensemble_config_hash") != digest:
                _add(problems, "run manifest config hash mismatch")
            if manifest.get("frozen_instance_manifest_sha256") != sha256_file(MANIFEST_PATH):
                _add(problems, "run manifest frozen-instance hash mismatch")
            if manifest.get("expected_rows") != len(expected_keys):
                _add(problems, "run manifest expected row count mismatch")
            if manifest.get("result_file", {}).get("sha256") != sha256_file(result_path):
                _add(problems, "run manifest result checksum mismatch")
    return problems


def _parse_ids(spec: str | None) -> list[int] | None:
    return None if spec is None else [int(value) for value in spec.split(",") if value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--instance-ids", default=None, help="Test-only comma-separated subset.")
    args = parser.parse_args()
    config = load_config()
    output_root = default_output_root(config)
    problems = validate(
        args.run_dir or output_root / "run",
        args.results_dir or output_root / "results",
        instance_ids=_parse_ids(args.instance_ids),
        require_manifest=args.instance_ids is None,
    )
    if problems:
        print("HETEROGENEOUS ENSEMBLE VALIDATION FAILED")
        for problem in problems[:100]:
            print(f"- {problem}")
        return 1
    print("HETEROGENEOUS ENSEMBLE VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
