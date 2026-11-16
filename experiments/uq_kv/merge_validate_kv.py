#!/usr/bin/env python3
"""Merge and validate completed TTPFTS_KNOWN_VAR UQ replay shards."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from experiments.final_config_loader import load_final_config
from experiments.uq.metrics import (
    uncertainty_from_moments,
    uq_directional_bc_from_moments,
    uq_dominance_risk_from_moments,
)

from .config import config_hash, environment_config, load_config
from .replay_kv import UQ_KV_COLUMNS, shard_is_complete, shard_paths


IDENTITY_COLUMNS = (
    "environment", "instance_hash", "algorithm", "posterior_model", "repetition",
    "arm_pull", "max_budget", "evaluation_grid", "rho", "master_seed", "bernoulli",
    "jaccard", "misidentification", "recommendation", "true_pareto",
)


def _source_provenance(allow_dirty: bool = False) -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    git_dir = root / ".git"
    if git_dir.exists():
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True).stdout.splitlines()
        dirty = [e for e in status if not (e[3:].endswith(".pyc") and "__pycache__/" in e[3:])]
        if dirty and not allow_dirty:
            raise RuntimeError("uncommitted edits present: " + "; ".join(dirty))
        return {"source_commit": commit, "dirty": bool(dirty)}
    prov = root / "SOURCE_PROVENANCE.txt"
    if prov.is_file():
        return {"source_provenance_file": prov.read_text(encoding="utf-8").strip()}
    return {}


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-uqkv-merge-", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=UQ_KV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def merge_environment(
    run_dir: Path, results_dir: Path, environment: str, config: dict, repetitions: list[int] | None = None
) -> int:
    details = environment_config(environment, config)
    grid = details["grid"]
    digest = config_hash(config)
    repetitions = list(range(config["num_runs"])) if repetitions is None else repetitions
    rows = []
    for rep in repetitions:
        if not shard_is_complete(run_dir, environment, rep, grid, digest):
            raise ValueError(f"{environment}/rep{rep}: incomplete or stale UQ KV shard")
        csv_path, _ = shard_paths(run_dir, environment, rep)
        with csv_path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    rows.sort(key=lambda row: (int(row["repetition"]), int(row["arm_pull"])))
    output = results_dir / f"{environment}.csv"
    _atomic_csv(output, rows)
    return len(rows)


def _matches_float(actual: str, expected: float, tol: float = 1e-10) -> bool:
    try:
        return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tol)
    except ValueError:
        return False


def _validate_one_rep(
    task: tuple[Path, str, int, list[int], str, int, dict[int, dict[str, str]]]
) -> list[str]:
    run_dir, env_name, rep, grid, digest, num_objectives, final_rep_rows = task
    label = f"{env_name}/rep{rep:05d}"
    if not shard_is_complete(run_dir, env_name, rep, grid, digest):
        return [f"{label}: incomplete/stale shard"]

    problems: list[str] = []
    csv_path, state_path = shard_paths(run_dir, env_name, rep)
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with np.load(state_path, allow_pickle=False) as state:
        counts, means, stds = state["counts"], state["means"], state["stds"]
    for idx, (row, pull) in enumerate(zip(rows, grid)):
        rlbl = f"{label}/t={pull}"
        if int(row["arm_pull"]) != pull or row["uq_config_hash"] != digest:
            problems.append(f"{rlbl}: grid/hash mismatch")
            continue
        if int(counts[idx].sum()) != pull * num_objectives:
            problems.append(f"{rlbl}: posterior count sum mismatch")
        final = final_rep_rows[pull]
        for col in IDENTITY_COLUMNS:
            if row[col] != final[col]:
                problems.append(f"{rlbl}: final identity mismatch in {col}")
                break
        if pull > 0 and row["uq_available"] == "1":
            v1_exp = uq_directional_bc_from_moments(means[idx], stds[idx])
            v2_exp = uq_dominance_risk_from_moments(means[idx], stds[idx])
            if not _matches_float(row["uq_v1"], v1_exp):
                problems.append(f"{rlbl}: uq_v1 recompute mismatch")
            if not _matches_float(row["uq_v2"], v2_exp):
                problems.append(f"{rlbl}: uq_v2 recompute mismatch")
            eq10_res = uncertainty_from_moments(means[idx], stds[idx])
            eq10_exp = float(eq10_res.bhattacharyya) if eq10_res.available else 0.0
            if not _matches_float(row["bhattacharyya"], eq10_exp):
                problems.append(f"{rlbl}: eq10 recompute mismatch")
    return problems


def validate(
    run_dir: Path,
    final_results_dir: Path,
    environments: list[str] | None = None,
    repetitions: list[int] | None = None,
    n_jobs: int = 1,
) -> list[str]:
    config = load_config()
    digest = config_hash(config)
    if environments is None:
        environments = [item["name"] for item in load_final_config()["environments"]]
    repetitions = list(range(config["num_runs"])) if repetitions is None else repetitions
    problems: list[str] = []
    tasks: list[tuple[Path, str, int, list[int], str, int, dict[int, dict[str, str]]]] = []

    for env_name in environments:
        details = environment_config(env_name, config)
        grid = details["grid"]
        num_objectives = int(details["num_objectives"])
        final_path = final_results_dir / f"{env_name}.csv"
        if not final_path.is_file():
            problems.append(f"{env_name}: missing final results CSV {final_path}")
            continue
        with final_path.open(newline="") as handle:
            final_rows = {
                (int(row["repetition"]), int(row["arm_pull"])): {col: row[col] for col in IDENTITY_COLUMNS}
                for row in csv.DictReader(handle)
                if row["algorithm"] == "TTPFTS_KNOWN_VAR"
            }
        expected_keys = {(rep, pull) for rep in repetitions for pull in grid}
        if not expected_keys.issubset(final_rows):
            problems.append(f"{env_name}: final TTPFTS_KNOWN_VAR lacks requested coverage")
            continue

        for rep in repetitions:
            final_rep_rows = {pull: final_rows[(rep, pull)] for pull in grid}
            tasks.append((run_dir, env_name, int(rep), grid, digest, num_objectives, final_rep_rows))

    if n_jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            for rep_problems in pool.map(_validate_one_rep, tasks):
                problems.extend(rep_problems)
    else:
        for task in tasks:
            problems.extend(_validate_one_rep(task))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--final-results-dir", type=Path, required=True)
    parser.add_argument("--environments", default=None)
    parser.add_argument("--repetitions", default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    config = load_config()
    environments = (
        args.environments.split(",")
        if args.environments
        else [item["name"] for item in load_final_config()["environments"]]
    )
    repetitions = None if args.repetitions is None else [int(v) for v in args.repetitions.split(",")]
    for env in environments:
        n_rows = merge_environment(args.run_dir, args.results_dir, env, config, repetitions)
        print(f"Merged {env}: {n_rows} rows", flush=True)
    prov = _source_provenance(allow_dirty=args.allow_dirty)
    manifest = {
        "schema_version": "known-var-uq-results-v1",
        "uq_config_hash": config_hash(config),
        "environments": environments,
        **prov,
    }
    (args.run_dir / "uq_kv_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    problems = validate(args.run_dir, args.final_results_dir, environments, repetitions, n_jobs=args.n_jobs)
    if problems:
        print("UQ KV VALIDATION FAILED:")
        for p in problems[:100]:
            print(f" - {p}")
        return 1
    print("UQ KV VALIDATION PASSED (100% identity + metric integrity verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
