#!/usr/bin/env python3
"""Merge complete UQ replay shards into one tidy CSV per environment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

from .config import config_hash, environment_config, load_config
from .replay import UQ_COLUMNS, shard_is_complete, shard_paths


def _atomic_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-uq-merge-", suffix=".csv")
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=UQ_COLUMNS)
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
    rows = []
    repetitions = list(range(config["num_runs"])) if repetitions is None else repetitions
    for repetition in repetitions:
        if not shard_is_complete(run_dir, environment, repetition, grid, digest):
            raise ValueError(f"{environment}/rep{repetition}: incomplete or stale UQ shard")
        csv_path, _state_path = shard_paths(run_dir, environment, repetition)
        with csv_path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    rows.sort(key=lambda row: (int(row["repetition"]), int(row["arm_pull"])))
    output = results_dir / f"{environment}.csv"
    _atomic_write(output, rows)
    return len(rows)


def write_manifest(
    run_dir: Path, results_dir: Path, config: dict, environments: list[str], repetitions: list[int] | None = None
) -> None:
    payload = {
        "schema_version": "hardness-budget-uq-results-v1",
        "uq_config_hash": config_hash(config),
        "num_runs": config["num_runs"] if repetitions is None else len(repetitions),
        "repetitions": list(range(config["num_runs"])) if repetitions is None else repetitions,
        "algorithm": config["algorithm"],
        "environments": {
            environment: environment_config(environment, config)
            for environment in environments
        },
        "results_dir": str(results_dir),
        "run_dir": str(run_dir),
    }
    target = run_dir / "uq_manifest.json"
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=".tmp-uq-manifest-", suffix=".json")
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--environments", default=None)
    parser.add_argument("--repetitions", default=None, help="Comma-separated repetition subset for a smoke run.")
    args = parser.parse_args()
    config = load_config()
    environments = [item["name"] for item in config["environments"]] if "environments" in config else []
    if not environments:
        from experiments.final_config_loader import load_final_config

        environments = [item["name"] for item in load_final_config()["environments"]]
    if args.environments:
        environments = args.environments.split(",")
    repetitions = None if args.repetitions is None else [int(value) for value in args.repetitions.split(",")]
    for environment in environments:
        rows = merge_environment(args.run_dir, args.results_dir, environment, config, repetitions)
        print(f"{environment}: {rows} rows")
    write_manifest(args.run_dir, args.results_dir, config, environments, repetitions)


if __name__ == "__main__":
    main()
