#!/usr/bin/env python3
"""Merge complete TTPFTS p-sweep shards and record source/result provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from .config import (
    config_hash,
    default_output_root,
    environment_config,
    expected_environments,
    load_config,
    p_tag,
)
from .run import SWEEP_COLUMNS, shard_is_complete, shard_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_provenance() -> dict[str, object]:
    """Return committed source provenance and reject uncommitted scientific code."""
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.splitlines()
    source_changes, ignored_bytecode_changes = [], []
    for entry in status:
        path = entry[3:]
        if path.endswith(".pyc") and "__pycache__/" in path:
            ignored_bytecode_changes.append(entry)
        else:
            source_changes.append(entry)
    if source_changes:
        raise RuntimeError(
            "refusing to merge p-sweep results from an uncommitted source tree: "
            + "; ".join(source_changes)
        )
    return {
        "source_commit": commit,
        "ignored_generated_bytecode_changes": ignored_bytecode_changes,
    }


def _atomic_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-p-sweep-merge-", suffix=".csv")
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SWEEP_COLUMNS)
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
    run_dir: Path,
    results_dir: Path,
    environment: str,
    p_values: list[float],
    repetitions: list[int],
    config: dict,
) -> int:
    """Strictly merge an exact environment/p/repetition rectangle."""
    details = environment_config(environment, config)
    digest = config_hash(config)
    rows: list[dict] = []
    for p in p_values:
        for repetition in repetitions:
            if not shard_is_complete(run_dir, environment, p, repetition, details["grid"], digest):
                raise ValueError(f"{environment}/{p_tag(p)}/rep{repetition:05d}: incomplete or stale shard")
            with shard_path(run_dir, environment, p, repetition).open(newline="") as handle:
                rows.extend(csv.DictReader(handle))
    rows.sort(key=lambda row: (float(row["rho"]), int(row["repetition"]), int(row["arm_pull"])))
    _atomic_write(results_dir / f"{environment}.csv", rows)
    return len(rows)


def write_manifest(
    run_dir: Path,
    results_dir: Path,
    environments: list[str],
    p_values: list[float],
    repetitions: list[int],
    config: dict,
) -> Path:
    """Write a checksummed manifest for the selected sweep result rectangle."""
    result_files = {
        environment: {
            "path": str((results_dir / f"{environment}.csv").resolve()),
            "sha256": _sha256_file(results_dir / f"{environment}.csv"),
        }
        for environment in environments
    }
    payload = {
        "schema_version": "hardness-normalized-ttpfts-p-sweep-results-v1",
        "sweep_config_hash": config_hash(config),
        "algorithm": config["algorithm"],
        "posterior": config["posterior"],
        "seeding": config["seeding"],
        "environments": {
            environment: environment_config(environment, config)
            for environment in environments
        },
        "p_values": [float(value) for value in p_values],
        "repetitions": repetitions,
        "expected_rows": len(environments) * len(p_values) * len(repetitions) * 21,
        "run_dir": str(run_dir.resolve()),
        "results_dir": str(results_dir.resolve()),
        "result_files": result_files,
        **source_provenance(),
    }
    target = run_dir / "p_sweep_manifest.json"
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=".tmp-p-sweep-manifest-", suffix=".json")
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
    return target


def _parse_values(spec: str | None, transform, fallback: list) -> list:
    return fallback if spec is None else [transform(value) for value in spec.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--environments", default=None)
    parser.add_argument("--p-values", default=None)
    parser.add_argument("--repetitions", default=None)
    args = parser.parse_args()

    config = load_config()
    output_root = default_output_root(config)
    run_dir = args.run_dir or output_root / "run"
    results_dir = args.results_dir or output_root / "results"
    environments = _parse_values(
        args.environments, str, [item["name"] for item in expected_environments()]
    )
    p_values = _parse_values(args.p_values, float, [float(value) for value in config["p_values"]])
    repetitions = _parse_values(args.repetitions, int, list(range(config["num_runs"])))
    for environment in environments:
        count = merge_environment(run_dir, results_dir, environment, p_values, repetitions, config)
        print(f"{environment}: {count} rows")
    print(f"Wrote {write_manifest(run_dir, results_dir, environments, p_values, repetitions, config)}")


if __name__ == "__main__":
    main()
