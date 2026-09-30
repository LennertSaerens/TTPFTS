#!/usr/bin/env python3
"""Merge complete heterogeneous-ensemble shards with committed-source provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .config import config_hash, default_output_root, sha256_file, load_config
from .instances import MANIFEST_PATH, FrozenInstance, load_frozen_instances
from .run import (
    ALGORITHMS,
    ANYTIME_ALGORITHMS,
    EGE_ALGORITHMS,
    RESULT_COLUMNS,
    anytime_is_complete,
    anytime_paths,
    ege_is_complete,
    ege_path,
)


RESULT_FILE = "ensemble_results.csv"
RUN_MANIFEST_FILE = "ensemble_manifest.json"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _atomic_write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-ensemble-merge-", suffix=".csv")
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def source_provenance() -> dict[str, Any]:
    """Return the source revision and reject modified scientific source files."""
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
            "refusing to merge ensemble results from an uncommitted source tree: "
            + "; ".join(source_changes)
        )
    return {
        "source_commit": commit,
        "ignored_generated_bytecode_changes": ignored_bytecode_changes,
    }


def merge(
    run_dir: Path,
    results_dir: Path,
    instances: list[FrozenInstance],
    config: dict,
) -> Path:
    """Strictly merge a complete rectangular instance/algorithm/checkpoint cohort."""
    digest = config_hash(config)
    rows: list[dict[str, str]] = []
    for instance in instances:
        for algorithm in ALGORITHMS:
            if algorithm in ANYTIME_ALGORITHMS:
                if not anytime_is_complete(run_dir, instance, algorithm, digest):
                    raise ValueError(f"{instance.name}/{algorithm}: incomplete anytime trajectory")
                csv_path, _state_path = anytime_paths(run_dir, instance, algorithm)
                rows.extend(_read_rows(csv_path))
            elif algorithm in EGE_ALGORITHMS:
                for budget in instance.grid:
                    if not ege_is_complete(run_dir, instance, algorithm, budget, digest):
                        raise ValueError(f"{instance.name}/{algorithm}/budget={budget}: incomplete EGE task")
                    rows.extend(_read_rows(ege_path(run_dir, instance, algorithm, budget)))
            else:
                raise ValueError(f"unknown configured algorithm {algorithm}")
    expected_count = len(instances) * len(ALGORITHMS) * 21
    if len(rows) != expected_count:
        raise ValueError(f"merged row count {len(rows)}, expected {expected_count}")
    rows.sort(key=lambda row: (int(row["proposal_id"]), row["algorithm"], int(row["grid_index"])))
    output = results_dir / RESULT_FILE
    _atomic_write_csv(output, rows)
    return output


def write_manifest(
    run_dir: Path,
    results_dir: Path,
    result_path: Path,
    instances: list[FrozenInstance],
    config: dict,
    *,
    require_clean_source: bool,
) -> Path:
    """Record all hashes that connect generated rows to their frozen cohort/source."""
    provenance = source_provenance() if require_clean_source else {"source_commit": "test_uncommitted"}
    payload = {
        "schema_version": "heterogeneous-ensemble-results-v1",
        "ensemble_config_hash": config_hash(config),
        "algorithms": list(ALGORITHMS),
        "instance_count": len(instances),
        "expected_rows": len(instances) * len(ALGORITHMS) * 21,
        "run_dir": str(run_dir.resolve()),
        "results_dir": str(results_dir.resolve()),
        "frozen_instance_manifest_sha256": sha256_file(MANIFEST_PATH),
        "result_file": {
            "path": str(result_path.resolve()),
            "sha256": sha256_file(result_path),
        },
        **provenance,
    }
    target = run_dir / RUN_MANIFEST_FILE
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=".tmp-ensemble-manifest-", suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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


def _parse_ids(spec: str | None) -> list[int] | None:
    return None if spec is None else [int(value) for value in spec.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--instance-ids", default=None, help="Test-only comma-separated subset.")
    args = parser.parse_args()
    config = load_config()
    output_root = default_output_root(config)
    instances = load_frozen_instances(_parse_ids(args.instance_ids))
    run_dir = args.run_dir or output_root / "run"
    results_dir = args.results_dir or output_root / "results"
    result = merge(run_dir, results_dir, instances, config)
    manifest = write_manifest(
        run_dir,
        results_dir,
        result,
        instances,
        config,
        require_clean_source=args.instance_ids is None,
    )
    print(f"Merged {len(instances) * len(ALGORITHMS) * 21} rows -> {result}")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
