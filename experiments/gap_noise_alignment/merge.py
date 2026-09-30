#!/usr/bin/env python3
"""Strictly merge complete gap-noise metric shards and EGE trace sidecars."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .config import ALGORITHMS, config_hash, default_output_root, load_config, sha256_file
from .instances import MANIFEST_PATH, FrozenCondition, load_conditions
from .run import (
    ANYTIME_ALGORITHMS,
    EGE_ALGORITHMS,
    RESULT_COLUMNS,
    TRACE_COLUMNS,
    anytime_is_complete,
    anytime_paths,
    ege_is_complete,
    ege_paths,
)


RESULT_FILE = "gap_noise_results.csv"
TRACE_FILE = "ege_classification_trace.csv"
RUN_MANIFEST_FILE = "gap_noise_manifest.json"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _atomic_write(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-gap-noise-merge-", suffix=".csv")
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
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
    """Bind production outputs to a clean committed source tree."""
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.splitlines()
    scientific_changes = [
        item for item in status
        if not (item[3:].endswith(".pyc") and "__pycache__/" in item[3:])
    ]
    if scientific_changes:
        raise RuntimeError(
            "refusing to merge from an uncommitted source tree: " + "; ".join(scientific_changes)
        )
    return {"source_commit": commit}


def merge(
    run_dir: Path,
    results_dir: Path,
    conditions: Iterable[FrozenCondition],
    repetitions: Iterable[int],
    config: dict[str, Any],
) -> tuple[Path, Path]:
    """Merge exactly the requested rectangular condition/repetition coverage."""
    digest = config_hash(config)
    conditions, repetitions = list(conditions), list(repetitions)
    metric_rows: list[dict[str, str]] = []
    trace_rows: list[dict[str, str]] = []
    for item in conditions:
        for repetition in repetitions:
            for algorithm in ALGORITHMS:
                if algorithm in ANYTIME_ALGORITHMS:
                    if not anytime_is_complete(run_dir, item, repetition, algorithm, digest):
                        raise ValueError(f"{item.condition}/rep={repetition}/{algorithm}: incomplete anytime shard")
                    metric_rows.extend(_read_rows(anytime_paths(run_dir, item, repetition, algorithm)[0]))
                elif algorithm in EGE_ALGORITHMS:
                    for budget in item.grid:
                        if not ege_is_complete(run_dir, item, repetition, algorithm, budget, digest):
                            raise ValueError(f"{item.condition}/rep={repetition}/{algorithm}/budget={budget}: incomplete EGE shard")
                        result_path, trace_path = ege_paths(run_dir, item, repetition, algorithm, budget)
                        metric_rows.extend(_read_rows(result_path))
                        trace_rows.extend(_read_rows(trace_path))
                else:
                    raise ValueError(f"unknown algorithm {algorithm}")
    expected_metrics = len(conditions) * len(repetitions) * len(ALGORITHMS) * 21
    expected_traces = len(conditions) * len(repetitions) * len(EGE_ALGORITHMS) * 21 * 50
    if len(metric_rows) != expected_metrics or len(trace_rows) != expected_traces:
        raise ValueError(
            f"merged metrics/traces={len(metric_rows)}/{len(trace_rows)}, "
            f"expected {expected_metrics}/{expected_traces}"
        )
    metric_rows.sort(key=lambda row: (row["condition"], int(row["repetition"]), row["algorithm"], int(row["grid_index"])))
    trace_rows.sort(key=lambda row: (
        row["condition"], int(row["repetition"]), row["algorithm"], int(row["grid_index"]), int(row["arm"])
    ))
    result_path, trace_path = results_dir / RESULT_FILE, results_dir / TRACE_FILE
    _atomic_write(result_path, RESULT_COLUMNS, metric_rows)
    _atomic_write(trace_path, TRACE_COLUMNS, trace_rows)
    return result_path, trace_path


def write_manifest(
    run_dir: Path,
    results_dir: Path,
    result_path: Path,
    trace_path: Path,
    conditions: list[FrozenCondition],
    repetitions: list[int],
    config: dict[str, Any],
    *,
    require_clean_source: bool,
) -> Path:
    """Write checksum-bound provenance for both merged output tables."""
    payload = {
        "schema_version": "gap-noise-alignment-results-v1",
        "config_hash": config_hash(config),
        "conditions": [item.condition for item in conditions],
        "repetitions": repetitions,
        "algorithms": list(ALGORITHMS),
        "expected_metric_rows": len(conditions) * len(repetitions) * len(ALGORITHMS) * 21,
        "expected_trace_rows": len(conditions) * len(repetitions) * len(EGE_ALGORITHMS) * 21 * 50,
        "frozen_condition_manifest_sha256": sha256_file(MANIFEST_PATH),
        "result_file": {"path": str(result_path.resolve()), "sha256": sha256_file(result_path)},
        "trace_file": {"path": str(trace_path.resolve()), "sha256": sha256_file(trace_path)},
        "run_dir": str(run_dir.resolve()),
        "results_dir": str(results_dir.resolve()),
        **(source_provenance() if require_clean_source else {"source_commit": "test_uncommitted"}),
    }
    target = run_dir / RUN_MANIFEST_FILE
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=".tmp-gap-noise-manifest-", suffix=".json")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()
    config = load_config()
    root = default_output_root(config)
    conditions, repetitions = load_conditions(), list(range(config["repetitions"]))
    result, trace = merge(args.run_dir or root / "run", args.results_dir or root / "results", conditions, repetitions, config)
    manifest = write_manifest(
        args.run_dir or root / "run", args.results_dir or root / "results", result, trace,
        conditions, repetitions, config, require_clean_source=True,
    )
    print(f"Merged {len(conditions) * len(repetitions) * len(ALGORITHMS) * 21} metric rows -> {result}")
    print(f"Merged {len(conditions) * len(repetitions) * len(EGE_ALGORITHMS) * 21 * 50} trace rows -> {trace}")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
