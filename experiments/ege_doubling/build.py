#!/usr/bin/env python3
"""Turn merged stage runs into exact doubling-trick trajectories and slim plot inputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from runner.ege_allocation import expected_ege_draws
from runner.schema import SCHEMA_COLUMNS, encode_int_set

from .config import (
    ANYTIME_ALGORITHMS,
    DOUBLING_ALGORITHMS,
    assert_safe_output,
    config_hash,
    default_output_root,
    environment_spec,
    environment_specs,
    load_config,
    sha256_file,
    source_results_dir,
    verify_protected_hashes,
    verify_source_hashes,
)
from .run import stage_manifest_path, stage_results_path, write_json_atomic
from .schedule import doubling_schedule, evaluation_points, stage_at


DOUBLING_COLUMNS = [
    *SCHEMA_COLUMNS,
    "base_algorithm",
    "stage",
    "stage_budget",
    "stage_start",
    "stage_end",
    "stage_actual_draws",
    "doubling_config_hash",
]
PLOT_COLUMNS = [
    "environment", "algorithm", "posterior_model", "repetition", "arm_pull", "max_budget",
    "bernoulli", "jaccard", "misidentification",
]


def doubling_path(output_root: Path, environment: str) -> Path:
    return Path(output_root) / "doubling" / f"{environment}.csv"


def plot_input_path(output_root: Path, environment: str) -> Path:
    return Path(output_root) / "plot_inputs" / f"{environment}.csv"


def build_manifest_path(output_root: Path, environment: str) -> Path:
    return Path(output_root) / "doubling" / "manifests" / f"{environment}.json"


def _atomic_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".csv")
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def read_stage_index(path: Path) -> dict[tuple[str, int, int], dict]:
    """Index merged stage rows by (base algorithm, repetition, stage budget), rejecting duplicates."""
    index: dict[tuple[str, int, int], dict] = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["algorithm"], int(row["repetition"]), int(row["arm_pull"]))
            if key in index:
                raise ValueError(f"duplicate stage row {key} in {path}")
            index[key] = row
    return index


def doubling_rows(spec: dict, stage_index: dict, algorithms: list[str], num_runs: int, digest: str) -> list[dict]:
    """Hold each completed stage's recommendation until the next stage endpoint."""
    schedule = doubling_schedule(spec["num_arms"], spec["budget"])
    points = evaluation_points(schedule, spec["grid"], spec["budget"])
    encoded_points = encode_int_set(points)
    rows = []
    for algorithm in algorithms:
        base = DOUBLING_ALGORITHMS[algorithm]
        for repetition in range(num_runs):
            for arm_pull in points:
                stage = stage_at(schedule, arm_pull)
                source = stage_index[(base, repetition, stage.budget)]
                row = {column: source[column] for column in SCHEMA_COLUMNS}
                row.update({
                    "algorithm": algorithm,
                    "arm_pull": arm_pull,
                    "max_budget": spec["budget"],
                    "evaluation_grid": encoded_points,
                    "base_algorithm": base,
                    "stage": stage.index,
                    "stage_budget": stage.budget,
                    "stage_start": stage.start,
                    "stage_end": stage.end,
                    "stage_actual_draws": expected_ege_draws(base, stage.budget, spec["num_arms"]),
                    "doubling_config_hash": digest,
                })
                rows.append(row)
    return rows


def anytime_plot_rows(config: dict, environment: str) -> pd.DataFrame:
    """Read the preregistered Uniform/APE/TTPFTS rows verbatim (as strings)."""
    frame = pd.read_csv(
        source_results_dir(config) / f"{environment}.csv",
        usecols=PLOT_COLUMNS, dtype=str, keep_default_na=False,
    )
    frame = frame[frame["algorithm"].isin(ANYTIME_ALGORITHMS)].loc[:, PLOT_COLUMNS]
    return frame.reset_index(drop=True)


def build_environment(
    output_root: Path, environment: str, config: dict, algorithms: list[str], num_runs: int
) -> dict[str, int]:
    spec = environment_spec(environment, config)
    digest = config_hash(config)
    stage_path = stage_results_path(output_root, environment)
    with stage_manifest_path(output_root, environment).open() as handle:
        stage_manifest = json.load(handle)
    if stage_manifest["doubling_config_hash"] != digest or stage_manifest["num_runs"] != num_runs:
        raise ValueError(f"{environment}: stage manifest does not match this build")
    if stage_manifest["result_file"]["sha256"] != sha256_file(stage_path):
        raise ValueError(f"{environment}: stage results changed after their manifest was written")
    if not set(algorithms) <= set(stage_manifest["algorithms"]):
        raise ValueError(f"{environment}: stage runs do not cover {algorithms}")

    rows = doubling_rows(spec, read_stage_index(stage_path), algorithms, num_runs, digest)
    output = doubling_path(output_root, environment)
    _atomic_csv(output, DOUBLING_COLUMNS, rows)

    anytime = anytime_plot_rows(config, environment)
    doubling = pd.DataFrame([{column: str(row[column]) for column in PLOT_COLUMNS} for row in rows])
    plot_frame = pd.concat([anytime, doubling], ignore_index=True)
    plot_path = plot_input_path(output_root, environment)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=plot_path.parent, prefix=".tmp-", suffix=".csv")
    os.close(descriptor)
    try:
        plot_frame.to_csv(temporary, index=False)
        os.replace(temporary, plot_path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise

    schedule = doubling_schedule(spec["num_arms"], spec["budget"])
    write_json_atomic(build_manifest_path(output_root, environment), {
        "schema_version": "ege-doubling-trajectories-v1",
        "doubling_config_hash": digest,
        "environment": spec,
        "algorithms": algorithms,
        "num_runs": num_runs,
        "evaluation_points": evaluation_points(schedule, spec["grid"], spec["budget"]),
        "stage_result_sha256": sha256_file(stage_path),
        "doubling_file": {"path": str(output.resolve()), "sha256": sha256_file(output), "rows": len(rows)},
        "plot_input_file": {
            "path": str(plot_path.resolve()), "sha256": sha256_file(plot_path),
            "anytime_rows": len(anytime), "doubling_rows": len(doubling),
        },
        "source_commit": stage_manifest["source_commit"],
        "source_dirty": stage_manifest["source_dirty"],
    })
    return {"doubling_rows": len(rows), "anytime_rows": len(anytime)}


def build(
    output_root: Path | None = None,
    config: dict | None = None,
    environments: list[str] | None = None,
    algorithms: list[str] | None = None,
    num_runs: int | None = None,
) -> dict[str, dict[str, int]]:
    config = load_config() if config is None else config
    output_root = Path(default_output_root(config) if output_root is None else output_root)
    assert_safe_output(output_root, config)
    verify_source_hashes(config)
    verify_protected_hashes(config)
    environments = environments or [spec["name"] for spec in environment_specs(config)]
    algorithms = algorithms or list(DOUBLING_ALGORITHMS)
    num_runs = int(config["num_runs"] if num_runs is None else num_runs)
    counts = {name: build_environment(output_root, name, config, algorithms, num_runs) for name in environments}
    verify_source_hashes(config)
    verify_protected_hashes(config)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--environments", default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    args = parser.parse_args()
    environments = [item.strip() for item in args.environments.split(",")] if args.environments else None
    counts = build(
        output_root=Path(args.output_root) if args.output_root else None,
        environments=environments,
        num_runs=args.num_runs,
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
