#!/usr/bin/env python3
"""Run APE-UV trajectories on the preregistered benchmark with the shared anytime runner.

Each repetition uses ``momab_main._run_repetition_anytime`` with the
preregistered environment seed and its own ``APE_UV`` seed namespace, the
shared 101-point grid, and exactly ``B`` pulls. Workers write one atomic shard
per repetition as soon as it finishes, so an interruption loses at most the
repetitions in flight, and a rerun skips every complete shard.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

from joblib import Parallel, delayed

from momab_main import ENVIRONMENTS, _construct_environment, _make_bandit, _run_repetition_anytime
from runner.shards import merge_shards, shard_is_complete, shard_path, write_shard_atomic

from .config import (
    NEW_ANYTIME_ALGORITHM,
    assert_safe_output,
    config_hash,
    default_output_root,
    environment_spec,
    environment_specs,
    load_config,
    sha256_file,
    source_provenance,
    verify_protected,
    verify_sources,
)
from .fileio import atomic_json


ALGORITHM = NEW_ANYTIME_ALGORITHM


def run_dir(output_root: Path) -> Path:
    return Path(output_root) / "ape_uv" / "run"


def results_path(output_root: Path, environment: str) -> Path:
    return Path(output_root) / "ape_uv" / f"{environment}.csv"


def manifest_path(output_root: Path, environment: str) -> Path:
    return Path(output_root) / "ape_uv" / "manifests" / f"{environment}.json"


def run_task(output_root: str, environment_name: str, repetition: int, config: dict) -> dict:
    """Compute and atomically persist one APE-UV repetition (skipped when already complete)."""
    spec = environment_spec(environment_name, config)
    path = shard_path(str(run_dir(Path(output_root))), environment_name, ALGORITHM, repetition)
    if shard_is_complete(path, spec["grid"]):
        return {"environment": environment_name, "repetition": repetition, "skipped": True, "seconds": 0.0}
    environment = _construct_environment(ENVIRONMENTS[environment_name][0], config["dist"])
    bandit = _make_bandit(ALGORITHM, environment, p=0.5)
    start = time.perf_counter()
    rows = _run_repetition_anytime(
        environment, environment_name, config["dist"], ALGORITHM, bandit,
        spec["budget"], spec["grid"], int(config["master_seed"]), repetition,
    )
    seconds = time.perf_counter() - start
    if any(row["instance_hash"] != spec["instance_hash"] for row in rows):
        raise ValueError(f"{environment_name}: live instance differs from the preregistered instance")
    write_shard_atomic(path, rows)
    return {"environment": environment_name, "repetition": repetition, "skipped": False, "seconds": seconds}


def ordered_tasks(names: list[str], repetitions: list[int], config: dict) -> list[tuple[str, int]]:
    """Longest-first: per-pull cost grows with K, total cost with B."""
    cost = {spec["name"]: spec["budget"] * (1.0 + spec["num_arms"] / 100.0) for spec in environment_specs(config)}
    return [(name, repetition) for name in sorted(names, key=lambda n: -cost[n]) for repetition in repetitions]


def merge_environment(output_root: Path, environment_name: str, config: dict, repetitions: list[int],
                      provenance: dict, timings: list[dict]) -> int:
    spec = environment_spec(environment_name, config)
    output = results_path(output_root, environment_name)
    rows = merge_shards(str(run_dir(output_root)), environment_name, str(output),
                        expected_grids={ALGORITHM: spec["grid"]}, validate_metrics=True)
    expected = len(repetitions) * len(spec["grid"])
    if rows != expected:
        raise ValueError(f"{environment_name}: merged {rows} APE-UV rows, expected exactly {expected}")
    seconds = [item["seconds"] for item in timings if item["environment"] == environment_name and not item["skipped"]]
    atomic_json(manifest_path(output_root, environment_name), {
        "schema_version": "information-matched-ape-uv-v1",
        "config_hash": config_hash(config),
        "algorithm": ALGORITHM,
        "ape_uv": config["ape_uv"],
        "environment": spec,
        "repetitions": repetitions,
        "expected_rows": expected,
        "result_file": {"path": str(output.resolve()), "sha256": sha256_file(output)},
        "computed_repetition_seconds": seconds,
        **provenance,
    })
    return rows


def run(output_root: Path | None = None, config: dict | None = None, environments: list[str] | None = None,
        repetitions: list[int] | None = None, n_jobs: int = 1, allow_dirty: bool = False) -> dict[str, int]:
    config = load_config() if config is None else config
    output_root = Path(default_output_root(config) if output_root is None else output_root)
    assert_safe_output(output_root, config)
    provenance = source_provenance(allow_dirty=allow_dirty)
    verify_sources(config)
    verify_protected(config)
    names = environments or [spec["name"] for spec in environment_specs(config)]
    repetitions = list(range(int(config["num_runs"]))) if repetitions is None else list(repetitions)
    tasks = ordered_tasks(names, repetitions, config)
    start = time.perf_counter()
    if n_jobs == 1:
        timings = [run_task(str(output_root), name, repetition, config) for name, repetition in tasks]
    else:
        timings = Parallel(n_jobs=n_jobs, batch_size=1)(
            delayed(run_task)(str(output_root), name, repetition, config) for name, repetition in tasks
        )
    print(f"APE-UV tasks finished in {time.perf_counter() - start:.1f} s", flush=True)
    counts = {name: merge_environment(output_root, name, config, repetitions, provenance, timings) for name in names}
    verify_sources(config)
    verify_protected(config)
    return counts


def runtime_probe(config: dict | None = None, steady_pulls: int = 20_000) -> dict:
    """Measure steady-state APE-UV cost per environment and project total core time."""
    from runner.seeding import seed_rngs

    config = load_config() if config is None else config
    report = {}
    for spec in environment_specs(config):
        environment = _construct_environment(ENVIRONMENTS[spec["name"]][0], config["dist"])
        bandit = _make_bandit(ALGORITHM, environment, p=0.5)
        seed_rngs(0, "probe", spec["name"])
        bandit.reset(None)
        warm = 4 * spec["num_arms"] + 500
        pulls = min(steady_pulls, spec["budget"] - warm)
        for _ in range(warm):
            arm = bandit.choose_arm()
            bandit.learn(arm, environment.pull_arm(arm))
        start = time.perf_counter()
        for _ in range(pulls):
            arm = bandit.choose_arm()
            bandit.learn(arm, environment.pull_arm(arm))
        per_pull = (time.perf_counter() - start) / pulls
        report[spec["name"]] = {
            "microseconds_per_pull": per_pull * 1e6,
            "projected_core_seconds": per_pull * spec["budget"] * int(config["num_runs"]),
        }
    report["total_projected_core_seconds"] = sum(item["projected_core_seconds"] for item in report.values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--environments", default=None)
    parser.add_argument("--repetitions", default=None, help="Comma-separated subset (testing).")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument("--probe", action="store_true", help="Only measure steady-state runtime.")
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(runtime_probe(), indent=2))
        return
    split = lambda value, cast=str: [cast(v.strip()) for v in value.split(",") if v.strip()] if value else None
    counts = run(
        output_root=Path(args.output_root) if args.output_root else None,
        environments=split(args.environments),
        repetitions=split(args.repetitions, int),
        n_jobs=args.n_jobs,
        allow_dirty=args.allow_dirty_source,
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
