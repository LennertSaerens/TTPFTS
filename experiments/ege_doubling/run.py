#!/usr/bin/env python3
"""Run every completed doubling stage of EGE-SH/EGE-SR with the unchanged sharded runner.

Each stage is one fresh fixed-budget EGE call at budget ``2**(s-1) * K``,
computed by ``momab_main.run_algorithm_sharded`` with an explicit stage-budget
grid. Seeding therefore follows the preregistered per-budget EGE contract
exactly, shards are atomic and resumable, and repetitions run in parallel.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

# Repetition-parallel workload: nested BLAS/OpenMP threads would only contend.
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

from momab_main import EGE_ALGORITHMS, ENVIRONMENTS, _construct_environment, run_algorithm_sharded
from runner.ege_allocation import expected_ege_draws
from runner.instance import compute_instance_hash
from runner.seeding import seed_rngs
from runner.shards import merge_shards

from .config import (
    DOUBLING_ALGORITHMS,
    assert_safe_output,
    config_hash,
    default_output_root,
    environment_spec,
    environment_specs,
    load_config,
    sha256_file,
    source_provenance,
    verify_protected_hashes,
    verify_source_hashes,
)
from .schedule import doubling_schedule


def stage_run_dir(output_root: Path) -> Path:
    return Path(output_root) / "stage_runs" / "run"


def stage_results_path(output_root: Path, environment: str) -> Path:
    return Path(output_root) / "stage_runs" / f"{environment}.csv"


def stage_manifest_path(output_root: Path, environment: str) -> Path:
    return Path(output_root) / "stage_runs" / "manifests" / f"{environment}.json"


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def schedule_payload(num_arms: int, budget: int) -> list[dict]:
    return [
        {
            "stage": stage.index,
            "budget": stage.budget,
            "start": stage.start,
            "end": stage.end,
            "actual_draws": {base: expected_ege_draws(base, stage.budget, num_arms)
                             for base in DOUBLING_ALGORITHMS.values()},
        }
        for stage in doubling_schedule(num_arms, budget)
    ]


def run_environment(
    output_root: Path,
    environment_name: str,
    config: dict,
    algorithms: list[str],
    num_runs: int,
    n_jobs: int,
    provenance: dict,
) -> int:
    """Run, merge, and manifest every stage of one environment."""
    spec = environment_spec(environment_name, config)
    schedule = doubling_schedule(spec["num_arms"], spec["budget"])
    stage_grid = [stage.budget for stage in schedule]
    dist, master_seed = config["dist"], int(config["master_seed"])
    environment = _construct_environment(ENVIRONMENTS[environment_name][0], dist)
    seed_rngs(master_seed, f"{environment_name}|{dist}", "env", 0)
    environment.reset()
    if environment.num_arms != spec["num_arms"] or compute_instance_hash(environment) != spec["instance_hash"]:
        raise ValueError(f"{environment_name}: live instance differs from the preregistered instance")

    run_dir = stage_run_dir(output_root)
    bases = [DOUBLING_ALGORITHMS[name] for name in algorithms]
    print(f"== {environment_name}: K={spec['num_arms']} B={spec['budget']} stages={stage_grid} ==", flush=True)
    for base in bases:
        run_algorithm_sharded(
            str(run_dir), environment, environment_name, dist, base, base, spec["budget"], stage_grid,
            num_runs, master_seed, n_jobs=n_jobs, write=True, is_ege=True, ege_func=EGE_ALGORITHMS[base],
        )

    results_path = stage_results_path(output_root, environment_name)
    rows = merge_shards(
        str(run_dir), environment_name, str(results_path),
        expected_grids={base: stage_grid for base in bases}, validate_metrics=True,
    )
    expected = num_runs * len(bases) * len(stage_grid)
    if rows != expected:
        raise ValueError(f"{environment_name}: merged {rows} stage rows, expected exactly {expected}")
    write_json_atomic(stage_manifest_path(output_root, environment_name), {
        "schema_version": "ege-doubling-stage-runs-v1",
        "doubling_config_hash": config_hash(config),
        "environment": spec,
        "algorithms": algorithms,
        "base_algorithms": bases,
        "num_runs": num_runs,
        "master_seed": master_seed,
        "dist": dist,
        "schedule": schedule_payload(spec["num_arms"], spec["budget"]),
        "expected_rows": expected,
        "result_file": {"path": str(results_path.resolve()), "sha256": sha256_file(results_path)},
        "n_jobs": n_jobs,
        **provenance,
    })
    print(f"  merged {rows} stage rows -> {results_path}", flush=True)
    return rows


def run(
    output_root: Path | None = None,
    config: dict | None = None,
    environments: list[str] | None = None,
    algorithms: list[str] | None = None,
    num_runs: int | None = None,
    n_jobs: int = 1,
    allow_dirty: bool = False,
) -> dict[str, int]:
    config = load_config() if config is None else config
    output_root = Path(default_output_root(config) if output_root is None else output_root)
    assert_safe_output(output_root, config)
    provenance = source_provenance(allow_dirty=allow_dirty)
    verify_source_hashes(config)
    verify_protected_hashes(config)
    environments = environments or [spec["name"] for spec in environment_specs(config)]
    algorithms = algorithms or list(DOUBLING_ALGORITHMS)
    unknown = sorted(set(algorithms) - set(DOUBLING_ALGORITHMS))
    if unknown:
        raise ValueError(f"unknown doubling algorithm(s): {unknown}")
    num_runs = int(config["num_runs"] if num_runs is None else num_runs)
    counts = {
        name: run_environment(output_root, name, config, algorithms, num_runs, n_jobs, provenance)
        for name in environments
    }
    verify_source_hashes(config)
    verify_protected_hashes(config)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--environments", default=None, help="Comma-separated subset (default: all eight).")
    parser.add_argument("--algorithms", default=None, help="Comma-separated subset of D_EGE_SH,D_EGE_SR.")
    parser.add_argument("--num-runs", type=int, default=None, help="Repetition count override for tests.")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--allow-dirty-source", action="store_true",
                        help="Testing only: permit uncommitted source (recorded in manifests).")
    args = parser.parse_args()
    split = lambda value: [item.strip() for item in value.split(",") if item.strip()] if value else None
    counts = run(
        output_root=Path(args.output_root) if args.output_root else None,
        environments=split(args.environments),
        algorithms=split(args.algorithms),
        num_runs=args.num_runs,
        n_jobs=args.n_jobs,
        allow_dirty=args.allow_dirty_source,
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
