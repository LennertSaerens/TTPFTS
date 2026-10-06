#!/usr/bin/env python3
"""Empirical-mean doubling-trick EGE (Jun & Nowak's DSAR analog).

For every environment, base algorithm and repetition, stages ``s = 1..S+1``
of the doubling schedule are re-executed with the unchanged native EGE code
and the naive stages' per-budget seed contract. A passive recorder captures
every ``environment.sample`` call (each call is exactly one EGE round block)
without changing random-number consumption, so every completed stage
reproduces the stored naive stage recommendation.

Recorded pulls are placed on the doubling clock: stage ``s`` draws at
``e_{s-1}+1 .. e_{s-1}+a_s`` and is idle until ``e_s``; the incomplete stage
``S+1`` fills ``(e_S, B]``. Within each round block pulls are interleaved
round-robin over the round's active arms (the k-th visit to arm ``a`` takes
the k-th reward drawn for ``a``), so within-round recommendations are not
biased towards low arm indices. At every evaluation point the recommendation
is the Pareto set of the empirical means of every sample drawn so far, with
unsampled arms at zero exactly as in Uniform/APE. Native block order is
computed alongside as a sensitivity analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
from joblib import Parallel, delayed

from bandits.APE import _pareto_optimal_arms_mask
from experiments.ege_doubling.schedule import doubling_schedule
from momab_main import EGE_ALGORITHMS, ENVIRONMENTS, _construct_environment
from runner.ege_allocation import expected_ege_draws
from runner.instance import compute_instance_hash
from runner.schema import SCHEMA_COLUMNS, build_row, encode_int_set
from runner.seeding import seed_rngs

from .config import (
    EMPIRICAL_ALGORITHMS,
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
from .fileio import atomic_csv, atomic_json, read_csv


TRAJECTORY_COLUMNS = [*SCHEMA_COLUMNS, "base_algorithm", "active_stage", "samples_drawn", "unsampled_arms", "config_hash"]
STAGE_COLUMNS = [
    "environment", "algorithm", "base_algorithm", "repetition", "stage", "stage_budget", "stage_start",
    "stage_end", "completed", "actual_draws", "used_draws", "recommendation", "config_hash",
]
ORDERS = {"round_robin": "", "block": "_BLOCK"}


class SampleRecorder:
    """Pass-through environment proxy that records every sampled round block."""

    def __init__(self, environment) -> None:
        self._environment = environment
        self.blocks: list[tuple[np.ndarray, np.ndarray]] = []

    def sample(self, arms):
        arms = np.asarray(arms)
        rewards = self._environment.sample(arms)
        self.blocks.append((arms.copy(), np.array(rewards, dtype=float, copy=True)))
        return rewards


def round_robin(arms: np.ndarray, rewards: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interleave one ``np.repeat(active, n)`` round block into round-robin order."""
    if arms.size == 0:
        return arms, rewards
    _, first = np.unique(arms, return_index=True)
    active = arms[np.sort(first)]
    per_arm = arms.size // active.size
    if not np.array_equal(arms, np.repeat(active, per_arm)):
        raise ValueError("EGE round block is not an equal-allocation np.repeat block")
    order = np.arange(arms.size).reshape(active.size, per_arm).T.ravel()
    return arms[order], rewards[order]


def stage_plan(num_arms: int, budget: int) -> list[dict]:
    """Stages 1..S (complete) and the incomplete stage S+1 on the doubling clock."""
    schedule = doubling_schedule(num_arms, budget)
    plan = [{"stage": stage.index, "budget": stage.budget, "start": stage.start - stage.budget,
             "end": stage.start, "completed": True} for stage in schedule]
    last = schedule[-1]
    plan.append({"stage": last.index + 1, "budget": 2 * last.budget, "start": last.start,
                 "end": int(budget), "completed": False})
    return plan


def evaluation_points(spec: dict) -> list[int]:
    endpoints = {stage.start for stage in doubling_schedule(spec["num_arms"], spec["budget"])}
    return sorted(set(spec["grid"]) | endpoints)


def active_stage(plan: list[dict], arm_pull: int) -> int:
    for item in plan:
        if item["start"] < arm_pull <= item["end"]:
            return item["stage"]
    return 0


def empirical_trajectory(times: np.ndarray, arms: np.ndarray, rewards: np.ndarray, points: list[int],
                         num_arms: int) -> list[tuple[int, np.ndarray, int, int]]:
    """``(t, recommendation, samples_drawn, unsampled_arms)`` at every evaluation point.

    Sums accumulate pull by pull in time order (``np.add.at`` is unbuffered),
    matching a sequential reference implementation bit for bit.
    """
    sums = np.zeros((num_arms, rewards.shape[1] if rewards.ndim == 2 else 0), dtype=float)
    counts = np.zeros(num_arms, dtype=np.int64)
    position, out = 0, []
    for arm_pull in points:
        upto = int(np.searchsorted(times, arm_pull, side="right"))
        if upto > position:
            np.add.at(sums, arms[position:upto], rewards[position:upto])
            np.add.at(counts, arms[position:upto], 1)
            position = upto
        means = np.zeros_like(sums)
        sampled = counts > 0
        means[sampled] = sums[sampled] / counts[sampled, None]
        out.append((arm_pull, np.flatnonzero(_pareto_optimal_arms_mask(means)), position, int((~sampled).sum())))
    return out


def task_dir(output_root: Path, environment: str, algorithm: str, repetition: int) -> Path:
    return Path(output_root) / "empirical" / "run" / "shards" / environment / algorithm / f"rep{repetition:05d}"


def _task_complete(directory: Path, digest: str) -> bool:
    marker = directory / "complete.json"
    if not marker.is_file():
        return False
    try:
        with marker.open() as handle:
            payload = json.load(handle)
        return payload["config_hash"] == digest and all(
            sha256_file(directory / name) == value for name, value in payload["files"].items()
        )
    except (OSError, KeyError, ValueError):
        return False


def run_task(output_root: str, environment_name: str, algorithm: str, repetition: int, config: dict) -> dict:
    """Re-execute every stage of one trajectory and persist its rows atomically."""
    digest = config_hash(config)
    directory = task_dir(Path(output_root), environment_name, algorithm, repetition)
    if _task_complete(directory, digest):
        return {"environment": environment_name, "algorithm": algorithm, "repetition": repetition, "skipped": True}
    start_time = time.perf_counter()
    spec = environment_spec(environment_name, config)
    K, budget = spec["num_arms"], spec["budget"]
    base = EMPIRICAL_ALGORITHMS[algorithm]
    dist, master_seed = config["dist"], int(config["master_seed"])
    env_tag = f"{environment_name}|{dist}"
    environment = _construct_environment(ENVIRONMENTS[environment_name][0], dist)
    seed_rngs(master_seed, env_tag, "env", repetition)
    environment.reset()
    instance_hash = compute_instance_hash(environment)
    if instance_hash != spec["instance_hash"]:
        raise ValueError(f"{environment_name}: live instance differs from the preregistered instance")
    true_pareto = list(environment.get_top_arms())

    plan = stage_plan(K, budget)
    pieces = {order: ([], [], []) for order in ORDERS}
    stage_rows = []
    for item in plan:
        seed_rngs(master_seed, env_tag, base, repetition, "budget", item["budget"])
        recorder = SampleRecorder(environment)
        recommendation = list(EGE_ALGORITHMS[base](
            T=item["budget"], K=K, D=environment.num_objectives, environment=recorder,
        ))
        actual = sum(block[0].size for block in recorder.blocks)
        if actual != expected_ege_draws(base, item["budget"], K):
            raise ValueError(f"{environment_name}/{base}/stage {item['stage']}: unexpected draw count {actual}")
        used = min(actual, item["end"] - item["start"])
        stamps = item["start"] + 1 + np.arange(used)
        for order in ORDERS:
            blocks = recorder.blocks if order == "block" else [round_robin(*block) for block in recorder.blocks]
            arms = np.concatenate([b[0] for b in blocks]) if blocks else np.empty(0, dtype=int)
            rewards = (np.concatenate([b[1] for b in blocks]) if blocks
                       else np.empty((0, environment.num_objectives)))
            pieces[order][0].append(stamps)
            pieces[order][1].append(arms[:used].astype(np.int64))
            pieces[order][2].append(rewards[:used])
        stage_rows.append({
            "environment": environment_name, "algorithm": algorithm, "base_algorithm": base,
            "repetition": repetition, "stage": item["stage"], "stage_budget": item["budget"],
            "stage_start": item["start"], "stage_end": item["end"], "completed": int(item["completed"]),
            "actual_draws": actual, "used_draws": used, "recommendation": encode_int_set(recommendation),
            "config_hash": digest,
        })

    points = evaluation_points(spec)
    encoded_points = encode_int_set(points)
    files = {}
    for order, suffix in ORDERS.items():
        times, arms, rewards = (np.concatenate(part) for part in pieces[order])
        rows = []
        for arm_pull, recommendation, drawn, unsampled in empirical_trajectory(times, arms, rewards, points, K):
            recommendation = list(recommendation)
            row = build_row(
                environment=environment_name, instance_hash=instance_hash, algorithm=f"{algorithm}{suffix}",
                posterior_model="", repetition=repetition, arm_pull=arm_pull, max_budget=budget,
                evaluation_grid=points, rho=None, master_seed=master_seed,
                recommendation=recommendation, true_pareto=true_pareto,
                bernoulli=environment.bernoulli_metric(recommendation),
                jaccard=environment.jaccard_metric(recommendation),
                misidentification=environment.mis_id_metric(recommendation),
            )
            row["evaluation_grid"] = encoded_points
            row.update({"base_algorithm": base, "active_stage": active_stage(plan, arm_pull),
                        "samples_drawn": drawn, "unsampled_arms": unsampled, "config_hash": digest})
            rows.append(row)
        name = f"trajectory{suffix.lower()}.csv"
        atomic_csv(directory / name, TRAJECTORY_COLUMNS, rows)
        files[name] = sha256_file(directory / name)
    atomic_csv(directory / "stages.csv", STAGE_COLUMNS, stage_rows)
    files["stages.csv"] = sha256_file(directory / "stages.csv")
    seconds = time.perf_counter() - start_time
    atomic_json(directory / "complete.json", {"config_hash": digest, "files": files, "seconds": seconds})
    return {"environment": environment_name, "algorithm": algorithm, "repetition": repetition,
            "skipped": False, "seconds": seconds}


def results_path(output_root: Path, environment: str, kind: str) -> Path:
    suffix = {"trajectory": "", "block": "_block_order", "stages": "_stages"}[kind]
    return Path(output_root) / "empirical" / f"{environment}{suffix}.csv"


def manifest_path(output_root: Path, environment: str) -> Path:
    return Path(output_root) / "empirical" / "manifests" / f"{environment}.json"


def merge_environment(output_root: Path, environment_name: str, config: dict, algorithms: list[str],
                      repetitions: list[int], provenance: dict) -> dict[str, int]:
    digest = config_hash(config)
    spec = environment_spec(environment_name, config)
    collected = {"trajectory": [], "block": [], "stages": []}
    for algorithm in algorithms:
        for repetition in repetitions:
            directory = task_dir(output_root, environment_name, algorithm, repetition)
            if not _task_complete(directory, digest):
                raise ValueError(f"{environment_name}/{algorithm}/rep{repetition}: incomplete or stale task")
            collected["trajectory"].extend(read_csv(directory / "trajectory.csv")[1])
            collected["block"].extend(read_csv(directory / "trajectory_block.csv")[1])
            collected["stages"].extend(read_csv(directory / "stages.csv")[1])
    points = len(evaluation_points(spec))
    stages = len(stage_plan(spec["num_arms"], spec["budget"]))
    expected = {
        "trajectory": len(algorithms) * len(repetitions) * points,
        "block": len(algorithms) * len(repetitions) * points,
        "stages": len(algorithms) * len(repetitions) * stages,
    }
    files = {}
    for kind, rows in collected.items():
        if len(rows) != expected[kind]:
            raise ValueError(f"{environment_name}: {len(rows)} {kind} rows, expected {expected[kind]}")
        columns = STAGE_COLUMNS if kind == "stages" else TRAJECTORY_COLUMNS
        path = results_path(output_root, environment_name, kind)
        atomic_csv(path, columns, rows)
        files[kind] = {"path": str(path.resolve()), "sha256": sha256_file(path), "rows": len(rows)}
    atomic_json(manifest_path(output_root, environment_name), {
        "schema_version": "information-matched-empirical-doubling-v1",
        "config_hash": digest,
        "empirical_doubling": config["empirical_doubling"],
        "environment": spec,
        "algorithms": algorithms,
        "repetitions": repetitions,
        "evaluation_points": evaluation_points(spec),
        "stage_plan": stage_plan(spec["num_arms"], spec["budget"]),
        "files": files,
        **provenance,
    })
    return {kind: item["rows"] for kind, item in files.items()}


def run(output_root: Path | None = None, config: dict | None = None, environments: list[str] | None = None,
        algorithms: list[str] | None = None, repetitions: list[int] | None = None, n_jobs: int = 1,
        allow_dirty: bool = False) -> dict:
    config = load_config() if config is None else config
    output_root = Path(default_output_root(config) if output_root is None else output_root)
    assert_safe_output(output_root, config)
    provenance = source_provenance(allow_dirty=allow_dirty)
    verify_sources(config)
    verify_protected(config)
    names = environments or [spec["name"] for spec in environment_specs(config)]
    algorithms = algorithms or list(EMPIRICAL_ALGORITHMS)
    repetitions = list(range(int(config["num_runs"]))) if repetitions is None else list(repetitions)
    budgets = {spec["name"]: spec["budget"] for spec in environment_specs(config)}
    tasks = [(name, algorithm, repetition) for name in sorted(names, key=lambda n: -budgets[n])
             for algorithm in algorithms for repetition in repetitions]
    start = time.perf_counter()
    if n_jobs == 1:
        for task in tasks:
            run_task(str(output_root), *task, config)
    else:
        Parallel(n_jobs=n_jobs, batch_size=1)(delayed(run_task)(str(output_root), *task, config) for task in tasks)
    print(f"empirical D-EGE tasks finished in {time.perf_counter() - start:.1f} s", flush=True)
    counts = {name: merge_environment(output_root, name, config, algorithms, repetitions, provenance) for name in names}
    verify_sources(config)
    verify_protected(config)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--environments", default=None)
    parser.add_argument("--repetitions", default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--allow-dirty-source", action="store_true")
    args = parser.parse_args()
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
