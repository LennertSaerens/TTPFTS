#!/usr/bin/env python3
"""Exact, state-instrumented replay of final TTPFTS_UNKNOWN_VAR trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from experiments.final_config_loader import load_final_config
from momab_main import ENVIRONMENTS, _construct_environment, _make_bandit
from runner.instance import compute_instance_hash
from runner.schema import build_row
from runner.seeding import seed_rngs

from .config import config_hash, environment_config, load_config
from .metrics import uncertainty_from_tposterior_state


UQ_COLUMNS = [
    "environment", "instance_hash", "algorithm", "posterior_model", "repetition",
    "arm_pull", "max_budget", "normalized_budget", "evaluation_grid", "rho",
    "master_seed", "bernoulli", "jaccard", "misidentification", "recommendation",
    "true_pareto", "uq_available", "uq_reason", "bhattacharyya", "symmetric_kl",
    "entropy", "first_front_size", "second_front_size", "uq_config_hash",
]
STATE_SCHEMA = "tposterior-welford-v1"


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-uq-", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as handle:
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


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp-uq-", suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def shard_paths(run_dir: Path, environment: str, repetition: int) -> tuple[Path, Path]:
    root = run_dir / "shards" / environment
    return root / f"rep{repetition:05d}.csv", root / f"rep{repetition:05d}.npz"


def _decode_scalar(array: np.ndarray) -> str:
    value = array.item()
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def shard_is_complete(run_dir: Path, environment: str, repetition: int, grid: list[int], digest: str) -> bool:
    csv_path, state_path = shard_paths(run_dir, environment, repetition)
    if not csv_path.is_file() or not state_path.is_file():
        return False
    try:
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or set(rows[0]) != set(UQ_COLUMNS):
            return False
        if [int(row["arm_pull"]) for row in rows] != grid:
            return False
        if any(row["uq_config_hash"] != digest for row in rows):
            return False
        with np.load(state_path, allow_pickle=False) as state:
            if _decode_scalar(state["schema_version"]) != STATE_SCHEMA:
                return False
            if _decode_scalar(state["uq_config_hash"]) != digest:
                return False
            if state["grid"].tolist() != grid:
                return False
            points = len(grid)
            if state["n"].shape[0] != points or state["means"].shape != state["m2"].shape:
                return False
            if state["means"].shape[0] != points or state["means"].shape[1:] != state["n"].shape[1:]:
                return False
    except (OSError, ValueError, KeyError, csv.Error):
        return False
    return True


def _build_uq_row(
    *, environment, env_name: str, instance_hash: str, grid: list[int], budget: int,
    master_seed: int, repetition: int, arm_pull: int, bandit, digest: str,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    posterior = bandit.posterior
    n = posterior.n.copy()
    means = posterior.mean.copy()
    m2 = posterior.M2.copy()
    if arm_pull == 0:
        result = None
        available, reason = False, "budget_zero"
        first_size = second_size = 0
    else:
        result = uncertainty_from_tposterior_state(n, means, m2, alpha=posterior.alpha)
        available, reason = result.available, result.reason
        first_size, second_size = result.first_front_size, result.second_front_size
    recommendation = list(bandit.get_top_arms())
    true_pareto = list(environment.get_top_arms())
    base = build_row(
        environment=env_name, instance_hash=instance_hash, algorithm="TTPFTS_UNKNOWN_VAR",
        posterior_model=type(posterior).__name__, repetition=repetition, arm_pull=arm_pull,
        max_budget=budget, evaluation_grid=grid, rho=bandit.p, master_seed=master_seed,
        recommendation=recommendation, true_pareto=true_pareto,
        bernoulli=environment.bernoulli_metric(recommendation),
        jaccard=environment.jaccard_metric(recommendation),
        misidentification=environment.mis_id_metric(recommendation),
    )
    base.update(
        normalized_budget=arm_pull / budget,
        uq_available=int(available),
        uq_reason=reason,
        bhattacharyya="" if not available else result.bhattacharyya,
        symmetric_kl="" if not available else result.symmetric_kl,
        entropy="" if not available else result.entropy,
        first_front_size=first_size,
        second_front_size=second_size,
        uq_config_hash=digest,
    )
    return base, n, means, m2


def replay_repetition(environment_name: str, repetition: int, run_dir: Path, config: dict) -> None:
    """Replay one final trajectory and atomically persist row/state checkpoints."""
    environment_details = environment_config(environment_name, config)
    budget = environment_details["budget"]
    grid = environment_details["grid"]
    digest = config_hash(config)
    if shard_is_complete(run_dir, environment_name, repetition, grid, digest):
        return
    final_config = load_final_config()
    environment_class, _legacy_budget = ENVIRONMENTS[environment_name]
    environment = _construct_environment(environment_class, final_config["reward_model"]["distribution"])
    env_tag = f"{environment_name}|{final_config['reward_model']['distribution']}"
    seed_rngs(config["master_seed"], env_tag, "env", repetition)
    environment.reset()
    instance_hash = compute_instance_hash(environment)
    if instance_hash != environment_details["instance_hash"]:
        raise ValueError(f"{environment_name}: instance hash {instance_hash} differs from frozen config")
    seed_rngs(config["master_seed"], env_tag, "TTPFTS_UNKNOWN_VAR", repetition)
    bandit = _make_bandit("TTPFTS_UNKNOWN_VAR", environment, p=0.5)
    bandit.reset(environment.stds)
    grid_set = set(grid)
    rows, counts, means, m2 = [], [], [], []

    def log_checkpoint(arm_pull: int) -> None:
        row, n, posterior_means, posterior_m2 = _build_uq_row(
            environment=environment, env_name=environment_name, instance_hash=instance_hash,
            grid=grid, budget=budget, master_seed=config["master_seed"], repetition=repetition,
            arm_pull=arm_pull, bandit=bandit, digest=digest,
        )
        rows.append(row)
        counts.append(n)
        means.append(posterior_means)
        m2.append(posterior_m2)

    log_checkpoint(0)
    for pull in range(1, budget + 1):
        arm = bandit.choose_arm()
        bandit.learn(arm, environment.pull_arm(arm))
        if pull in grid_set:
            log_checkpoint(pull)
    csv_path, state_path = shard_paths(run_dir, environment_name, repetition)
    _atomic_csv(csv_path, rows)
    _atomic_npz(
        state_path,
        schema_version=np.array(STATE_SCHEMA),
        uq_config_hash=np.array(digest),
        environment=np.array(environment_name),
        repetition=np.array(repetition, dtype=np.int64),
        grid=np.asarray(grid, dtype=np.int64),
        n=np.asarray(counts, dtype=np.int64),
        means=np.asarray(means, dtype=np.float64),
        m2=np.asarray(m2, dtype=np.float64),
    )


def replay(environment_names: list[str], repetitions: list[int], run_dir: Path, config: dict, n_jobs: int) -> None:
    work = [
        (environment_name, repetition)
        for environment_name in environment_names
        for repetition in repetitions
        if not shard_is_complete(
            run_dir, environment_name, repetition, environment_config(environment_name, config)["grid"], config_hash(config)
        )
    ]
    if not work:
        print("All requested UQ shards are complete.")
        return
    if n_jobs == 1:
        for environment_name, repetition in work:
            replay_repetition(environment_name, repetition, run_dir, config)
    else:
        Parallel(n_jobs=n_jobs)(
            delayed(replay_repetition)(environment_name, repetition, run_dir, config)
            for environment_name, repetition in work
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--environments", default=None, help="Comma-separated EgeExp subset.")
    parser.add_argument("--repetitions", default=None, help="Comma-separated repetition IDs.")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    config = load_config()
    names = [item["name"] for item in config["environments"]] if "environments" in config else [
        environment["name"] for environment in load_final_config()["environments"]
    ]
    if args.environments:
        names = args.environments.split(",")
    repetitions = list(range(config["num_runs"])) if args.repetitions is None else [
        int(value) for value in args.repetitions.split(",")
    ]
    if any(repetition < 0 or repetition >= config["num_runs"] for repetition in repetitions):
        raise ValueError("repetition outside frozen UQ configuration")
    replay(names, repetitions, args.run_dir, config, args.n_jobs)


if __name__ == "__main__":
    main()
