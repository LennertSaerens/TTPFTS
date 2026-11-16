#!/usr/bin/env python3
"""Exact state-instrumented replay of TTPFTS_KNOWN_VAR (NormalPosterior) trajectories."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from experiments.final_config_loader import load_final_config
from experiments.uq.metrics import (
    uncertainty_from_moments,
    uq_directional_bc_from_moments,
    uq_dominance_risk_from_moments,
    uq_mc_disagreement_from_moments,
)
from momab_main import ENVIRONMENTS, _construct_environment, _make_bandit
from runner.instance import compute_instance_hash
from runner.schema import build_row
from runner.seeding import seed_rngs

from .config import config_hash, environment_config, load_config


UQ_KV_COLUMNS = [
    "environment", "instance_hash", "algorithm", "posterior_model", "repetition",
    "arm_pull", "max_budget", "normalized_budget", "evaluation_grid", "rho",
    "master_seed", "bernoulli", "jaccard", "misidentification", "recommendation",
    "true_pareto", "uq_available", "uq_reason", "eq10_nonempty_second",
    "bhattacharyya", "bhattacharyya_raw", "uq_v1", "uq_v2", "uq_mc", "uq_mc_se",
    "symmetric_kl", "entropy", "first_front_size", "second_front_size",
    "time_eq10_ms", "time_v1_ms", "time_v2_ms", "time_mc_ms", "uq_config_hash",
]
STATE_SCHEMA = "normal-posterior-kv-v1"


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-uqkv-", suffix=".csv")
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


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp-uqkv-", suffix=".npz", delete=False) as handle:
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
        if not rows or set(rows[0]) != set(UQ_KV_COLUMNS):
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
            if state["counts"].shape[0] != points or state["means"].shape != state["stds"].shape:
                return False
            if state["means"].shape[0] != points or state["means"].shape[1:] != state["counts"].shape[1:]:
                return False
    except (OSError, ValueError, KeyError, csv.Error):
        return False
    return True


def _build_kv_row(
    *,
    environment,
    env_name: str,
    env_index: int,
    instance_hash: str,
    grid: list[int],
    budget: int,
    master_seed: int,
    mc_seed_base: int,
    mc_draws: int,
    repetition: int,
    arm_pull: int,
    bandit,
    digest: str,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    posterior = bandit.posterior
    counts = posterior.counts.copy()
    means = posterior.get_mean().copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        stds = np.where(counts > 0, posterior.known_stds / np.sqrt(np.maximum(counts, 1)), 1e6)

    recommendation = list(bandit.get_top_arms())
    true_pareto = list(environment.get_top_arms())
    base = build_row(
        environment=env_name,
        instance_hash=instance_hash,
        algorithm="TTPFTS_KNOWN_VAR",
        posterior_model=type(posterior).__name__,
        repetition=repetition,
        arm_pull=arm_pull,
        max_budget=budget,
        evaluation_grid=grid,
        rho=bandit.p,
        master_seed=master_seed,
        recommendation=recommendation,
        true_pareto=true_pareto,
        bernoulli=environment.bernoulli_metric(recommendation),
        jaccard=environment.jaccard_metric(recommendation),
        misidentification=environment.mis_id_metric(recommendation),
    )

    if arm_pull == 0 or np.any(counts <= 0):
        base.update(
            normalized_budget=arm_pull / budget,
            uq_available=0,
            uq_reason="budget_zero" if arm_pull == 0 else "unobserved_arm",
            eq10_nonempty_second=0,
            bhattacharyya="",
            bhattacharyya_raw="",
            uq_v1="",
            uq_v2="",
            uq_mc="",
            uq_mc_se="",
            symmetric_kl="",
            entropy="",
            first_front_size=0,
            second_front_size=0,
            time_eq10_ms="",
            time_v1_ms="",
            time_v2_ms="",
            time_mc_ms="",
            uq_config_hash=digest,
        )
        return base, counts, means, stds

    t0 = time.perf_counter()
    eq10_res = uncertainty_from_moments(means, stds)
    t_eq10 = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    v1_val = uq_directional_bc_from_moments(means, stds)
    t_v1 = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    v2_val = uq_dominance_risk_from_moments(means, stds)
    t_v2 = (time.perf_counter() - t0) * 1000.0

    mc_rng = np.random.default_rng(
        np.random.SeedSequence([int(mc_seed_base), int(env_index), int(repetition), int(arm_pull)])
    )
    t0 = time.perf_counter()
    mc_q, mc_se = uq_mc_disagreement_from_moments(means, stds, n_draws=mc_draws, rng=mc_rng)
    t_mc = (time.perf_counter() - t0) * 1000.0

    if eq10_res.available:
        eq10_val = float(eq10_res.bhattacharyya)
        eq10_raw = float(eq10_res.bhattacharyya)
        kl_val = float(eq10_res.symmetric_kl)
        ent_val = float(eq10_res.entropy)
        nonempty_second = 1
        f1_size, f2_size = eq10_res.first_front_size, eq10_res.second_front_size
    else:
        eq10_val = 0.0
        eq10_raw = ""
        kl_val = ""
        ent_val = ""
        nonempty_second = 0
        f1_size, f2_size = eq10_res.first_front_size, 0

    base.update(
        normalized_budget=arm_pull / budget,
        uq_available=1,
        uq_reason="available",
        eq10_nonempty_second=nonempty_second,
        bhattacharyya=eq10_val,
        bhattacharyya_raw=eq10_raw,
        uq_v1=float(v1_val),
        uq_v2=float(v2_val),
        uq_mc=float(mc_q),
        uq_mc_se=float(mc_se),
        symmetric_kl=kl_val,
        entropy=ent_val,
        first_front_size=f1_size,
        second_front_size=f2_size,
        time_eq10_ms=t_eq10,
        time_v1_ms=t_v1,
        time_v2_ms=t_v2,
        time_mc_ms=t_mc,
        uq_config_hash=digest,
    )
    return base, counts, means, stds


def replay_repetition(environment_name: str, repetition: int, run_dir: Path, config: dict) -> None:
    env_details = environment_config(environment_name, config)
    budget = env_details["budget"]
    grid = env_details["grid"]
    digest = config_hash(config)
    if shard_is_complete(run_dir, environment_name, repetition, grid, digest):
        return

    final_config = load_final_config()
    env_names = [item["name"] for item in final_config["environments"]]
    env_index = env_names.index(environment_name)
    environment_class, _legacy_budget = ENVIRONMENTS[environment_name]
    environment = _construct_environment(environment_class, final_config["reward_model"]["distribution"])
    env_tag = f"{environment_name}|{final_config['reward_model']['distribution']}"

    seed_rngs(config["master_seed"], env_tag, "env", repetition)
    environment.reset()
    instance_hash = compute_instance_hash(environment)
    if instance_hash != env_details["instance_hash"]:
        raise ValueError(f"{environment_name}: instance hash {instance_hash} differs from frozen config")

    seed_rngs(config["master_seed"], env_tag, "TTPFTS_KNOWN_VAR", repetition)
    bandit = _make_bandit("TTPFTS_KNOWN_VAR", environment, p=0.5)
    bandit.reset(environment.stds)

    grid_set = set(grid)
    rows, counts_list, means_list, stds_list = [], [], [], []

    def log_checkpoint(arm_pull: int) -> None:
        row, counts, means, stds = _build_kv_row(
            environment=environment,
            env_name=environment_name,
            env_index=env_index,
            instance_hash=instance_hash,
            grid=grid,
            budget=budget,
            master_seed=config["master_seed"],
            mc_seed_base=int(config["mc_seed_base"]),
            mc_draws=int(config["mc_draws"]),
            repetition=repetition,
            arm_pull=arm_pull,
            bandit=bandit,
            digest=digest,
        )
        rows.append(row)
        counts_list.append(counts)
        means_list.append(means)
        stds_list.append(stds)

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
        counts=np.asarray(counts_list, dtype=np.int64),
        means=np.asarray(means_list, dtype=np.float64),
        stds=np.asarray(stds_list, dtype=np.float64),
    )


def replay(environment_names: list[str], repetitions: list[int], run_dir: Path, config: dict, n_jobs: int) -> None:
    digest = config_hash(config)
    work = [
        (env_name, rep)
        for env_name in environment_names
        for rep in repetitions
        if not shard_is_complete(run_dir, env_name, rep, environment_config(env_name, config)["grid"], digest)
    ]
    if not work:
        print("All requested UQ KV shards are complete.")
        return
    if n_jobs == 1:
        for env_name, rep in work:
            replay_repetition(env_name, rep, run_dir, config)
    else:
        Parallel(n_jobs=n_jobs)(
            delayed(replay_repetition)(env_name, rep, run_dir, config) for env_name, rep in work
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--environments", default=None)
    parser.add_argument("--repetitions", default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    config = load_config()
    names = [env["name"] for env in load_final_config()["environments"]]
    if args.environments:
        names = args.environments.split(",")
    repetitions = list(range(config["num_runs"])) if args.repetitions is None else [
        int(v) for v in args.repetitions.split(",")
    ]
    replay(names, repetitions, args.run_dir, config, args.n_jobs)


if __name__ == "__main__":
    main()
