#!/usr/bin/env python3
"""Run exact, resumable hardness-budget TTPFTS p-sweep trajectories locally."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

# These variables are inherited by joblib worker processes. The computation is
# trajectory-parallel, so nested BLAS/OpenMP parallelism only creates contention.
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

from joblib import Parallel, delayed

from bandits.Posteriors import TPosterior
from bandits.TTPFTS import TTPFTSBandit
from experiments.final_config_loader import load_final_config
from momab_main import ENVIRONMENTS, _construct_environment
from runner.instance import compute_instance_hash
from runner.schema import SCHEMA_COLUMNS, build_row
from runner.seeding import seed_rngs

from .config import (
    algorithm_label,
    config_hash,
    default_output_root,
    environment_config,
    expected_environments,
    load_config,
    p_tag,
)


SWEEP_COLUMNS = [*SCHEMA_COLUMNS, "normalized_budget", "sweep_config_hash"]


def shard_path(run_dir: Path, environment: str, p: float, repetition: int) -> Path:
    """Return the unique atomic result shard for one sweep trajectory."""
    return run_dir / "shards" / environment / p_tag(p) / f"rep{repetition:05d}.csv"


def _atomic_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-p-sweep-", suffix=".csv")
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


def shard_is_complete(
    run_dir: Path, environment: str, p: float, repetition: int, grid: list[int], digest: str
) -> bool:
    """Return whether a shard has the exact schema, grid, p, and config binding."""
    path = shard_path(run_dir, environment, p, repetition)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        return (
            len(rows) == len(grid)
            and bool(rows)
            and set(rows[0]) == set(SWEEP_COLUMNS)
            and [int(row["arm_pull"]) for row in rows] == grid
            and all(row["sweep_config_hash"] == digest for row in rows)
            and all(row["algorithm"] == algorithm_label(p) for row in rows)
            and all(float(row["rho"]) == float(p) for row in rows)
        )
    except (OSError, ValueError, csv.Error):
        return False


def make_sweep_bandit(environment, p: float) -> TTPFTSBandit:
    """Construct the current unknown-variance paper bandit at a swept p value."""
    return TTPFTSBandit(
        TPosterior(environment.num_arms, environment.num_objectives, alpha=0.0),
        p=float(p),
        num_warmup_pulls=4,
    )


def run_repetition(
    environment_name: str, p: float, repetition: int, run_dir: Path, config: dict
) -> Path:
    """Run exactly one `(environment, p, repetition)` trajectory and atomically save it."""
    details = environment_config(environment_name, config)
    budget, grid = details["budget"], details["grid"]
    digest = config_hash(config)
    if shard_is_complete(run_dir, environment_name, p, repetition, grid, digest):
        return shard_path(run_dir, environment_name, p, repetition)

    final_config = load_final_config()
    environment_class, _legacy_budget = ENVIRONMENTS[environment_name]
    environment = _construct_environment(environment_class, final_config["reward_model"]["distribution"])
    environment_tag = f"{environment_name}|{final_config['reward_model']['distribution']}"

    # The environment is shared across all p values. Sampling randomness stays
    # in the fixed TTPFTS namespace, yielding an exact p=0.5 identity gate.
    seed_rngs(config["master_seed"], environment_tag, "env", repetition)
    environment.reset()
    instance_hash = compute_instance_hash(environment)
    if instance_hash != details["instance_hash"]:
        raise ValueError(
            f"{environment_name}: instance hash {instance_hash} differs from frozen final config"
        )
    seed_rngs(config["master_seed"], environment_tag, config["seeding"]["bandit_namespace"], repetition)
    bandit = make_sweep_bandit(environment, p)
    bandit.reset(environment.stds)

    true_pareto = list(environment.get_top_arms())
    grid_set = set(grid)
    rows: list[dict] = []

    def log_checkpoint(arm_pull: int) -> None:
        recommendation = list(bandit.get_top_arms())
        row = build_row(
            environment=environment_name,
            instance_hash=instance_hash,
            algorithm=algorithm_label(p),
            posterior_model=type(bandit.posterior).__name__,
            repetition=repetition,
            arm_pull=arm_pull,
            max_budget=budget,
            evaluation_grid=grid,
            rho=p,
            master_seed=config["master_seed"],
            recommendation=recommendation,
            true_pareto=true_pareto,
            bernoulli=environment.bernoulli_metric(recommendation),
            jaccard=environment.jaccard_metric(recommendation),
            misidentification=environment.mis_id_metric(recommendation),
        )
        row["normalized_budget"] = arm_pull / budget
        row["sweep_config_hash"] = digest
        rows.append(row)

    log_checkpoint(0)
    for arm_pull in range(1, budget + 1):
        arm = bandit.choose_arm()
        bandit.learn(arm, environment.pull_arm(arm))
        if arm_pull in grid_set:
            log_checkpoint(arm_pull)

    path = shard_path(run_dir, environment_name, p, repetition)
    _atomic_write(path, rows)
    return path


def run(
    environment_names: list[str],
    p_values: list[float],
    repetitions: list[int],
    run_dir: Path,
    config: dict,
    n_jobs: int,
) -> int:
    """Run all incomplete requested trajectories with a globally balanced work queue."""
    digest = config_hash(config)
    work = [
        (environment_name, p, repetition)
        for environment_name in environment_names
        for p in p_values
        for repetition in repetitions
        if not shard_is_complete(
            run_dir,
            environment_name,
            p,
            repetition,
            environment_config(environment_name, config)["grid"],
            digest,
        )
    ]
    work.sort(key=lambda item: -environment_config(item[0], config)["budget"])
    if not work:
        print("All requested p-sweep shards are complete.")
        return 0
    print(f"Running {len(work)} trajectories with {n_jobs} worker(s).")
    if n_jobs == 1:
        for environment_name, p, repetition in work:
            run_repetition(environment_name, p, repetition, run_dir, config)
    else:
        Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(run_repetition)(environment_name, p, repetition, run_dir, config)
            for environment_name, p, repetition in work
        )
    return len(work)


def _parse_csv_values(value: str | None, transform) -> list:
    return [] if value is None else [transform(part) for part in value.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--environments", default=None, help="Comma-separated EgeExp subset.")
    parser.add_argument("--p-values", default=None, help="Comma-separated approved p-value subset.")
    parser.add_argument("--repetitions", default=None, help="Comma-separated repetition ID subset.")
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    config = load_config()
    environment_names = _parse_csv_values(args.environments, str) or [
        item["name"] for item in config_environments(config)
    ]
    p_values = _parse_csv_values(args.p_values, float) or [float(value) for value in config["p_values"]]
    repetitions = _parse_csv_values(args.repetitions, int) or list(range(config["num_runs"]))
    known_environments = {item["name"] for item in config_environments(config)}
    if set(environment_names) - known_environments:
        raise ValueError(f"unknown environment(s): {sorted(set(environment_names) - known_environments)}")
    if any(round(value, 1) != value or value not in config["p_values"] for value in p_values):
        raise ValueError("p-values must be approved tenths from 0.0 through 1.0")
    if any(repetition < 0 or repetition >= config["num_runs"] for repetition in repetitions):
        raise ValueError("repetition outside frozen p-sweep configuration")
    if args.n_jobs < 1:
        raise ValueError("n-jobs must be positive")
    run_dir = args.run_dir or default_output_root(config) / "run"
    run(environment_names, p_values, repetitions, run_dir, config, args.n_jobs)


def config_environments(config: dict) -> list[dict]:
    """Return final-config-bound environment specs without duplicating constants."""
    del config
    return expected_environments()


if __name__ == "__main__":
    main()
