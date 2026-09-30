#!/usr/bin/env python3
"""Execute resumable heterogeneous-noise ensemble tasks with atomic shards."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
import tempfile
from pathlib import Path
from typing import Any

# Task-level process parallelism is the only intended parallelism.
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
from joblib import Parallel, delayed

from bandits.ege_kone import EGE_SH, EGE_SR
from momab_main import _make_bandit
from runner.schema import SCHEMA_COLUMNS, build_row
from runner.seeding import derive_seed, seed_rngs

from .config import ALGORITHMS, config_hash, default_output_root, load_config
from .instances import FrozenHeterogeneousEnvironment, FrozenInstance, load_frozen_instances


EGE_ALGORITHMS = {"EGE_SH": EGE_SH, "EGE_SR": EGE_SR}
ANYTIME_ALGORITHMS = ("Uniform", "APE", "TTPFTS_UNKNOWN_VAR")
STATE_SCHEMA = "heterogeneous-ensemble-anytime-state-v1"
RESULT_COLUMNS = [
    *SCHEMA_COLUMNS,
    "proposal_id",
    "num_arms",
    "num_objectives",
    "pareto_size",
    "hardness",
    "noise_diversity",
    "grid_index",
    "normalized_budget",
    "execution_kind",
    "actual_reward_draws",
    "algorithm_seed",
    "noise_knowledge",
    "ape_sigma_bound",
    "warmup_pulls_per_arm",
    "ensemble_config_hash",
]


class CountingEnvironment(FrozenHeterogeneousEnvironment):
    """Frozen environment that records raw arm reward draws without changing samples."""

    def __init__(self, means: np.ndarray, stds: np.ndarray) -> None:
        super().__init__(means, stds)
        self.reward_draws = 0

    def pull_arm(self, arm: int) -> np.ndarray:
        self.reward_draws += 1
        return super().pull_arm(arm)

    def sample(self, arms) -> np.ndarray:
        arms = np.asarray(arms, dtype=int)
        self.reward_draws += int(arms.size)
        return super().sample(arms)


def instance_dir(run_dir: Path, instance: FrozenInstance) -> Path:
    """Return the config-independent shard directory for one immutable instance."""
    return run_dir / "shards" / instance.name


def anytime_paths(run_dir: Path, instance: FrozenInstance, algorithm: str) -> tuple[Path, Path]:
    root = instance_dir(run_dir, instance) / algorithm
    return root / "trajectory.csv", root / "trajectory.state.pkl"


def ege_path(run_dir: Path, instance: FrozenInstance, algorithm: str, budget: int) -> Path:
    return instance_dir(run_dir, instance) / algorithm / f"budget_{budget:09d}.csv"


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-ensemble-", suffix=".csv")
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


def _atomic_pickle(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-ensemble-", suffix=".pkl")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_matches(
    row: dict[str, str],
    instance: FrozenInstance,
    algorithm: str,
    arm_pull: int,
    grid_index: int,
    execution_kind: str,
    digest: str,
) -> bool:
    try:
        return (
            set(row) == set(RESULT_COLUMNS)
            and row["environment"] == instance.name
            and row["instance_hash"] == instance.instance_hash
            and row["algorithm"] == algorithm
            and int(row["repetition"]) == 0
            and int(row["arm_pull"]) == arm_pull
            and int(row["max_budget"]) == instance.budget
            and int(row["proposal_id"]) == instance.proposal_id
            and int(row["num_arms"]) == instance.num_arms
            and int(row["num_objectives"]) == instance.num_objectives
            and int(row["pareto_size"]) == len(instance.pareto_indices)
            and int(row["grid_index"]) == grid_index
            and row["execution_kind"] == execution_kind
            and row["ensemble_config_hash"] == digest
        )
    except (KeyError, TypeError, ValueError):
        return False


def anytime_is_complete(run_dir: Path, instance: FrozenInstance, algorithm: str, digest: str) -> bool:
    """Return whether an anytime trajectory has all and only its frozen checkpoints."""
    csv_path, state_path = anytime_paths(run_dir, instance, algorithm)
    if not csv_path.is_file() or state_path.exists():
        return False
    try:
        rows = _read_rows(csv_path)
        return (
            len(rows) == len(instance.grid)
            and [int(row["arm_pull"]) for row in rows] == list(instance.grid)
            and all(
                _row_matches(row, instance, algorithm, arm_pull, index, "anytime", digest)
                for index, (row, arm_pull) in enumerate(zip(rows, instance.grid))
            )
            and all(int(row["actual_reward_draws"]) == int(row["arm_pull"]) for row in rows)
        )
    except (OSError, ValueError, csv.Error):
        return False


def ege_is_complete(
    run_dir: Path, instance: FrozenInstance, algorithm: str, budget: int, digest: str
) -> bool:
    """Return whether one independently executable EGE budget cell is complete."""
    try:
        index = list(instance.grid).index(budget)
    except ValueError:
        return False
    path = ege_path(run_dir, instance, algorithm, budget)
    if not path.is_file():
        return False
    try:
        rows = _read_rows(path)
        return (
            len(rows) == 1
            and _row_matches(rows[0], instance, algorithm, budget, index, "ege", digest)
            and int(rows[0]["actual_reward_draws"]) >= 0
        )
    except (OSError, ValueError, csv.Error):
        return False


def _construct_bandit(algorithm: str, environment: CountingEnvironment):
    if algorithm not in ANYTIME_ALGORITHMS:
        raise ValueError(f"{algorithm} is not an anytime algorithm")
    bandit = _make_bandit(algorithm, environment, p=0.5)
    if bandit is None:
        raise ValueError(f"could not construct {algorithm}")
    bandit.reset(environment.stds)
    return bandit


def _semantics(algorithm: str, environment: CountingEnvironment) -> dict[str, Any]:
    if algorithm == "APE":
        return {
            "execution_kind": "anytime",
            "noise_knowledge": "global_max_standard_deviation_bound",
            "ape_sigma_bound": float(np.max(environment.stds)),
            "warmup_pulls_per_arm": 0,
        }
    if algorithm == "TTPFTS_UNKNOWN_VAR":
        return {
            "execution_kind": "anytime",
            "noise_knowledge": "unknown_variance_tposterior",
            "ape_sigma_bound": "",
            "warmup_pulls_per_arm": 4,
        }
    if algorithm == "Uniform":
        return {
            "execution_kind": "anytime",
            "noise_knowledge": "none",
            "ape_sigma_bound": "",
            "warmup_pulls_per_arm": 0,
        }
    if algorithm in EGE_ALGORITHMS:
        return {
            "execution_kind": "ege",
            "noise_knowledge": "none",
            "ape_sigma_bound": "",
            "warmup_pulls_per_arm": 0,
        }
    raise ValueError(f"unknown ensemble algorithm {algorithm}")


def _build_result_row(
    *,
    instance: FrozenInstance,
    environment: CountingEnvironment,
    algorithm: str,
    arm_pull: int,
    grid_index: int,
    recommendation,
    posterior_model: str,
    rho,
    actual_reward_draws: int,
    algorithm_seed: int,
    digest: str,
    master_seed: int,
) -> dict[str, Any]:
    """Build one self-describing result row for either execution family."""
    details = _semantics(algorithm, environment)
    true_pareto = list(instance.pareto_indices)
    base = build_row(
        environment=instance.name,
        instance_hash=instance.instance_hash,
        algorithm=algorithm,
        posterior_model=posterior_model,
        repetition=0,
        arm_pull=arm_pull,
        max_budget=instance.budget,
        evaluation_grid=instance.grid,
        rho=rho,
        master_seed=master_seed,
        recommendation=recommendation,
        true_pareto=true_pareto,
        bernoulli=environment.bernoulli_metric(recommendation),
        jaccard=environment.jaccard_metric(recommendation),
        misidentification=environment.mis_id_metric(recommendation),
    )
    base.update(
        {
            "proposal_id": instance.proposal_id,
            "num_arms": instance.num_arms,
            "num_objectives": instance.num_objectives,
            "pareto_size": len(instance.pareto_indices),
            "hardness": instance.hardness.hardness,
            "noise_diversity": instance.noise_diversity,
            "grid_index": grid_index,
            "normalized_budget": arm_pull / instance.budget,
            "actual_reward_draws": actual_reward_draws,
            "algorithm_seed": algorithm_seed,
            "ensemble_config_hash": digest,
            **details,
        }
    )
    return base


def _validate_state(
    state: dict[str, Any], instance: FrozenInstance, algorithm: str, digest: str
) -> None:
    if state.get("state_schema") != STATE_SCHEMA:
        raise ValueError("anytime state schema mismatch")
    expected = {
        "ensemble_config_hash": digest,
        "instance_hash": instance.instance_hash,
        "algorithm": algorithm,
        "budget": instance.budget,
        "grid": list(instance.grid),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"anytime state {key} mismatch")
    next_pull = state.get("next_pull")
    if not isinstance(next_pull, int) or not 1 <= next_pull <= instance.budget + 1:
        raise ValueError("anytime state has an invalid next pull")
    rows = state.get("rows")
    expected_completed = [pull for pull in instance.grid if pull < next_pull]
    if not isinstance(rows, list) or [int(row["arm_pull"]) for row in rows] != expected_completed:
        raise ValueError("anytime state checkpoint rows mismatch")


def _checkpoint_state(
    *,
    instance: FrozenInstance,
    algorithm: str,
    digest: str,
    bandit,
    rows: list[dict[str, Any]],
    next_pull: int,
) -> dict[str, Any]:
    return {
        "state_schema": STATE_SCHEMA,
        "ensemble_config_hash": digest,
        "instance_hash": instance.instance_hash,
        "algorithm": algorithm,
        "budget": instance.budget,
        "grid": list(instance.grid),
        "next_pull": next_pull,
        "rows": rows,
        "bandit": bandit,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }


def run_anytime_task(
    instance: FrozenInstance, algorithm: str, run_dir: Path, config: dict[str, Any]
) -> Path:
    """Run/resume one full-budget anytime trajectory, preserving every checkpoint."""
    if algorithm not in ANYTIME_ALGORITHMS:
        raise ValueError(f"{algorithm} is not an anytime algorithm")
    digest = config_hash(config)
    csv_path, state_path = anytime_paths(run_dir, instance, algorithm)
    if anytime_is_complete(run_dir, instance, algorithm, digest):
        return csv_path

    environment = CountingEnvironment(instance.means, instance.stds)
    algorithm_seed = derive_seed(
        config["master_seed"], "heterogeneous-ensemble", instance.instance_hash, algorithm
    )
    if state_path.is_file():
        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        _validate_state(state, instance, algorithm, digest)
        bandit = state["bandit"]
        rows = state["rows"]
        next_pull = state["next_pull"]
        np.random.set_state(state["numpy_rng_state"])
        random.setstate(state["python_rng_state"])
    else:
        seed_rngs(config["master_seed"], "heterogeneous-ensemble", instance.instance_hash, algorithm)
        bandit = _construct_bandit(algorithm, environment)
        rows = []
        recommendation = list(bandit.get_top_arms())
        posterior_model = type(getattr(bandit, "posterior", None)).__name__
        if posterior_model == "NoneType":
            posterior_model = ""
        rows.append(
            _build_result_row(
                instance=instance,
                environment=environment,
                algorithm=algorithm,
                arm_pull=0,
                grid_index=0,
                recommendation=recommendation,
                posterior_model=posterior_model,
                rho=getattr(bandit, "p", getattr(bandit, "rho", None)),
                actual_reward_draws=0,
                algorithm_seed=algorithm_seed,
                digest=digest,
                master_seed=config["master_seed"],
            )
        )
        next_pull = 1
        _atomic_pickle(
            state_path,
            _checkpoint_state(
                instance=instance,
                algorithm=algorithm,
                digest=digest,
                bandit=bandit,
                rows=rows,
                next_pull=next_pull,
            ),
        )

    grid_index = {pull: index for index, pull in enumerate(instance.grid)}
    posterior_model = type(getattr(bandit, "posterior", None)).__name__
    if posterior_model == "NoneType":
        posterior_model = ""
    rho = getattr(bandit, "p", getattr(bandit, "rho", None))
    for arm_pull in range(next_pull, instance.budget + 1):
        arm = bandit.choose_arm()
        bandit.learn(arm, environment.pull_arm(arm))
        if arm_pull in grid_index:
            recommendation = list(bandit.get_top_arms())
            rows.append(
                _build_result_row(
                    instance=instance,
                    environment=environment,
                    algorithm=algorithm,
                    arm_pull=arm_pull,
                    grid_index=grid_index[arm_pull],
                    recommendation=recommendation,
                    posterior_model=posterior_model,
                    rho=rho,
                    actual_reward_draws=arm_pull,
                    algorithm_seed=algorithm_seed,
                    digest=digest,
                    master_seed=config["master_seed"],
                )
            )
            _atomic_pickle(
                state_path,
                _checkpoint_state(
                    instance=instance,
                    algorithm=algorithm,
                    digest=digest,
                    bandit=bandit,
                    rows=rows,
                    next_pull=arm_pull + 1,
                ),
            )
    _atomic_csv(csv_path, rows)
    state_path.unlink(missing_ok=True)
    return csv_path


def run_ege_budget_task(
    instance: FrozenInstance, algorithm: str, budget: int, run_dir: Path, config: dict[str, Any]
) -> Path:
    """Run one fixed-budget EGE task; no unconfigured budgets are ever executed."""
    if algorithm not in EGE_ALGORITHMS:
        raise ValueError(f"{algorithm} is not an EGE algorithm")
    digest = config_hash(config)
    if budget not in instance.grid:
        raise ValueError(f"{budget} is not in {instance.name}'s configured grid")
    if ege_is_complete(run_dir, instance, algorithm, budget, digest):
        return ege_path(run_dir, instance, algorithm, budget)
    environment = CountingEnvironment(instance.means, instance.stds)
    algorithm_seed = derive_seed(
        config["master_seed"], "heterogeneous-ensemble", instance.instance_hash, algorithm, "budget", budget
    )
    seed_rngs(
        config["master_seed"], "heterogeneous-ensemble", instance.instance_hash, algorithm, "budget", budget
    )
    recommendation = list(
        EGE_ALGORITHMS[algorithm](
            T=budget,
            K=instance.num_arms,
            D=instance.num_objectives,
            environment=environment,
        )
    )
    index = list(instance.grid).index(budget)
    row = _build_result_row(
        instance=instance,
        environment=environment,
        algorithm=algorithm,
        arm_pull=budget,
        grid_index=index,
        recommendation=recommendation,
        posterior_model="",
        rho=None,
        actual_reward_draws=environment.reward_draws,
        algorithm_seed=algorithm_seed,
        digest=digest,
        master_seed=config["master_seed"],
    )
    path = ege_path(run_dir, instance, algorithm, budget)
    _atomic_csv(path, [row])
    return path


def run_ege_instance_tasks(
    instance: FrozenInstance, algorithm: str, run_dir: Path, config: dict[str, Any]
) -> list[Path]:
    """Run every missing independent EGE cell for one instance/algorithm bundle."""
    digest = config_hash(config)
    return [
        run_ege_budget_task(instance, algorithm, budget, run_dir, config)
        for budget in instance.grid
        if not ege_is_complete(run_dir, instance, algorithm, budget, digest)
    ]


def _task_cost(instance: FrozenInstance, algorithm: str) -> tuple[float, int, str]:
    """Deterministic coarse ordering score; runtime probing later refines this."""
    if algorithm == "APE":
        return (float(instance.budget * instance.num_arms**2 * instance.num_objectives), instance.instance_id, algorithm)
    if algorithm == "TTPFTS_UNKNOWN_VAR":
        return (float(instance.budget * instance.num_arms * instance.num_objectives), instance.instance_id, algorithm)
    if algorithm in EGE_ALGORITHMS:
        return (float(instance.budget * 10.5), instance.instance_id, algorithm)
    return (float(instance.budget), instance.instance_id, algorithm)


def run(
    instances: list[FrozenInstance], run_dir: Path, config: dict[str, Any], n_jobs: int
) -> int:
    """Dynamically execute all pending tasks, longest expected work first."""
    digest = config_hash(config)
    work: list[tuple[FrozenInstance, str]] = []
    for instance in instances:
        for algorithm in ALGORITHMS:
            if algorithm in ANYTIME_ALGORITHMS:
                if not anytime_is_complete(run_dir, instance, algorithm, digest):
                    work.append((instance, algorithm))
            elif any(
                not ege_is_complete(run_dir, instance, algorithm, budget, digest)
                for budget in instance.grid
            ):
                work.append((instance, algorithm))
    work.sort(key=lambda task: _task_cost(*task), reverse=True)
    if not work:
        print("All requested heterogeneous-ensemble tasks are complete.")
        return 0
    print(f"Running {len(work)} instance/algorithm bundles with {n_jobs} worker(s).")

    def execute(instance: FrozenInstance, algorithm: str):
        if algorithm in ANYTIME_ALGORITHMS:
            return run_anytime_task(instance, algorithm, run_dir, config)
        return run_ege_instance_tasks(instance, algorithm, run_dir, config)

    if n_jobs == 1:
        for instance, algorithm in work:
            execute(instance, algorithm)
    else:
        Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(execute)(instance, algorithm) for instance, algorithm in work
        )
    return len(work)


def _parse_ids(spec: str | None) -> list[int] | None:
    return None if spec is None else [int(value) for value in spec.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--instance-ids", default=None, help="Comma-separated frozen instance IDs.")
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    if args.n_jobs < 1:
        raise ValueError("n-jobs must be positive")
    config = load_config()
    instances = load_frozen_instances(_parse_ids(args.instance_ids))
    run_dir = args.run_dir or default_output_root(config) / "run"
    run(instances, run_dir, config, args.n_jobs)


if __name__ == "__main__":
    main()
