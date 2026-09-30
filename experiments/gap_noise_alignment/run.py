#!/usr/bin/env python3
"""Execute paired custom noise/gap-alignment tasks with atomic resumable shards."""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
import tempfile
from pathlib import Path
from typing import Any

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_variable, "1")

import numpy as np
from joblib import Parallel, delayed

from bandits.ege_kone import EGE_SH, EGE_SR
from momab_main import _make_bandit
from runner.schema import SCHEMA_COLUMNS, build_row
from runner.seeding import derive_seed, seed_rngs

from .config import ALGORITHMS, CONDITIONS, config_hash, default_output_root, load_config
from .instances import FrozenCondition, PairedGaussianEnvironment, load_conditions


EGE_ALGORITHMS = {"EGE_SH": EGE_SH, "EGE_SR": EGE_SR}
ANYTIME_ALGORITHMS = ("Uniform", "APE", "TTPFTS_UNKNOWN_VAR")
STATE_SCHEMA = "gap-noise-alignment-anytime-state-v1"
TRACE_COLUMNS = [
    "condition", "instance_hash", "algorithm", "repetition", "arm_pull", "grid_index",
    "round", "active_count", "arm", "decision", "true_pareto", "decision_correct",
    "true_gap", "std", "noise_rank", "samples_at_decision", "empirical_g",
    "empirical_f", "empirical_delta", "empirical_pareto", "algorithm_seed",
    "reward_stream_seed", "gap_noise_config_hash",
]
RESULT_COLUMNS = [
    *SCHEMA_COLUMNS,
    "condition",
    "num_arms",
    "num_objectives",
    "pareto_size",
    "hardness",
    "grid_index",
    "normalized_budget",
    "execution_kind",
    "actual_reward_draws",
    "algorithm_seed",
    "reward_stream_seed",
    "noise_knowledge",
    "ape_sigma_bound",
    "warmup_pulls_per_arm",
    "gap_noise_config_hash",
]


def shard_dir(run_dir: Path, item: FrozenCondition, repetition: int, algorithm: str) -> Path:
    return run_dir / "shards" / item.condition / f"rep_{repetition:03d}" / algorithm


def anytime_paths(
    run_dir: Path, item: FrozenCondition, repetition: int, algorithm: str
) -> tuple[Path, Path]:
    root = shard_dir(run_dir, item, repetition, algorithm)
    return root / "trajectory.csv", root / "trajectory.state.pkl"


def ege_paths(
    run_dir: Path, item: FrozenCondition, repetition: int, algorithm: str, budget: int
) -> tuple[Path, Path]:
    root = shard_dir(run_dir, item, repetition, algorithm)
    return root / f"budget_{budget:06d}.csv", root / f"budget_{budget:06d}.trace.csv"


def _atomic_write(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-gap-noise-", suffix=".csv")
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


def _atomic_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-gap-noise-", suffix=".pkl")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
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


def _seeds(config: dict[str, Any], item: FrozenCondition, repetition: int, algorithm: str, budget: int | None) -> tuple[int, int]:
    """Pair both conditions by excluding condition from algorithm/reward seeds."""
    parts: tuple[object, ...] = ("gap-noise-alignment", "rep", repetition, algorithm)
    if budget is not None:
        parts += ("budget", budget)
    algorithm_seed = derive_seed(config["master_seed"], *parts, "algorithm")
    reward_seed = derive_seed(config["master_seed"], *parts, "reward")
    if algorithm_seed is None or reward_seed is None:
        raise RuntimeError("this experiment requires a concrete master seed")
    return algorithm_seed, reward_seed


def _environment(config: dict[str, Any], item: FrozenCondition, repetition: int, algorithm: str, budget: int | None) -> tuple[PairedGaussianEnvironment, int, int]:
    algorithm_seed, reward_seed = _seeds(config, item, repetition, algorithm, budget)
    return PairedGaussianEnvironment(item.means, item.stds, reward_seed), algorithm_seed, reward_seed


def _semantics(algorithm: str, environment: PairedGaussianEnvironment) -> dict[str, Any]:
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
        return {"execution_kind": "anytime", "noise_knowledge": "none", "ape_sigma_bound": "", "warmup_pulls_per_arm": 0}
    if algorithm in EGE_ALGORITHMS:
        return {"execution_kind": "ege", "noise_knowledge": "none", "ape_sigma_bound": "", "warmup_pulls_per_arm": 0}
    raise ValueError(f"unknown algorithm {algorithm}")


def _result_row(
    *,
    item: FrozenCondition,
    environment: PairedGaussianEnvironment,
    config: dict[str, Any],
    algorithm: str,
    repetition: int,
    arm_pull: int,
    grid_index: int,
    recommendation: list[int],
    posterior_model: str,
    rho: Any,
    algorithm_seed: int,
    reward_stream_seed: int,
) -> dict[str, Any]:
    base = build_row(
        environment=item.name,
        instance_hash=item.instance_hash,
        algorithm=algorithm,
        posterior_model=posterior_model,
        repetition=repetition,
        arm_pull=arm_pull,
        max_budget=item.budget,
        evaluation_grid=item.grid,
        rho=rho,
        master_seed=config["master_seed"],
        recommendation=recommendation,
        true_pareto=item.pareto_indices.tolist(),
        bernoulli=environment.bernoulli_metric(recommendation),
        jaccard=environment.jaccard_metric(recommendation),
        misidentification=environment.mis_id_metric(recommendation),
    )
    base.update(
        {
            "condition": item.condition,
            "num_arms": item.num_arms,
            "num_objectives": item.num_objectives,
            "pareto_size": len(item.pareto_indices),
            "hardness": item.hardness.hardness,
            "grid_index": grid_index,
            "normalized_budget": arm_pull / item.budget,
            "actual_reward_draws": environment.reward_draws,
            "algorithm_seed": algorithm_seed,
            "reward_stream_seed": reward_stream_seed,
            "gap_noise_config_hash": config_hash(config),
            **_semantics(algorithm, environment),
        }
    )
    return base


def _row_matches(
    row: dict[str, str], item: FrozenCondition, repetition: int, algorithm: str,
    arm_pull: int, grid_index: int, kind: str, digest: str,
) -> bool:
    try:
        return (
            set(row) == set(RESULT_COLUMNS)
            and row["environment"] == item.name
            and row["condition"] == item.condition
            and row["instance_hash"] == item.instance_hash
            and row["algorithm"] == algorithm
            and int(row["repetition"]) == repetition
            and int(row["arm_pull"]) == arm_pull
            and int(row["max_budget"]) == item.budget
            and int(row["grid_index"]) == grid_index
            and row["execution_kind"] == kind
            and row["gap_noise_config_hash"] == digest
        )
    except (KeyError, TypeError, ValueError):
        return False


def anytime_is_complete(
    run_dir: Path, item: FrozenCondition, repetition: int, algorithm: str, digest: str
) -> bool:
    csv_path, state_path = anytime_paths(run_dir, item, repetition, algorithm)
    if not csv_path.is_file() or state_path.exists():
        return False
    try:
        rows = _read_rows(csv_path)
        return (
            len(rows) == len(item.grid)
            and [int(row["arm_pull"]) for row in rows] == list(item.grid)
            and all(
                _row_matches(row, item, repetition, algorithm, budget, index, "anytime", digest)
                for index, (row, budget) in enumerate(zip(rows, item.grid))
            )
            and all(int(row["actual_reward_draws"]) == int(row["arm_pull"]) for row in rows)
        )
    except (OSError, ValueError, csv.Error):
        return False


def ege_is_complete(
    run_dir: Path, item: FrozenCondition, repetition: int, algorithm: str, budget: int, digest: str
) -> bool:
    if budget not in item.grid:
        return False
    result_path, trace_path = ege_paths(run_dir, item, repetition, algorithm, budget)
    if not result_path.is_file() or not trace_path.is_file():
        return False
    try:
        rows, traces = _read_rows(result_path), _read_rows(trace_path)
        index = list(item.grid).index(budget)
        return (
            len(rows) == 1
            and _row_matches(rows[0], item, repetition, algorithm, budget, index, "ege", digest)
            and len(traces) == item.num_arms
            and all(set(trace) == set(TRACE_COLUMNS) for trace in traces)
            and {int(trace["arm"]) for trace in traces} == set(range(item.num_arms))
        )
    except (OSError, ValueError, csv.Error):
        return False


def _bandit(algorithm: str, environment: PairedGaussianEnvironment):
    bandit = _make_bandit(algorithm, environment, p=0.5)
    if bandit is None:
        raise ValueError(f"could not construct {algorithm}")
    bandit.reset(environment.stds)
    return bandit


def _posterior_metadata(bandit) -> tuple[str, Any]:
    posterior = getattr(bandit, "posterior", None)
    name = "" if posterior is None else type(posterior).__name__
    return name, getattr(bandit, "p", getattr(bandit, "rho", None))


def _state(
    item: FrozenCondition, repetition: int, algorithm: str, config: dict[str, Any],
    bandit, environment: PairedGaussianEnvironment, rows: list[dict[str, Any]], next_pull: int,
) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "config_hash": config_hash(config),
        "condition": item.condition,
        "instance_hash": item.instance_hash,
        "repetition": repetition,
        "algorithm": algorithm,
        "budget": item.budget,
        "grid": list(item.grid),
        "next_pull": next_pull,
        "rows": rows,
        "bandit": bandit,
        "environment_stream_state": environment.export_stream_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }


def _validate_state(state: dict[str, Any], item: FrozenCondition, repetition: int, algorithm: str, config: dict[str, Any]) -> None:
    expected = {
        "schema": STATE_SCHEMA, "config_hash": config_hash(config), "condition": item.condition,
        "instance_hash": item.instance_hash, "repetition": repetition, "algorithm": algorithm,
        "budget": item.budget, "grid": list(item.grid),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"anytime state {key} mismatch")
    next_pull = state.get("next_pull")
    if not isinstance(next_pull, int) or not 1 <= next_pull <= item.budget + 1:
        raise ValueError("anytime state has invalid next_pull")
    if state["environment_stream_state"].get("reward_draws") != next_pull - 1:
        raise ValueError("anytime state reward-draw count mismatch")
    expected_pulls = [budget for budget in item.grid if budget < next_pull]
    if [int(row["arm_pull"]) for row in state.get("rows", [])] != expected_pulls:
        raise ValueError("anytime state checkpoint coverage mismatch")


def run_anytime_task(
    item: FrozenCondition, repetition: int, algorithm: str, run_dir: Path, config: dict[str, Any],
    *, stop_after_checkpoint: int | None = None,
) -> Path:
    """Run/resume exactly one paired anytime trajectory."""
    if algorithm not in ANYTIME_ALGORITHMS:
        raise ValueError(f"{algorithm} is not an anytime algorithm")
    digest = config_hash(config)
    csv_path, state_path = anytime_paths(run_dir, item, repetition, algorithm)
    if anytime_is_complete(run_dir, item, repetition, algorithm, digest):
        return csv_path
    environment, algorithm_seed, reward_seed = _environment(config, item, repetition, algorithm, None)
    if state_path.is_file():
        with state_path.open("rb") as handle:
            state = pickle.load(handle)
        _validate_state(state, item, repetition, algorithm, config)
        bandit, rows, next_pull = state["bandit"], state["rows"], state["next_pull"]
        environment.restore_stream_state(state["environment_stream_state"])
        np.random.set_state(state["numpy_rng_state"])
        random.setstate(state["python_rng_state"])
    else:
        seed_rngs(config["master_seed"], "gap-noise-alignment", "rep", repetition, algorithm, "algorithm")
        bandit = _bandit(algorithm, environment)
        posterior, rho = _posterior_metadata(bandit)
        rows = [_result_row(
            item=item, environment=environment, config=config, algorithm=algorithm, repetition=repetition,
            arm_pull=0, grid_index=0, recommendation=list(bandit.get_top_arms()), posterior_model=posterior,
            rho=rho, algorithm_seed=algorithm_seed, reward_stream_seed=reward_seed,
        )]
        next_pull = 1
        _atomic_pickle(state_path, _state(item, repetition, algorithm, config, bandit, environment, rows, next_pull))
        if stop_after_checkpoint == 0:
            raise RuntimeError("controlled checkpoint interruption")
    posterior, rho = _posterior_metadata(bandit)
    grid_indices = {budget: index for index, budget in enumerate(item.grid)}
    for arm_pull in range(next_pull, item.budget + 1):
        arm = bandit.choose_arm()
        bandit.learn(arm, environment.pull_arm(arm))
        if arm_pull in grid_indices:
            rows.append(_result_row(
                item=item, environment=environment, config=config, algorithm=algorithm, repetition=repetition,
                arm_pull=arm_pull, grid_index=grid_indices[arm_pull], recommendation=list(bandit.get_top_arms()),
                posterior_model=posterior, rho=rho, algorithm_seed=algorithm_seed, reward_stream_seed=reward_seed,
            ))
            _atomic_pickle(
                state_path,
                _state(item, repetition, algorithm, config, bandit, environment, rows, arm_pull + 1),
            )
            if stop_after_checkpoint == arm_pull:
                raise RuntimeError("controlled checkpoint interruption")
    _atomic_write(csv_path, RESULT_COLUMNS, rows)
    state_path.unlink(missing_ok=True)
    return csv_path


def _trace_rows(
    trace: list[dict[str, Any]], item: FrozenCondition, repetition: int, algorithm: str, budget: int,
    grid_index: int, algorithm_seed: int, reward_seed: int, digest: str,
) -> list[dict[str, Any]]:
    pareto = set(item.pareto_indices.tolist())
    rows = []
    for event in trace:
        arm, decision = int(event["arm"]), str(event["decision"])
        accept = decision.startswith("accept")
        rows.append(
            {
                "condition": item.condition, "instance_hash": item.instance_hash, "algorithm": algorithm,
                "repetition": repetition, "arm_pull": budget, "grid_index": grid_index,
                "round": event["round"], "active_count": event["active_count"], "arm": arm,
                "decision": decision, "true_pareto": int(arm in pareto),
                "decision_correct": int((accept and arm in pareto) or (not accept and arm not in pareto)),
                "true_gap": item.gaps[arm], "std": item.arm_stds[arm], "noise_rank": item.noise_ranks[arm],
                "samples_at_decision": event["samples_at_decision"], "empirical_g": event["empirical_g"],
                "empirical_f": event["empirical_f"], "empirical_delta": event["empirical_delta"],
                "empirical_pareto": event["empirical_pareto"], "algorithm_seed": algorithm_seed,
                "reward_stream_seed": reward_seed, "gap_noise_config_hash": digest,
            }
        )
    return rows


def run_ege_budget_task(
    item: FrozenCondition, repetition: int, algorithm: str, budget: int, run_dir: Path, config: dict[str, Any]
) -> Path:
    """Run one EGE grid point with a full classification trace."""
    if algorithm not in EGE_ALGORITHMS or budget not in item.grid:
        raise ValueError("invalid EGE task")
    digest = config_hash(config)
    if ege_is_complete(run_dir, item, repetition, algorithm, budget, digest):
        return ege_paths(run_dir, item, repetition, algorithm, budget)[0]
    environment, algorithm_seed, reward_seed = _environment(config, item, repetition, algorithm, budget)
    seed_rngs(config["master_seed"], "gap-noise-alignment", "rep", repetition, algorithm, "budget", budget, "algorithm")
    trace: list[dict[str, Any]] = []
    recommendation = list(EGE_ALGORITHMS[algorithm](
        T=budget, K=item.num_arms, D=item.num_objectives, environment=environment, trace=trace
    ))
    if len(trace) != item.num_arms:
        raise RuntimeError(f"{algorithm} produced {len(trace)} trace events, expected {item.num_arms}")
    index = list(item.grid).index(budget)
    result = _result_row(
        item=item, environment=environment, config=config, algorithm=algorithm, repetition=repetition,
        arm_pull=budget, grid_index=index, recommendation=recommendation, posterior_model="", rho=None,
        algorithm_seed=algorithm_seed, reward_stream_seed=reward_seed,
    )
    result_path, trace_path = ege_paths(run_dir, item, repetition, algorithm, budget)
    _atomic_write(result_path, RESULT_COLUMNS, [result])
    _atomic_write(
        trace_path, TRACE_COLUMNS,
        _trace_rows(trace, item, repetition, algorithm, budget, index, algorithm_seed, reward_seed, digest),
    )
    return result_path


def _cost(item: FrozenCondition, repetition: int, algorithm: str) -> tuple[float, int, str]:
    if algorithm == "APE":
        score = item.budget * item.num_arms**2 * item.num_objectives
    elif algorithm == "TTPFTS_UNKNOWN_VAR":
        score = item.budget * item.num_arms * item.num_objectives
    elif algorithm in EGE_ALGORITHMS:
        score = item.budget * 10.5
    else:
        score = item.budget
    return float(score), repetition, algorithm


def run(
    conditions: list[FrozenCondition],
    run_dir: Path,
    config: dict[str, Any],
    n_jobs: int,
    repetitions: list[int] | None = None,
) -> int:
    """Dynamically execute missing paired condition/repetition/algorithm bundles."""
    digest = config_hash(config)
    work = []
    selected_repetitions = list(range(int(config["repetitions"]))) if repetitions is None else list(repetitions)
    if not selected_repetitions or min(selected_repetitions) < 0 or max(selected_repetitions) >= int(config["repetitions"]):
        raise ValueError("selected repetitions fall outside the frozen range")
    for item in conditions:
        for repetition in selected_repetitions:
            for algorithm in ALGORITHMS:
                complete = (
                    anytime_is_complete(run_dir, item, repetition, algorithm, digest)
                    if algorithm in ANYTIME_ALGORITHMS
                    else all(ege_is_complete(run_dir, item, repetition, algorithm, budget, digest) for budget in item.grid)
                )
                if not complete:
                    work.append((item, repetition, algorithm))
    work.sort(key=lambda task: _cost(*task), reverse=True)
    if not work:
        print("All gap-noise alignment tasks are complete.")
        return 0

    def execute(item: FrozenCondition, repetition: int, algorithm: str):
        if algorithm in ANYTIME_ALGORITHMS:
            return run_anytime_task(item, repetition, algorithm, run_dir, config)
        return [
            run_ege_budget_task(item, repetition, algorithm, budget, run_dir, config)
            for budget in item.grid
            if not ege_is_complete(run_dir, item, repetition, algorithm, budget, digest)
        ]

    if n_jobs == 1:
        for task in work:
            execute(*task)
    else:
        Parallel(n_jobs=n_jobs, prefer="processes")(delayed(execute)(*task) for task in work)
    return len(work)


def _parse_repetitions(spec: str | None) -> list[int] | None:
    return None if spec is None else [int(value) for value in spec.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--repetitions", default=None, help="Comma-separated repetition subset for testing/recovery.")
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    config = load_config()
    run(
        load_conditions(), args.run_dir or default_output_root(config) / "run", config, args.n_jobs,
        repetitions=_parse_repetitions(args.repetitions),
    )


if __name__ == "__main__":
    main()
