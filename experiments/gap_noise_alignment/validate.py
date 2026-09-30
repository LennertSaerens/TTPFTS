#!/usr/bin/env python3
"""Independently validate gap-noise metric shards, EGE traces, and provenance."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

from runner.ege_allocation import expected_ege_draws
from runner.schema import decode_int_set, encode_int_set, validate_row_metrics
from runner.seeding import derive_seed

from .config import ALGORITHMS, CONDITIONS, config_hash, default_output_root, load_config, sha256_file
from .instances import MANIFEST_PATH, FrozenCondition, load_conditions
from .merge import RESULT_FILE, RUN_MANIFEST_FILE, TRACE_FILE
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


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _add(problems: list[str], message: str) -> None:
    problems.append(message)


def _seed(config: dict, repetition: int, algorithm: str, budget: int | None, kind: str) -> int:
    parts: tuple[object, ...] = ("gap-noise-alignment", "rep", repetition, algorithm)
    if budget is not None:
        parts += ("budget", budget)
    value = derive_seed(config["master_seed"], *parts, kind)
    if value is None:
        raise RuntimeError("validation requires configured seeds")
    return value


def _expected_noise(algorithm: str, item: FrozenCondition) -> tuple[str, str, str, int]:
    if algorithm == "APE":
        return "", "global_max_standard_deviation_bound", str(float(item.arm_stds.max())), 0
    if algorithm == "TTPFTS_UNKNOWN_VAR":
        return "TPosterior", "unknown_variance_tposterior", "", 4
    if algorithm == "Uniform":
        return "", "none", "", 0
    if algorithm in EGE_ALGORITHMS:
        return "", "none", "", 0
    raise ValueError(algorithm)


def _validate_metric_row(
    row: dict[str, str], item: FrozenCondition, repetition: int, algorithm: str,
    budget: int, grid_index: int, config: dict, problems: list[str],
) -> None:
    label = f"{item.condition}/rep={repetition}/{algorithm}/t={budget}"
    digest, kind = config_hash(config), "anytime" if algorithm in ANYTIME_ALGORITHMS else "ege"
    if set(row) != set(RESULT_COLUMNS):
        _add(problems, f"{label}: result schema mismatch")
        return
    expected = {
        "environment": item.condition, "condition": item.condition, "instance_hash": item.instance_hash,
        "algorithm": algorithm, "repetition": str(repetition), "arm_pull": str(budget),
        "max_budget": str(item.budget), "evaluation_grid": encode_int_set(item.grid),
        "master_seed": str(config["master_seed"]), "num_arms": "50", "num_objectives": "2",
        "pareto_size": str(len(item.pareto_indices)), "grid_index": str(grid_index),
        "execution_kind": kind, "gap_noise_config_hash": digest,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            _add(problems, f"{label}: {key}={row.get(key)!r}, expected {value!r}")
    posterior, knowledge, ape_bound, warmup = _expected_noise(algorithm, item)
    if (
        row.get("posterior_model") != posterior
        or row.get("noise_knowledge") != knowledge
        or row.get("ape_sigma_bound") != ape_bound
        or row.get("warmup_pulls_per_arm") != str(warmup)
    ):
        _add(problems, f"{label}: noise/posterior semantics mismatch")
    if (algorithm == "TTPFTS_UNKNOWN_VAR" and row.get("rho") != "0.5") or (
        algorithm != "TTPFTS_UNKNOWN_VAR" and row.get("rho") != ""
    ):
        _add(problems, f"{label}: rho semantics mismatch")
    seed_budget = None if algorithm in ANYTIME_ALGORITHMS else budget
    if row.get("algorithm_seed") != str(_seed(config, repetition, algorithm, seed_budget, "algorithm")):
        _add(problems, f"{label}: algorithm seed mismatch")
    if row.get("reward_stream_seed") != str(_seed(config, repetition, algorithm, seed_budget, "reward")):
        _add(problems, f"{label}: reward-stream seed mismatch")
    try:
        if not math.isclose(float(row["hardness"]), item.hardness.hardness, rel_tol=0.0, abs_tol=1e-10):
            _add(problems, f"{label}: hardness mismatch")
        if not math.isclose(float(row["normalized_budget"]), budget / item.budget, rel_tol=0.0, abs_tol=1e-12):
            _add(problems, f"{label}: normalized budget mismatch")
        recommendation, pareto = decode_int_set(row["recommendation"]), decode_int_set(row["true_pareto"])
        if not recommendation <= set(range(item.num_arms)) or pareto != set(item.pareto_indices.tolist()):
            _add(problems, f"{label}: invalid recommendation or true Pareto set")
        for issue in validate_row_metrics(row, item.num_arms, atol=1e-12):
            _add(problems, f"{label}: {issue}")
        expected_draws = budget if algorithm in ANYTIME_ALGORITHMS else expected_ege_draws(
            algorithm, budget, item.num_arms
        )
        if int(row["actual_reward_draws"]) != expected_draws:
            _add(problems, f"{label}: actual reward draws mismatch")
    except (KeyError, ValueError, TypeError) as exc:
        _add(problems, f"{label}: malformed result value ({exc})")


def _validate_traces(
    trace_rows: dict[tuple[str, int, str, int], list[dict[str, str]]],
    metric_rows: dict[tuple[str, int, str, int], dict[str, str]],
    items: dict[str, FrozenCondition], repetitions: list[int], config: dict, problems: list[str],
) -> None:
    digest = config_hash(config)
    for condition, item in items.items():
        pareto = set(item.pareto_indices.tolist())
        for repetition in repetitions:
            for algorithm in EGE_ALGORITHMS:
                for grid_index, budget in enumerate(item.grid):
                    label = f"{condition}/rep={repetition}/{algorithm}/t={budget}"
                    traces = trace_rows.get((condition, repetition, algorithm, grid_index), [])
                    result = metric_rows.get((condition, repetition, algorithm, grid_index))
                    if len(traces) != item.num_arms:
                        _add(problems, f"{label}: trace rows={len(traces)}, expected {item.num_arms}")
                        continue
                    if {int(trace["arm"]) for trace in traces} != set(range(item.num_arms)):
                        _add(problems, f"{label}: trace arms do not cover 0..49")
                    for trace in traces:
                        try:
                            arm, decision = int(trace["arm"]), trace["decision"]
                            expected_correct = int(
                                (decision.startswith("accept") and arm in pareto)
                                or (not decision.startswith("accept") and arm not in pareto)
                            )
                            if (
                                set(trace) != set(TRACE_COLUMNS)
                                or trace["condition"] != condition
                                or trace["instance_hash"] != item.instance_hash
                                or trace["algorithm"] != algorithm
                                or int(trace["repetition"]) != repetition
                                or int(trace["arm_pull"]) != budget
                                or int(trace["grid_index"]) != grid_index
                                or int(trace["true_pareto"]) != int(arm in pareto)
                                or int(trace["decision_correct"]) != expected_correct
                                or not math.isclose(float(trace["true_gap"]), item.gaps[arm], abs_tol=1e-12)
                                or not math.isclose(float(trace["std"]), item.arm_stds[arm], abs_tol=1e-12)
                                or int(trace["noise_rank"]) != int(item.noise_ranks[arm])
                                or trace["gap_noise_config_hash"] != digest
                                or int(trace["algorithm_seed"]) != _seed(config, repetition, algorithm, budget, "algorithm")
                                or int(trace["reward_stream_seed"]) != _seed(config, repetition, algorithm, budget, "reward")
                                or int(trace["samples_at_decision"]) < 0
                                or (result is not None and int(trace["samples_at_decision"]) > int(result["actual_reward_draws"]))
                            ):
                                _add(problems, f"{label}: invalid trace for arm {arm}")
                        except (KeyError, ValueError, TypeError) as exc:
                            _add(problems, f"{label}: malformed trace ({exc})")


def _expected_paths(run_dir: Path, items: Iterable[FrozenCondition], repetitions: Iterable[int]) -> set[Path]:
    paths = set()
    for item in items:
        for repetition in repetitions:
            for algorithm in ALGORITHMS:
                if algorithm in ANYTIME_ALGORITHMS:
                    paths.add(anytime_paths(run_dir, item, repetition, algorithm)[0])
                else:
                    for budget in item.grid:
                        paths.update(ege_paths(run_dir, item, repetition, algorithm, budget))
    return paths


def validate(
    run_dir: Path,
    results_dir: Path,
    *,
    conditions: Iterable[FrozenCondition] | None = None,
    repetitions: Iterable[int] | None = None,
    require_manifest: bool = True,
) -> list[str]:
    """Return every independent validation failure; an empty list is a pass."""
    config = load_config()
    selected_conditions = list(load_conditions() if conditions is None else conditions)
    selected_repetitions = list(range(config["repetitions"]) if repetitions is None else repetitions)
    problems: list[str] = []
    result_path, trace_path = results_dir / RESULT_FILE, results_dir / TRACE_FILE
    if not result_path.is_file() or not trace_path.is_file():
        return [f"missing merged result/trace file in {results_dir}"]
    rows, traces = _read_rows(result_path), _read_rows(trace_path)
    items = {item.condition: item for item in selected_conditions}
    expected_metric_keys = {
        (item.condition, repetition, algorithm, grid_index)
        for item in selected_conditions for repetition in selected_repetitions
        for algorithm in ALGORITHMS for grid_index in range(len(item.grid))
    }
    observed: dict[tuple[str, int, str, int], dict[str, str]] = {}
    for row in rows:
        try:
            key = (row["condition"], int(row["repetition"]), row["algorithm"], int(row["grid_index"]))
        except (KeyError, ValueError):
            _add(problems, "unrecognized metric identity")
            continue
        if key in observed:
            _add(problems, f"duplicate metric row {key}")
        observed[key] = row
    if set(observed) != expected_metric_keys:
        _add(problems, f"metric coverage mismatch missing={len(expected_metric_keys - set(observed))} extra={len(set(observed) - expected_metric_keys)}")
    for item in selected_conditions:
        for repetition in selected_repetitions:
            for algorithm in ALGORITHMS:
                if algorithm in ANYTIME_ALGORITHMS:
                    if not anytime_is_complete(run_dir, item, repetition, algorithm, config_hash(config)):
                        _add(problems, f"{item.condition}/rep={repetition}/{algorithm}: incomplete anytime shard")
                else:
                    for budget in item.grid:
                        if not ege_is_complete(run_dir, item, repetition, algorithm, budget, config_hash(config)):
                            _add(problems, f"{item.condition}/rep={repetition}/{algorithm}/t={budget}: incomplete EGE shard")
                for grid_index, budget in enumerate(item.grid):
                    row = observed.get((item.condition, repetition, algorithm, grid_index))
                    if row is not None:
                        _validate_metric_row(row, item, repetition, algorithm, budget, grid_index, config, problems)
    trace_index: dict[tuple[str, int, str, int], list[dict[str, str]]] = {}
    for trace in traces:
        try:
            key = (trace["condition"], int(trace["repetition"]), trace["algorithm"], int(trace["grid_index"]))
            trace_index.setdefault(key, []).append(trace)
        except (KeyError, ValueError):
            _add(problems, "unrecognized trace identity")
    _validate_traces(trace_index, observed, items, selected_repetitions, config, problems)
    paired = {}
    for row in rows:
        key = (row.get("algorithm"), row.get("repetition"), row.get("grid_index"))
        paired.setdefault(key, set()).add((row.get("algorithm_seed"), row.get("reward_stream_seed")))
    if any(len(seeds) != 1 for seeds in paired.values()):
        _add(problems, "condition-paired rows do not share their prescribed seeds")
    expected_paths = _expected_paths(run_dir, selected_conditions, selected_repetitions)
    shard_root = run_dir / "shards"
    actual_paths = set(shard_root.rglob("*.csv")) if shard_root.is_dir() else set()
    if actual_paths - expected_paths:
        _add(problems, f"unexpected shard CSVs, first={sorted(map(str, actual_paths - expected_paths))[:3]}")
    if expected_paths - actual_paths:
        _add(problems, f"missing shard CSVs, first={sorted(map(str, expected_paths - actual_paths))[:3]}")
    state_paths = list(shard_root.rglob("*.state.pkl")) if shard_root.is_dir() else []
    if state_paths:
        _add(problems, f"unfinished anytime states, first={sorted(map(str, state_paths))[:3]}")
    if require_manifest:
        manifest_path = run_dir / RUN_MANIFEST_FILE
        if not manifest_path.is_file():
            _add(problems, f"missing run manifest {manifest_path}")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("config_hash") != config_hash(config)
                or manifest.get("frozen_condition_manifest_sha256") != sha256_file(MANIFEST_PATH)
                or manifest.get("result_file", {}).get("sha256") != sha256_file(result_path)
                or manifest.get("trace_file", {}).get("sha256") != sha256_file(trace_path)
                or manifest.get("expected_metric_rows") != len(expected_metric_keys)
                or manifest.get("expected_trace_rows") != len(selected_conditions) * len(selected_repetitions) * len(EGE_ALGORITHMS) * 21 * 50
            ):
                _add(problems, "run manifest mismatch")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    args = parser.parse_args()
    config = load_config()
    root = default_output_root(config)
    problems = validate(args.run_dir or root / "run", args.results_dir or root / "results")
    if problems:
        print("GAP-NOISE ALIGNMENT VALIDATION FAILED")
        for problem in problems[:100]:
            print(f"- {problem}")
        return 1
    print("GAP-NOISE ALIGNMENT VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
