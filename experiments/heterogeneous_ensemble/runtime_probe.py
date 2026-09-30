#!/usr/bin/env python3
"""Measure stratified ensemble-task costs and project conservative local work."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bandits.ege_kone import EGE_SH, EGE_SR
from momab_main import _make_bandit
from runner.seeding import seed_rngs

from .config import ALGORITHMS, config_hash, load_config
from .instances import FrozenInstance, load_frozen_instances
from .run import CountingEnvironment
from .validate import expected_ege_draws


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
EGE_FUNCTIONS = {"EGE_SH": EGE_SH, "EGE_SR": EGE_SR}


def _score(instance: FrozenInstance, algorithm: str) -> float:
    if algorithm == "APE":
        return float(instance.budget * instance.num_arms**2 * instance.num_objectives)
    if algorithm == "TTPFTS_UNKNOWN_VAR":
        return float(instance.budget * instance.num_arms * instance.num_objectives)
    if algorithm in EGE_FUNCTIONS:
        return float(sum(expected_ege_draws(algorithm, budget, instance.num_arms) for budget in instance.grid))
    return float(instance.budget)


def select_probe_instances(instances: list[FrozenInstance]) -> list[tuple[FrozenInstance, str]]:
    """Select deterministic cost/shape strata and reserve every fourth for holdout."""
    selected: dict[int, FrozenInstance] = {}
    quantiles = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
    for algorithm in ALGORITHMS:
        values = np.asarray([_score(instance, algorithm) for instance in instances], dtype=float)
        for quantile in quantiles:
            target = float(np.quantile(values, quantile))
            chosen = min(instances, key=lambda item: abs(_score(item, algorithm) - target))
            selected[chosen.instance_id] = chosen
    ordered = [selected[key] for key in sorted(selected)]
    if len(ordered) < 8:
        raise ValueError("runtime probe selection is too small for calibration/holdout")
    return [(instance, "validation" if index % 4 == 0 else "calibration") for index, instance in enumerate(ordered)]


def _warm_dependencies(instance: FrozenInstance) -> None:
    """Run one unreported minimal operation per algorithm before timing."""
    for algorithm in ("Uniform", "APE", "TTPFTS_UNKNOWN_VAR"):
        environment = CountingEnvironment(instance.means, instance.stds)
        seed_rngs(20261001, "heterogeneous-ensemble-runtime-warmup", algorithm)
        bandit = _make_bandit(algorithm, environment, p=0.5)
        if bandit is None:
            raise ValueError(f"could not construct {algorithm}")
        bandit.reset(environment.stds)
        arm = bandit.choose_arm()
        bandit.learn(arm, environment.pull_arm(arm))
    for algorithm, function in EGE_FUNCTIONS.items():
        environment = CountingEnvironment(instance.means, instance.stds)
        seed_rngs(20261001, "heterogeneous-ensemble-runtime-warmup", algorithm)
        function(T=max(1, instance.num_arms), K=instance.num_arms, D=instance.num_objectives, environment=environment)


def _time_anytime(instance: FrozenInstance, algorithm: str, steady_pulls: int) -> dict[str, Any]:
    environment = CountingEnvironment(instance.means, instance.stds)
    seed_rngs(20261001, "heterogeneous-ensemble-runtime-probe", instance.instance_hash, algorithm)
    bandit = _make_bandit(algorithm, environment, p=0.5)
    if bandit is None:
        raise ValueError(f"could not construct {algorithm}")
    bandit.reset(environment.stds)
    warmup = 4 * instance.num_arms if algorithm == "TTPFTS_UNKNOWN_VAR" else (
        instance.num_arms if algorithm == "APE" else 0
    )
    for _ in range(min(warmup, instance.budget)):
        arm = bandit.choose_arm()
        bandit.learn(arm, environment.pull_arm(arm))
    measured = min(steady_pulls, max(0, instance.budget - environment.reward_draws))
    start = time.perf_counter()
    for _ in range(measured):
        arm = bandit.choose_arm()
        bandit.learn(arm, environment.pull_arm(arm))
    elapsed = time.perf_counter() - start
    if measured < 1:
        raise ValueError(f"{instance.name}/{algorithm}: insufficient post-warmup pulls for timing")
    return {
        "algorithm": algorithm,
        "kind": "anytime",
        "probe_budget": environment.reward_draws,
        "probe_actual_draws": measured,
        "warmup_pulls": min(warmup, instance.budget),
        "elapsed_seconds": elapsed,
        "seconds_per_draw": elapsed / measured,
    }


def _time_ege(instance: FrozenInstance, algorithm: str, max_budget: int) -> dict[str, Any]:
    budget = min(instance.budget, max_budget)
    environment = CountingEnvironment(instance.means, instance.stds)
    seed_rngs(20261001, "heterogeneous-ensemble-runtime-probe", instance.instance_hash, algorithm, budget)
    start = time.perf_counter()
    EGE_FUNCTIONS[algorithm](
        T=budget, K=instance.num_arms, D=instance.num_objectives, environment=environment
    )
    elapsed = time.perf_counter() - start
    draws = environment.reward_draws
    return {
        "algorithm": algorithm,
        "kind": "ege",
        "probe_budget": budget,
        "probe_actual_draws": draws,
        "warmup_pulls": 0,
        "elapsed_seconds": elapsed,
        "seconds_per_draw": elapsed / max(1, draws),
    }


def run_probe(steady_anytime_pulls: int = 1000, max_ege_budget: int = 50_000) -> pd.DataFrame:
    """Benchmark every selected stratum/algorithm after one-time dependency warm-up."""
    config = load_config()
    instances = load_frozen_instances()
    selected = select_probe_instances(instances)
    _warm_dependencies(selected[0][0])
    rows = []
    for instance, split in selected:
        common = {
            "instance_id": instance.instance_id,
            "proposal_id": instance.proposal_id,
            "split": split,
            "num_arms": instance.num_arms,
            "num_objectives": instance.num_objectives,
            "hardness": instance.hardness.hardness,
            "budget": instance.budget,
        }
        for algorithm in ("Uniform", "APE", "TTPFTS_UNKNOWN_VAR"):
            rows.append({**common, **_time_anytime(instance, algorithm, steady_anytime_pulls)})
        for algorithm in EGE_FUNCTIONS:
            rows.append({**common, **_time_ege(instance, algorithm, max_ege_budget)})
    table = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "runtime_probe.csv", index=False)
    return table


def _feature(row: pd.Series) -> float:
    if row["algorithm"] == "APE":
        return float(row["probe_actual_draws"] * row["num_arms"] ** 2 * row["num_objectives"])
    if row["algorithm"] == "TTPFTS_UNKNOWN_VAR":
        return float(row["probe_actual_draws"] * row["num_arms"] * row["num_objectives"])
    return float(row["probe_actual_draws"])


def _instance_feature(instance: FrozenInstance, algorithm: str, budget: int) -> float:
    if algorithm == "APE":
        return float(budget * instance.num_arms**2 * instance.num_objectives)
    if algorithm == "TTPFTS_UNKNOWN_VAR":
        return float(budget * instance.num_arms * instance.num_objectives)
    if algorithm == "Uniform":
        return float(budget)
    if algorithm not in EGE_FUNCTIONS:
        raise ValueError(f"unsupported algorithm {algorithm}")
    return float(expected_ege_draws(algorithm, budget, instance.num_arms))


def fit_projection(table: pd.DataFrame, instances: list[FrozenInstance]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit a held-out calibrated upper-envelope model and project full cohort work."""
    records, diagnostics = [], []
    for algorithm in ALGORITHMS:
        rows = table[table["algorithm"] == algorithm].copy()
        calibration = rows[rows["split"] == "calibration"].copy()
        validation = rows[rows["split"] == "validation"].copy()
        if calibration.empty or validation.empty:
            raise ValueError(f"{algorithm}: missing calibration or validation timing rows")
        calibration_feature = calibration.apply(_feature, axis=1)
        coefficient = float((calibration["elapsed_seconds"] / calibration_feature).max())
        validation_predictions = validation.apply(_feature, axis=1) * coefficient
        ratios = validation["elapsed_seconds"].to_numpy() / validation_predictions.to_numpy()
        safety_factor = max(1.25, float(np.max(ratios)) * 1.25)
        conservative_coefficient = coefficient * safety_factor
        for _, row in validation.iterrows():
            diagnostics.append(
                {
                    "algorithm": algorithm,
                    "instance_id": int(row["instance_id"]),
                    "actual_seconds": float(row["elapsed_seconds"]),
                    "calibration_prediction_seconds": float(_feature(row) * coefficient),
                    "actual_over_prediction": float(row["elapsed_seconds"] / (_feature(row) * coefficient)),
                }
            )
        if algorithm in EGE_FUNCTIONS:
            task_overhead = float(calibration["elapsed_seconds"].max()) * safety_factor
            task_count = 21 * len(instances)
            feature_total = sum(
                _instance_feature(instance, algorithm, budget)
                for instance in instances
                for budget in instance.grid
            )
        else:
            task_overhead = float(calibration["elapsed_seconds"].max()) * safety_factor
            task_count = len(instances)
            feature_total = sum(_instance_feature(instance, algorithm, instance.budget) for instance in instances)
        projected_seconds = conservative_coefficient * feature_total + task_overhead * task_count
        records.append(
            {
                "algorithm": algorithm,
                "calibration_coefficient": coefficient,
                "safety_factor": safety_factor,
                "conservative_coefficient": conservative_coefficient,
                "per_task_overhead_seconds": task_overhead,
                "task_count": task_count,
                "feature_total": feature_total,
                "projected_seconds": projected_seconds,
                "projected_core_hours": projected_seconds / 3600.0,
            }
        )
    projection = pd.DataFrame(records)
    diagnostics_frame = pd.DataFrame(diagnostics)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    projection.to_csv(REPORTS_DIR / "runtime_projection.csv", index=False)
    diagnostics_frame.to_csv(REPORTS_DIR / "runtime_validation.csv", index=False)
    total_seconds = float(projection["projected_seconds"].sum())
    summary = {
        "schema_version": "heterogeneous-ensemble-runtime-projection-v1",
        "ensemble_config_hash": config_hash(load_config()),
        "calibration_method": (
            "Algorithm-specific linear operation-feature envelope fit on deterministic "
            "calibration strata, inflated by 1.25 times the largest held-out underprediction."
        ),
        "probe_instances": int(table["instance_id"].nunique()),
        "validation_instances": int(table.loc[table["split"] == "validation", "instance_id"].nunique()),
        "projected_total_core_hours": total_seconds / 3600.0,
        "projected_balanced_eight_worker_hours": total_seconds / 3600.0 / 8.0,
        "projection_is_conservative_not_a_runtime_guarantee": True,
    }
    (REPORTS_DIR / "runtime_projection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return projection, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steady-anytime-pulls", type=int, default=1000)
    parser.add_argument("--max-ege-budget", type=int, default=50_000)
    args = parser.parse_args()
    if args.steady_anytime_pulls < 1 or args.max_ege_budget < 1:
        parser.error("probe budgets must be positive")
    table = run_probe(args.steady_anytime_pulls, args.max_ege_budget)
    instances = load_frozen_instances()
    projection, summary = fit_projection(table, instances)
    print(projection.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
