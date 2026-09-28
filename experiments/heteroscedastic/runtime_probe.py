#!/usr/bin/env python3
"""Measure warm-up-aware local runtime for frozen heteroscedastic candidates.

This is a feasibility instrument, not an experiment result.  It never writes
performance metrics and never executes a trajectory to its full hardness
budget.  EGE timings at a capped probe budget are extrapolated explicitly and
labelled as such because production EGE reruns from scratch at every grid
budget.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bandits.ege_kone import EGE_SH, EGE_SR
from experiments.heteroscedastic.instances import FrozenHeteroscedasticEnvironment, load_frozen_candidates
from momab_main import _make_bandit
from runner.seeding import seed_rngs


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
ANYTIME_ALGORITHMS = ("Uniform", "APE", "TTPFTS_UNKNOWN_VAR")
EGE_ALGORITHMS = {"EGE_SH": EGE_SH, "EGE_SR": EGE_SR}


def _representatives(candidates) -> list:
    """One deterministic heterogeneous candidate nearest each hardness quantile."""
    hetero = sorted((candidate for candidate in candidates if candidate.condition == "heterogeneous"),
                    key=lambda candidate: candidate.hardness.hardness)
    values = np.asarray([candidate.hardness.hardness for candidate in hetero])
    selected = []
    for quantile in (0.0, 0.25, 0.5, 0.75, 0.9):
        target = np.quantile(values, quantile)
        selected.append(min(hetero, key=lambda candidate: abs(candidate.hardness.hardness - target)))
    return list({candidate.instance_id: candidate for candidate in selected}.values())


def _time_anytime(candidate, algorithm: str, measured_pulls: int) -> dict:
    env = FrozenHeteroscedasticEnvironment(candidate.means, candidate.stds)
    seed_rngs(20260928, "heteroscedastic-runtime-probe", candidate.instance_id, algorithm)
    bandit = _make_bandit(algorithm, env, 0.5)
    if bandit is None:
        raise ValueError(f"could not construct {algorithm}")
    bandit.reset(env.stds)
    warmup_pulls = (
        4 * candidate.means.shape[0] if algorithm == "TTPFTS_UNKNOWN_VAR"
        else candidate.means.shape[0] if algorithm == "APE"
        else 0
    )
    warmup_start = time.perf_counter()
    for _ in range(warmup_pulls):
        arm = bandit.choose_arm()
        bandit.learn(arm, env.pull_arm(arm))
    warmup_seconds = time.perf_counter() - warmup_start
    start = time.perf_counter()
    for _ in range(measured_pulls):
        arm = bandit.choose_arm()
        bandit.learn(arm, env.pull_arm(arm))
    elapsed = time.perf_counter() - start
    return {
        "algorithm": algorithm,
        "kind": "anytime",
        "probe_budget": warmup_pulls + measured_pulls,
        "warmup_pulls": warmup_pulls,
        "warmup_seconds": warmup_seconds,
        "timed_steady_pulls": measured_pulls,
        "elapsed_seconds": elapsed,
        "seconds_per_pull_conservative": elapsed / measured_pulls,
    }


def _time_ege(candidate, algorithm: str, probe_budget: int) -> dict:
    env = FrozenHeteroscedasticEnvironment(candidate.means, candidate.stds)
    seed_rngs(20260928, "heteroscedastic-runtime-probe", candidate.instance_id, algorithm, probe_budget)
    start = time.perf_counter()
    EGE_ALGORITHMS[algorithm](
        T=probe_budget,
        K=env.num_arms,
        D=env.num_objectives,
        environment=env,
    )
    elapsed = time.perf_counter() - start
    return {
        "algorithm": algorithm,
        "kind": "ege",
        "probe_budget": probe_budget,
        "warmup_pulls": 0,
        "warmup_seconds": 0.0,
        "timed_steady_pulls": probe_budget,
        "elapsed_seconds": elapsed,
        "seconds_per_pull_conservative": elapsed / probe_budget,
    }


def run_probe(measured_anytime_pulls: int = 200, max_ege_probe_budget: int = 50_000) -> pd.DataFrame:
    """Run selected feasibility timings and write an auditable table."""
    candidates = load_frozen_candidates()
    _warm_runtime_dependencies(candidates)
    rows = []
    for candidate in _representatives(candidates):
        for algorithm in ANYTIME_ALGORITHMS:
            row = _time_anytime(candidate, algorithm, measured_anytime_pulls)
            rows.append({"instance_id": candidate.instance_id, "hardness": candidate.hardness.hardness,
                         "num_arms": candidate.means.shape[0], "num_objectives": candidate.means.shape[1],
                         **row})
        probe_budget = min(candidate.budget, max_ege_probe_budget)
        for algorithm in EGE_ALGORITHMS:
            row = _time_ege(candidate, algorithm, probe_budget)
            rows.append({"instance_id": candidate.instance_id, "hardness": candidate.hardness.hardness,
                         "num_arms": candidate.means.shape[0], "num_objectives": candidate.means.shape[1],
                         **row})
    table = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "runtime_probe.csv", index=False)
    return table


def _warm_runtime_dependencies(candidates) -> None:
    """Exclude one-time JIT/import compilation from all reported timings."""
    candidate = min(
        (item for item in candidates if item.condition == "heterogeneous"),
        key=lambda item: item.means.shape[0],
    )
    for algorithm in ANYTIME_ALGORITHMS:
        _time_anytime(candidate, algorithm, measured_pulls=1)
    for algorithm in EGE_ALGORITHMS:
        _time_ege(candidate, algorithm, probe_budget=max(1, candidate.means.shape[0]))


def write_projection(table: pd.DataFrame) -> dict:
    """Conservative linear projection based on measured rows, never a claim of exact runtime."""
    projections = []
    for _, row in table.iterrows():
        hardness = float(row["hardness"])
        per_pull = float(row["seconds_per_pull_conservative"])
        multiplier = 5.5 if row["kind"] == "ege" else 1.0
        projections.append(
            {
                "instance_id": int(row["instance_id"]),
                "algorithm": row["algorithm"],
                "kind": row["kind"],
                "hardness": hardness,
                "grid_multiplier": multiplier,
                "projected_seconds_one_condition": hardness * per_pull * multiplier,
            }
        )
    projection = pd.DataFrame(projections)
    projection.to_csv(REPORTS_DIR / "runtime_projection.csv", index=False)
    summary = {
        "status": "preliminary_projection_not_a_full_experiment",
        "assumption": (
            "Linear extrapolation from representative local timing; EGE is multiplied by 5.5 "
            "because 0 plus ten normalized budgets have a budget sum of 5.5B."
        ),
        "representatives": sorted(int(value) for value in table["instance_id"].unique()),
        "max_projected_core_hours_one_condition_algorithm": float(
            projection["projected_seconds_one_condition"].max() / 3600.0
        ),
    }
    (REPORTS_DIR / "runtime_projection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anytime-pulls", type=int, default=200)
    parser.add_argument("--max-ege-probe-budget", type=int, default=50_000)
    args = parser.parse_args()
    if args.anytime_pulls < 1 or args.max_ege_probe_budget < 1:
        parser.error("probe budgets must be positive")
    table = run_probe(args.anytime_pulls, args.max_ege_probe_budget)
    summary = write_projection(table)
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
