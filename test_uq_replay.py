"""Regression tests for the hardness-budget uncertainty-quantification replay."""

from __future__ import annotations

import csv
import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.uq.analyze import trajectory_correlations
from experiments.uq.config import environment_config, load_config
from experiments.uq.metrics import uncertainty_from_moments, uncertainty_from_tposterior_state
from experiments.uq.metrics import true_pareto_boundary_pairs, truth_conditioned_uncertainty_from_moments
from experiments.uq.replay import replay_repetition, shard_paths
from environments.EgeExp6 import EgeExp6
from environments.EgeExp8 import EgeExp8


FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def run(name: str, fn) -> None:
    print(f"\n--- {name} ---")
    try:
        fn()
    except Exception:
        FAILURES.append(name)
        print(f"[FAIL] {name} raised:")
        traceback.print_exc()


def test_moment_metric_validity() -> None:
    means = np.array([[1.0], [0.0]])
    stds = np.ones((2, 1))
    result = uncertainty_from_moments(means, stds)
    check("finite two-front moments are available", result.available)
    check("Bhattacharyya has analytic value", np.isclose(result.bhattacharyya, np.exp(-0.125)))
    check("symmetric KL has analytic value", np.isclose(result.symmetric_kl, 0.5))
    check("all-one-front posterior is unavailable", not uncertainty_from_moments(
        np.array([[1.0], [1.0]]), stds
    ).available)
    check("infinite posterior variance is unavailable", not uncertainty_from_moments(
        means, np.array([[np.inf], [1.0]])
    ).available)


def test_tposterior_finite_variance_boundary() -> None:
    means = np.array([[1.0], [0.0]])
    m2 = np.ones((2, 1))
    n_three = np.full((2, 1), 3, dtype=int)
    n_four = np.full((2, 1), 4, dtype=int)
    check(
        "three observations produce unavailable infinite variance",
        not uncertainty_from_tposterior_state(n_three, means, m2).available,
    )
    check(
        "four observations produce finite valid uncertainty",
        uncertainty_from_tposterior_state(n_four, means, m2).available,
    )


def test_trajectory_correlation_contract() -> None:
    rows = pd.DataFrame(
        {
            "environment": ["EgeExp1"] * 5,
            "repetition": [0] * 5,
            "arm_pull": [0, 1, 2, 3, 4],
            "normalized_budget": [0, 0.25, 0.5, 0.75, 1],
            "uq_available": [0, 1, 1, 1, 1],
            "bhattacharyya": [np.nan, 1.0, 0.8, 0.5, 0.2],
            "jaccard": [0.0, 0.2, 0.4, 0.7, 1.0],
        }
    )
    correlation = trajectory_correlations(rows, "bhattacharyya").iloc[0]
    check("correlation omits budget zero", correlation["n_valid_checkpoints"] == 4)
    check("correlation is trajectory-wise Pearson", correlation["pearson_r"] < -0.9)


def test_truth_conditioned_boundary_contract() -> None:
    environment = EgeExp8()
    pairs = true_pareto_boundary_pairs(environment.arms, environment.pareto_indices)
    check("EgeExp8 fixed boundary is optimal arm 4 versus challenger 3", pairs == ((4, 3),))
    result = truth_conditioned_uncertainty_from_moments(
        environment.arms, np.full_like(environment.arms, 0.003), pairs
    )
    check("EgeExp8 fixed boundary UQ is available", result.available)
    no_boundary = true_pareto_boundary_pairs(EgeExp6().arms, EgeExp6().pareto_indices)
    check("all-Pareto EgeExp6 has no dominated boundary", no_boundary == ())
    unavailable = truth_conditioned_uncertainty_from_moments(
        EgeExp6().arms, np.full_like(EgeExp6().arms, 0.003), no_boundary
    )
    check("all-Pareto boundary UQ is explicitly unavailable", not unavailable.available)


def test_instrumented_replay_checkpoint_contract() -> None:
    config = load_config()
    root = Path(tempfile.mkdtemp(prefix="mots-uq-test-"))
    try:
        replay_repetition("EgeExp2", 0, root, config)
        csv_path, state_path = shard_paths(root, "EgeExp2", 0)
        with csv_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        details = environment_config("EgeExp2", config)
        check("replay writes every normalized checkpoint", [int(row["arm_pull"]) for row in rows] == details["grid"])
        check("budget zero is explicitly unavailable", rows[0]["uq_reason"] == "budget_zero")
        check("pre-warmup checkpoint is unavailable", rows[1]["uq_available"] == "0")
        check("post-warmup checkpoint is available", rows[2]["uq_available"] == "1")
        with np.load(state_path, allow_pickle=False) as state:
            check("state checkpoint count matches rows", state["n"].shape[0] == len(rows))
            check("state counts equal pulls times objectives", all(
                int(state["n"][index].sum()) == int(row["arm_pull"]) * details["num_objectives"]
                for index, row in enumerate(rows)
            ))
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    run("Moment metric validity", test_moment_metric_validity)
    run("Finite-variance warm-up boundary", test_tposterior_finite_variance_boundary)
    run("Trajectory correlation contract", test_trajectory_correlation_contract)
    run("Truth-conditioned boundary contract", test_truth_conditioned_boundary_contract)
    run("Instrumented replay checkpoint contract", test_instrumented_replay_checkpoint_contract)
    print("\n" + "=" * 80)
    if FAILURES:
        print("FAILED:", ", ".join(FAILURES))
        raise SystemExit(1)
    print("ALL TESTS PASSED")
