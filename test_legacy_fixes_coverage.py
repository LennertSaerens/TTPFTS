#!/usr/bin/env python3
"""Retirement coverage for the former ``verify_fixes.py`` script.

``verify_fixes.py`` was a one-off regression script from an earlier revision
and is absent from git history/worktree.  The checks below document and
exercise the five behaviours it covered so the coverage is not silently lost.
Run directly with ``python test_legacy_fixes_coverage.py`` or via
``run_all_gates.py``.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def run(name, fn) -> None:
    print(f"\n--- {name} ---")
    try:
        fn()
    except Exception:
        FAILURES.append(name)
        print(f"[FAIL] {name} raised an exception:")
        traceback.print_exc()


def _invoke_cli_expect_system_exit(cli_args):
    import runpy

    old_argv = sys.argv
    sys.argv = ["momab_main.py", *cli_args]
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            runpy.run_module("momab_main", run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0), stderr.getvalue()
    finally:
        sys.argv = old_argv
    return 0, stderr.getvalue()


def test_anytime_pull_logging_decoupling() -> None:
    from environments.EgeExp8 import EgeExp8
    from momab_main import _make_bandit, _run_repetition_anytime

    env = EgeExp8()
    bandit = _make_bandit("Uniform", env, p=0.5)
    rows = _run_repetition_anytime(
        env, "EgeExp8", "gaussian", "Uniform", bandit,
        max_budget=37, grid=[0, 10, 37], master_seed=101, repetition=0,
    )
    check("sparse grid logs only requested arm-pulls",
          [int(r["arm_pull"]) for r in rows] == [0, 10, 37])
    check("sparse grid still performs exactly B pulls",
          int(bandit.arm_counts.sum()) == 37, f"got {bandit.arm_counts.sum()}")


def test_ege_low_budget_initialisation_and_random_ties() -> None:
    from bandits.ege_kone import EGE_SR
    from environments.EgeExp8 import EgeExp8

    # Regression for the old np.empty/no-sample artifact: at T=0 there are no
    # rewards, so recommendations should not be a deterministic perfect hit.
    env = EgeExp8()
    rec0 = set(EGE_SR(T=0, K=env.num_arms, D=env.num_objectives, environment=env))
    true = set(env.get_top_arms().tolist())
    check("EGE_SR(T=0) no longer reads uninitialised means into a perfect hit",
          rec0 != true, f"recommendation={sorted(rec0)} true={sorted(true)}")

    # Regression for index-ordered tie-breaking: with all empirical means tied
    # at tiny budgets, different RNG seeds should be able to choose different
    # arms instead of always following the same low-index path.
    recs = set()
    for seed in range(20):
        np.random.seed(seed)
        env = EgeExp8()
        recs.add(tuple(EGE_SR(T=0, K=env.num_arms, D=env.num_objectives, environment=env)))
    check("EGE_SR tie-breaking is randomised across seeds",
          len(recs) > 1, f"distinct recommendations={sorted(recs)}")


def test_per_run_seeding_invariance() -> None:
    from environments.EgeExp8 import EgeExp8
    from momab_main import _make_bandit, _run_repetition_anytime

    def run_one(seed, repetition):
        env = EgeExp8()
        bandit = _make_bandit("TTPFTS_KNOWN_VAR", env, p=0.5)
        rows = _run_repetition_anytime(
            env, "EgeExp8", "gaussian", "TTPFTS_KNOWN_VAR", bandit,
            max_budget=25, grid=[0, 5, 25], master_seed=seed, repetition=repetition,
        )
        return [r["recommendation"] for r in rows]

    check("same seed/repetition is deterministic", run_one(7, 0) == run_one(7, 0))
    check("different repetition changes trajectory", run_one(7, 0) != run_one(7, 1))
    check("different master seed changes trajectory", run_one(7, 0) != run_one(8, 0))


def test_paretoset_distinct_false_for_ties() -> None:
    from environments.BaseEnvironment import BaseEnvironment
    from environments.EgeExp8 import EgeExp8
    from runner.schema import build_row

    tied = np.array([[1.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    indices = BaseEnvironment._compute_pareto_indices(tied)
    check("BaseEnvironment keeps identical Pareto rows (distinct=False)",
          indices.tolist() == [0, 1], f"got {indices.tolist()}")

    env = EgeExp8()
    row = build_row(
        environment="EgeExp8", instance_hash="x", algorithm="dummy",
        posterior_model="", repetition=0, arm_pull=0, max_budget=0,
        evaluation_grid=[0], rho=None, master_seed=0,
        recommendation=[0, 1], true_pareto=[0, 1],
    )
    check("schema preserves tied set recommendations exactly",
          json.loads(row["recommendation"]) == [0, 1])


def test_cli_budget_environment_length_validation() -> None:
    code, err = _invoke_cli_expect_system_exit([
        "--algorithms", "Uniform",
        "--environments", "EgeExp7,EgeExp8",
        "--budgets", "10,20,30",
        "--num_runs", "1",
        "--no_write",
    ])
    check("budget/environment length mismatch exits non-zero", code != 0)
    check("budget/environment length mismatch reports parser error",
          "--budgets has 3 value(s) but --environments has 2" in err,
          err.strip())


if __name__ == "__main__":
    run("anytime pull/logging decoupling", test_anytime_pull_logging_decoupling)
    run("EGE initialisation + random tie-breaking", test_ege_low_budget_initialisation_and_random_ties)
    run("per-run seeding", test_per_run_seeding_invariance)
    run("distinct=False Pareto ties", test_paretoset_distinct_false_for_ties)
    run("CLI budget/environment validation", test_cli_budget_environment_length_validation)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)/test(s) failed: {FAILURES}")
        sys.exit(1)
    print("ALL TESTS PASSED")
    sys.exit(0)
