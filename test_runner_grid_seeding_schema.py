"""
Standalone regression tests for Phase 5 runner primitives: evaluation grids,
deterministic seeding, instance hashing, and the tidy result schema
(including recomputable metrics).

No pytest dependency (matches the repo's existing convention, e.g.
test_ape_equivalence.py / test_environment_reward_hardness_gate.py): run
directly with

    python test_runner_grid_seeding_schema.py

and check for "ALL TESTS PASSED" at the end / a non-zero exit code on
failure.

Scope: runner/grid.py, runner/seeding.py, runner/instance.py, runner/schema.py
only. Does NOT touch shards/merge (see test_runner_shards_integrity.py) or the
end-to-end momab_main.py CLI pipeline (see test_runner_pipeline_invariance.py).
"""

import sys
import traceback

import numpy as np

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def run(name, fn):
    print(f"\n--- {name} ---")
    try:
        fn()
    except Exception:
        FAILURES.append(name)
        print(f"[FAIL] {name} raised an exception:")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 1. Endpoint-inclusive evaluation grid.
# ---------------------------------------------------------------------------
def test_grid_endpoints():
    from runner.grid import build_evaluation_grid

    # Non-divisible stride: 0 and max_budget must still appear.
    grid = build_evaluation_grid(max_budget=100, step=7)
    check("grid includes 0", 0 in grid)
    check("grid includes max_budget=100", 100 in grid)
    check("grid is sorted+unique", grid == sorted(set(grid)))
    check("grid matches expected stride points",
          grid == sorted(set(range(0, 101, 7)) | {0, 100}), detail=str(grid))

    # step == max_budget with no other multiples in between.
    grid2 = build_evaluation_grid(max_budget=10, step=1000)
    check("huge step still yields exactly {0, B}", grid2 == [0, 10], detail=str(grid2))

    # step == 1 dense grid.
    grid3 = build_evaluation_grid(max_budget=5, step=1)
    check("step=1 dense grid is 0..B", grid3 == list(range(6)), detail=str(grid3))

    # max_budget == 0 edge case.
    grid4 = build_evaluation_grid(max_budget=0, step=5)
    check("max_budget=0 grid is [0]", grid4 == [0], detail=str(grid4))

    # Invalid step.
    try:
        build_evaluation_grid(max_budget=10, step=0)
        check("step=0 raises ValueError", False)
    except ValueError:
        check("step=0 raises ValueError", True)


def test_grid_explicit_budgets():
    from runner.grid import build_evaluation_grid, parse_explicit_budgets

    parsed = parse_explicit_budgets(" 0, 5,5, 20 ")
    check("parse_explicit_budgets dedups/parses ints", parsed == [0, 5, 5, 20],
          detail=str(parsed))  # parsing itself does not dedupe, only the grid builder does

    grid = build_evaluation_grid(max_budget=50, explicit_budgets=[5, 20, 20])
    check("explicit budgets include requested + endpoints",
          grid == [0, 5, 20, 50], detail=str(grid))

    grid_default_endpoints = build_evaluation_grid(max_budget=50, explicit_budgets=[0, 50])
    check("explicit budgets already covering endpoints stay minimal",
          grid_default_endpoints == [0, 50])

    try:
        build_evaluation_grid(max_budget=10, explicit_budgets=[5, 11])
        check("out-of-range explicit budget raises", False)
    except ValueError:
        check("out-of-range explicit budget raises", True)

    check("parse_explicit_budgets(None) is None", parse_explicit_budgets(None) is None)
    check("parse_explicit_budgets('') is None", parse_explicit_budgets("") is None)


def test_same_grid_every_algorithm():
    """The grid helper is a pure function of (max_budget, step/explicit); every
    algorithm in an environment must call it identically, so two independent
    calls with the same arguments must be identical (this is what guarantees
    'same grid for every algorithm')."""
    from runner.grid import build_evaluation_grid

    grid_a = build_evaluation_grid(max_budget=777, step=13)
    grid_b = build_evaluation_grid(max_budget=777, step=13)
    check("repeated grid construction is identical", grid_a == grid_b)


# ---------------------------------------------------------------------------
# 2. Deterministic seeding.
# ---------------------------------------------------------------------------
def test_seed_determinism_and_independence():
    from runner.seeding import derive_seed, seed_rngs

    s1 = derive_seed(42, "EgeExp8|gaussian", "env", 0)
    s2 = derive_seed(42, "EgeExp8|gaussian", "env", 0)
    check("derive_seed is deterministic", s1 == s2)

    s3 = derive_seed(42, "EgeExp8|gaussian", "env", 1)
    check("different repetition -> different seed", s1 != s3)

    s4 = derive_seed(43, "EgeExp8|gaussian", "env", 0)
    check("different master seed -> different seed", s1 != s4)

    s5 = derive_seed(42, "EgeExp1|gaussian", "env", 0)
    check("different environment tag -> different seed", s1 != s5)

    s_budget_10 = derive_seed(42, "EgeExp8|gaussian", "EGE_SR", 0, "budget", 10)
    s_budget_20 = derive_seed(42, "EgeExp8|gaussian", "EGE_SR", 0, "budget", 20)
    check("different EGE budget -> different seed", s_budget_10 != s_budget_20)

    check("derive_seed(None, ...) is None", derive_seed(None, "x") is None)
    check("seed_rngs(None, ...) is None (no-op)", seed_rngs(None, "x") is None)

    # seed_rngs actually seeds both numpy and stdlib random reproducibly.
    import random
    seed_rngs(7, "tagA")
    draw_np_1 = np.random.random()
    draw_py_1 = random.random()
    seed_rngs(7, "tagA")
    draw_np_2 = np.random.random()
    draw_py_2 = random.random()
    check("seed_rngs reproduces numpy draw", draw_np_1 == draw_np_2)
    check("seed_rngs reproduces stdlib random draw", draw_py_1 == draw_py_2)
    check("numpy and stdlib streams are not trivially identical", draw_np_1 != draw_py_1)


# ---------------------------------------------------------------------------
# 3. Instance hashing.
# ---------------------------------------------------------------------------
def test_instance_hash():
    from environments.EgeExp8 import EgeExp8
    from runner.instance import compute_instance_hash

    env_a = EgeExp8()
    env_b = EgeExp8()
    check("same frozen environment class -> same instance hash",
          compute_instance_hash(env_a) == compute_instance_hash(env_b))

    from environments.EgeExp1 import EgeExp1
    env_c = EgeExp1()
    check("different environment -> different instance hash",
          compute_instance_hash(env_a) != compute_instance_hash(env_c))


# ---------------------------------------------------------------------------
# 4. Schema: stable JSON encoding + recomputable metrics.
# ---------------------------------------------------------------------------
def test_schema_stable_set_encoding():
    from runner.schema import encode_int_set, decode_int_set

    e1 = encode_int_set([3, 1, 2, 1])
    e2 = encode_int_set({2, 3, 1})
    check("encode_int_set is order-independent", e1 == e2, detail=f"{e1} vs {e2}")
    check("encode_int_set is compact JSON", e1 == "[1,2,3]", detail=e1)
    check("decode_int_set round-trips", decode_int_set(e1) == {1, 2, 3})


def test_schema_columns_and_build_row():
    from runner.schema import SCHEMA_COLUMNS, build_row

    row = build_row(
        environment="EgeExp8", instance_hash="deadbeef", algorithm="APE",
        posterior_model="", repetition=0, arm_pull=10, max_budget=30,
        evaluation_grid=[0, 10, 30], rho=None, master_seed=42,
        recommendation=[4], true_pareto=[4],
    )
    check("build_row produces exactly the schema columns",
          set(row.keys()) == set(SCHEMA_COLUMNS), detail=str(sorted(row.keys())))
    check("build_row computes bernoulli=1 for a perfect match", row["bernoulli"] == 1)
    check("build_row computes jaccard=1.0 for a perfect match", row["jaccard"] == 1.0)
    check("build_row computes misidentification=0.0 for a perfect match",
          row["misidentification"] == 0.0)
    check("evaluation_grid is stably encoded", row["evaluation_grid"] == "[0,10,30]")


def test_recompute_metrics_matches_base_environment():
    """Metrics recomputed purely from stored recommendation/true_pareto sets
    must match BaseEnvironment's own metric methods exactly, for arbitrary
    recommendation/pareto set pairs (not just the trivial exact-match case)."""
    from environments.EgeExp8 import EgeExp8
    from runner.schema import recompute_metrics

    env = EgeExp8()
    true_pareto = set(env.get_top_arms().tolist())
    rng = np.random.default_rng(0)
    for _ in range(50):
        k = rng.integers(0, env.num_arms + 1)
        rec = set(rng.choice(env.num_arms, size=k, replace=False).tolist())
        expected = {
            "bernoulli": env.bernoulli_metric(rec),
            "jaccard": env.jaccard_metric(rec),
            "misidentification": env.mis_id_metric(rec),
        }
        got = recompute_metrics(rec, true_pareto, env.num_arms)
        check(f"recompute_metrics matches BaseEnvironment for rec={sorted(rec)}",
              got == expected, detail=f"{got} vs {expected}")


def test_validate_row_metrics_detects_corruption():
    from runner.schema import build_row, validate_row_metrics

    row = build_row(
        environment="EgeExp8", instance_hash="deadbeef", algorithm="Uniform",
        posterior_model="", repetition=0, arm_pull=5, max_budget=30,
        evaluation_grid=[0, 5, 30], rho=None, master_seed=42,
        recommendation=[0, 1], true_pareto=[0, 1, 2, 3, 4],
    )
    problems = validate_row_metrics(row, num_arms=5)
    check("untouched row has no metric mismatches", problems == [], detail=str(problems))

    corrupted = dict(row)
    corrupted["bernoulli"] = 1  # actual sets differ, so this should be 0
    problems2 = validate_row_metrics(corrupted, num_arms=5)
    check("corrupted bernoulli is detected", len(problems2) == 1, detail=str(problems2))


if __name__ == "__main__":
    run("grid endpoints", test_grid_endpoints)
    run("grid explicit budgets", test_grid_explicit_budgets)
    run("same grid for every algorithm", test_same_grid_every_algorithm)
    run("seed determinism and independence", test_seed_determinism_and_independence)
    run("instance hash", test_instance_hash)
    run("schema stable set encoding", test_schema_stable_set_encoding)
    run("schema columns and build_row", test_schema_columns_and_build_row)
    run("recompute_metrics matches BaseEnvironment", test_recompute_metrics_matches_base_environment)
    run("validate_row_metrics detects corruption", test_validate_row_metrics_detects_corruption)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)/test(s) failed: {FAILURES}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)
