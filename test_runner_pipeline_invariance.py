"""
Standalone end-to-end regression tests for the Phase 5 momab_main.py
runner: exact pull counts, seed/grid invariance (dense vs. sparse, serial vs.
parallel, grouped vs. isolated), and resume-from-partial-shard behavior.

No pytest dependency (matches the repo's existing convention). Run directly
with:

    python test_runner_pipeline_invariance.py

Uses tiny budgets/environments so the full suite runs in a few seconds; does
not choose or validate any "real" experiment budgets (out of scope).
"""

import os
import sys
import tempfile
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


def _read_csv_rows(path):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# 1. Budget B means exactly B pulls.
# ---------------------------------------------------------------------------
def test_exact_pull_count_anytime():
    from environments.EgeExp8 import EgeExp8
    from momab_main import _make_bandit, _run_repetition_anytime

    environment = EgeExp8()
    max_budget = 37
    grid = [0, 10, 37]
    bandit = _make_bandit("TTPFTS_KNOWN_VAR", environment, p=0.5)
    _run_repetition_anytime(environment, "EgeExp8", "gaussian", "TTPFTS_KNOWN_VAR", bandit,
                            max_budget, grid, master_seed=1, repetition=0)
    # counts has shape (num_arms, num_objectives); every pull increments one
    # arm's row across all objectives, so summing one objective column
    # gives the true pull count.
    total_pulls = int(bandit.posterior.counts[:, 0].sum())
    check("exactly max_budget pulls were made (counting posterior updates)",
          total_pulls == max_budget, detail=f"got {total_pulls}")

    # Same check for a bandit whose per-arm pull counter is directly exposed.
    bandit_uniform = _make_bandit("Uniform", environment, p=0.5)
    _run_repetition_anytime(environment, "EgeExp8", "gaussian", "Uniform", bandit_uniform,
                            max_budget, grid, master_seed=1, repetition=0)
    total_pulls_uniform = int(bandit_uniform.arm_counts.sum())
    check("Uniform bandit received exactly max_budget pulls",
          total_pulls_uniform == max_budget, detail=f"got {total_pulls_uniform}")


def test_prior_recommendation_logged_at_zero():
    """arm_pull=0 must log the *prior* recommendation (before any pull)."""
    from environments.EgeExp8 import EgeExp8
    from momab_main import _make_bandit, _run_repetition_anytime

    environment = EgeExp8()
    bandit = _make_bandit("Uniform", environment, p=0.5)
    rows = _run_repetition_anytime(environment, "EgeExp8", "gaussian", "Uniform", bandit,
                                   max_budget=20, grid=[0, 20], master_seed=3, repetition=0)
    row0 = next(r for r in rows if r["arm_pull"] == 0)
    # Uniform's prior (all-zero means) recommendation is every arm (all tied).
    import json
    check("arm_pull=0 recommendation is the untrained prior (all arms tied)",
          json.loads(row0["recommendation"]) == list(range(environment.num_arms)),
          detail=row0["recommendation"])


def test_ege_budget_means_fixed_budget():
    from environments.EgeExp8 import EgeExp8
    from momab_main import _run_repetition_ege
    from bandits.ege_kone import EGE_SR

    environment = EgeExp8()
    grid = [0, 5, 20]
    rows = _run_repetition_ege(environment, "EgeExp8", "gaussian", "EGE_SR", EGE_SR,
                               max_budget=20, grid=grid, master_seed=9, repetition=0)
    check("EGE logs exactly one row per grid point", len(rows) == len(grid))
    check("EGE's arm_pull values equal the requested (fixed) budgets",
          sorted(int(r["arm_pull"]) for r in rows) == grid)


# ---------------------------------------------------------------------------
# 2. Endpoint-inclusive grid, same for every algorithm (integration-level).
# ---------------------------------------------------------------------------
def test_grid_shared_across_algorithms_end_to_end():
    from momab_main import ENVIRONMENTS

    with tempfile.TemporaryDirectory() as tmp:
        _invoke_cli([
            "--algorithms", "Uniform,APE,EGE_SR",
            "--environments", "EgeExp8",
            "--budgets", "23",
            "--num_runs", "2",
            "--step", "9",
            "--results_dir", tmp,
            "--seed", "11",
        ])
        rows = _read_csv_rows(os.path.join(tmp, "EgeExp8.csv"))
        grids_by_algo = {}
        for r in rows:
            grids_by_algo.setdefault(r["algorithm"], set()).add(int(r["arm_pull"]))
        check("exactly 3 algorithms present", set(grids_by_algo) == {"Uniform", "APE", "EGE_SR"})
        grids = list(grids_by_algo.values())
        check("every algorithm was evaluated on the identical grid",
              all(g == grids[0] for g in grids), detail=str(grids_by_algo))
        check("grid includes 0 and 23 (endpoint-inclusive, step=9 doesn't divide 23)",
              grids[0] == {0, 9, 18, 23}, detail=str(grids[0]))


def _invoke_cli(cli_args):
    """Run momab_main.py's CLI body in-process (not a subprocess) so tests
    stay fast, by monkeypatching sys.argv and re-executing the module."""
    import runpy
    old_argv = sys.argv
    sys.argv = ["momab_main.py"] + cli_args
    try:
        runpy.run_module("momab_main", run_name="__main__")
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# 3. EGE seed/grid invariance: shared budgets are identical under dense vs.
#    sparse grids (and thus also under resume with a different, denser grid).
# ---------------------------------------------------------------------------
def test_ege_dense_sparse_grid_invariance():
    from environments.EgeExp3 import EgeExp3
    from momab_main import _run_repetition_ege
    from bandits.ege_kone import EGE_SR

    max_budget = 60
    sparse_grid = [0, 30, 60]
    dense_grid = list(range(0, 61, 5))
    shared_budgets = set(sparse_grid) & set(dense_grid)
    check("sparse and dense grids share budgets to compare", len(shared_budgets) >= 2)

    env_sparse = EgeExp3()
    rows_sparse = _run_repetition_ege(env_sparse, "EgeExp3", "gaussian", "EGE_SR", EGE_SR,
                                      max_budget, sparse_grid, master_seed=5, repetition=2)
    env_dense = EgeExp3()
    rows_dense = _run_repetition_ege(env_dense, "EgeExp3", "gaussian", "EGE_SR", EGE_SR,
                                     max_budget, dense_grid, master_seed=5, repetition=2)

    by_budget_sparse = {int(r["arm_pull"]): r for r in rows_sparse}
    by_budget_dense = {int(r["arm_pull"]): r for r in rows_dense}
    for budget in shared_budgets:
        rs, rd = by_budget_sparse[budget], by_budget_dense[budget]
        check(f"EGE budget={budget} recommendation identical (sparse vs dense grid)",
              rs["recommendation"] == rd["recommendation"],
              detail=f"{rs['recommendation']} vs {rd['recommendation']}")
        check(f"EGE budget={budget} metrics identical (sparse vs dense grid)",
              (rs["bernoulli"], rs["jaccard"], rs["misidentification"]) ==
              (rd["bernoulli"], rd["jaccard"], rd["misidentification"]))


def test_anytime_step_invariance():
    """A trajectory's logged value at a shared arm_pull must not depend on
    --step, since the trajectory pulls every step regardless of the grid."""
    from environments.EgeExp8 import EgeExp8
    from momab_main import _make_bandit, _run_repetition_anytime

    max_budget = 40
    env_a = EgeExp8()
    bandit_a = _make_bandit("TTPFTS_KNOWN_VAR", env_a, p=0.5)
    rows_a = _run_repetition_anytime(env_a, "EgeExp8", "gaussian", "TTPFTS_KNOWN_VAR", bandit_a,
                                     max_budget, grid=[0, 20, 40], master_seed=13, repetition=0)

    env_b = EgeExp8()
    bandit_b = _make_bandit("TTPFTS_KNOWN_VAR", env_b, p=0.5)
    rows_b = _run_repetition_anytime(env_b, "EgeExp8", "gaussian", "TTPFTS_KNOWN_VAR", bandit_b,
                                     max_budget, grid=list(range(0, 41, 4)), master_seed=13, repetition=0)

    by_t_a = {int(r["arm_pull"]): r for r in rows_a}
    by_t_b = {int(r["arm_pull"]): r for r in rows_b}
    for t in (0, 20, 40):
        check(f"anytime arm_pull={t} recommendation identical across grids",
              by_t_a[t]["recommendation"] == by_t_b[t]["recommendation"])


# ---------------------------------------------------------------------------
# 4. Serial vs. parallel and grouped vs. isolated invariance (via the CLI).
# ---------------------------------------------------------------------------
def test_serial_vs_parallel_invariance():
    with tempfile.TemporaryDirectory() as tmp_serial, tempfile.TemporaryDirectory() as tmp_parallel:
        common = [
            "--algorithms", "TTPFTS_UNKNOWN_VAR,EGE_SH",
            "--environments", "EgeExp8",
            "--budgets", "25",
            "--num_runs", "4",
            "--step", "5",
            "--seed", "21",
        ]
        _invoke_cli(common + ["--results_dir", tmp_serial, "--n_jobs", "1"])
        _invoke_cli(common + ["--results_dir", tmp_parallel, "--n_jobs", "2"])

        rows_serial = _read_csv_rows(os.path.join(tmp_serial, "EgeExp8.csv"))
        rows_parallel = _read_csv_rows(os.path.join(tmp_parallel, "EgeExp8.csv"))
        key = lambda r: (r["algorithm"], int(r["repetition"]), int(r["arm_pull"]))
        rows_serial.sort(key=key)
        rows_parallel.sort(key=key)
        check("serial and parallel (n_jobs=1 vs 2) produce identical row counts",
              len(rows_serial) == len(rows_parallel))
        mismatches = [
            (rs["algorithm"], rs["repetition"], rs["arm_pull"])
            for rs, rp in zip(rows_serial, rows_parallel)
            if rs["recommendation"] != rp["recommendation"] or rs["bernoulli"] != rp["bernoulli"]
        ]
        check("serial and parallel results are byte-identical", not mismatches,
              detail=str(mismatches[:5]))


def test_grouped_vs_isolated_invariance():
    with tempfile.TemporaryDirectory() as tmp_grouped, tempfile.TemporaryDirectory() as tmp_isolated:
        base = ["--environments", "EgeExp8", "--budgets", "22", "--num_runs", "3",
               "--step", "11", "--seed", "31"]

        _invoke_cli(["--algorithms", "TTPFTS_KNOWN_VAR,APE"] + base + ["--results_dir", tmp_grouped])

        _invoke_cli(["--algorithms", "TTPFTS_KNOWN_VAR"] + base + ["--results_dir", tmp_isolated])
        _invoke_cli(["--algorithms", "APE"] + base + ["--results_dir", tmp_isolated])

        grouped = _read_csv_rows(os.path.join(tmp_grouped, "EgeExp8.csv"))
        isolated_known_var = _read_csv_rows(os.path.join(tmp_isolated, "EgeExp8.csv"))

        def by_key(rows, algo):
            return {
                (int(r["repetition"]), int(r["arm_pull"])): r["recommendation"]
                for r in rows if r["algorithm"] == algo
            }

        for algo in ("TTPFTS_KNOWN_VAR", "APE"):
            g = by_key(grouped, algo)
            iso = by_key(isolated_known_var, algo)
            check(f"{algo}: grouped vs isolated invocation give identical recommendations",
                  g == iso, detail=f"mismatch count={sum(1 for k in g if g.get(k) != iso.get(k))}")


# ---------------------------------------------------------------------------
# 5. Partial shard rejection / resume, atomic merge (end-to-end via the CLI).
# ---------------------------------------------------------------------------
def test_resume_recomputes_only_incomplete_shards():
    from runner.shards import shard_path, read_shard, write_shard_atomic

    with tempfile.TemporaryDirectory() as tmp:
        args = ["--algorithms", "Uniform", "--environments", "EgeExp8", "--budgets", "12",
               "--num_runs", "3", "--step", "4", "--results_dir", tmp, "--seed", "77"]
        _invoke_cli(args)

        run_dir = os.path.join(tmp, "run")
        path_rep1 = shard_path(run_dir, "EgeExp8", "Uniform", 1)
        original_rep1 = read_shard(path_rep1)
        mtimes_before = {r: os.path.getmtime(shard_path(run_dir, "EgeExp8", "Uniform", r))
                        for r in range(3)}

        # Corrupt repetition 1's shard (drop the last row -> incomplete grid).
        write_shard_atomic(path_rep1, original_rep1[:-1])

        _invoke_cli(args)

        mtimes_after = {r: os.path.getmtime(shard_path(run_dir, "EgeExp8", "Uniform", r))
                       for r in range(3)}
        check("only the corrupted repetition's shard was rewritten",
              mtimes_after[1] != mtimes_before[1]
              and mtimes_after[0] == mtimes_before[0]
              and mtimes_after[2] == mtimes_before[2],
              detail=f"before={mtimes_before} after={mtimes_after}")

        repaired_rep1 = read_shard(path_rep1)
        check("recomputed shard is complete again (correct grid restored)",
              [r["arm_pull"] for r in repaired_rep1] == [r["arm_pull"] for r in original_rep1])
        check("recomputed shard content matches the original (deterministic recompute)",
              repaired_rep1 == original_rep1)


if __name__ == "__main__":
    run("exact pull count (anytime)", test_exact_pull_count_anytime)
    run("prior recommendation logged at arm_pull=0", test_prior_recommendation_logged_at_zero)
    run("EGE budget means fixed budget", test_ege_budget_means_fixed_budget)
    run("grid shared across algorithms (end-to-end)", test_grid_shared_across_algorithms_end_to_end)
    run("EGE dense/sparse grid invariance", test_ege_dense_sparse_grid_invariance)
    run("anytime step invariance", test_anytime_step_invariance)
    run("serial vs parallel invariance", test_serial_vs_parallel_invariance)
    run("grouped vs isolated invariance", test_grouped_vs_isolated_invariance)
    run("resume recomputes only incomplete shards", test_resume_recomputes_only_incomplete_shards)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)/test(s) failed: {FAILURES}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)
