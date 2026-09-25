"""
Standalone regression tests for Phase 5 atomic sharding + deterministic
merging (runner/shards.py).

No pytest dependency (matches the repo's existing convention). Run directly
with:

    python test_runner_shards_integrity.py

Scope: runner/shards.py only (shard completeness/atomicity/merge). Does NOT
exercise the end-to-end momab_main.py pipeline (see
test_runner_pipeline_invariance.py).
"""

import os
import shutil
import sys
import tempfile
import traceback

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


def _make_row(algorithm, repetition, arm_pull, grid):
    from runner.schema import build_row
    return build_row(
        environment="EgeExp8", instance_hash="deadbeef", algorithm=algorithm,
        posterior_model="", repetition=repetition, arm_pull=arm_pull, max_budget=grid[-1],
        evaluation_grid=grid, rho=None, master_seed=42,
        recommendation=[4], true_pareto=[4],
    )


def test_atomic_write_and_read_roundtrip():
    from runner.shards import read_shard, shard_path, write_shard_atomic

    with tempfile.TemporaryDirectory() as tmp:
        grid = [0, 5, 10]
        rows = [_make_row("APE", 0, t, grid) for t in grid]
        path = shard_path(tmp, "EgeExp8", "APE", 0)
        write_shard_atomic(path, rows)
        check("shard file exists after write", os.path.exists(path))
        check("no leftover temp files", not any(f.startswith(".tmp-") for f in os.listdir(os.path.dirname(path))))
        read_back = read_shard(path)
        check("round-tripped row count matches", len(read_back) == len(rows))
        check("round-tripped arm_pull values match",
              [int(r["arm_pull"]) for r in read_back] == grid)


def test_shard_completeness_detection():
    from runner.shards import shard_is_complete, shard_path, write_shard_atomic

    with tempfile.TemporaryDirectory() as tmp:
        grid = [0, 5, 10]
        path = shard_path(tmp, "EgeExp8", "APE", 0)

        check("missing shard is incomplete", not shard_is_complete(path, grid))

        write_shard_atomic(path, [_make_row("APE", 0, t, grid) for t in grid])
        check("full shard is complete", shard_is_complete(path, grid))

        # Partial shard (missing budget 10): must be rejected, not accepted.
        write_shard_atomic(path, [_make_row("APE", 0, t, grid) for t in [0, 5]])
        check("partial shard (missing a grid point) is rejected", not shard_is_complete(path, grid))

        # Extra/duplicate row: must also be rejected.
        write_shard_atomic(path, [_make_row("APE", 0, t, grid) for t in [0, 5, 5, 10]])
        check("shard with a duplicated arm_pull is rejected", not shard_is_complete(path, grid))

        # Different grid (dense vs sparse) must be rejected against a stale shard.
        write_shard_atomic(path, [_make_row("APE", 0, t, grid) for t in grid])
        denser_grid = [0, 2, 4, 5, 6, 8, 10]
        check("shard computed for a sparser grid is rejected against a denser grid",
              not shard_is_complete(path, denser_grid))

        # Corrupt/truncated file on disk (not via write_shard_atomic) must be rejected.
        with open(path, "w") as f:
            f.write("environment,instance_hash\ntruncated,row\n")
        check("corrupt/wrong-schema shard is rejected", not shard_is_complete(path, grid))

        # Empty file.
        open(path, "w").close()
        check("empty shard file is rejected", not shard_is_complete(path, grid))


def test_needs_recompute_matches_completeness():
    from runner.shards import needs_recompute, shard_path, write_shard_atomic

    with tempfile.TemporaryDirectory() as tmp:
        grid = [0, 3, 6]
        path = shard_path(tmp, "EgeExp8", "Uniform", 2)
        check("missing shard needs recompute", needs_recompute(path, grid))
        write_shard_atomic(path, [_make_row("Uniform", 2, t, grid) for t in grid])
        check("complete shard does not need recompute", not needs_recompute(path, grid))


def test_atomic_replace_leaves_no_partial_state():
    """Simulate a writer crashing mid-write by directly leaving a stray temp
    file; the previous complete shard (if any) must remain the one readers
    observe, never a half-written file."""
    from runner.shards import read_shard, shard_is_complete, shard_path, write_shard_atomic

    with tempfile.TemporaryDirectory() as tmp:
        grid = [0, 4, 8]
        path = shard_path(tmp, "EgeExp8", "APE", 0)
        write_shard_atomic(path, [_make_row("APE", 0, t, grid) for t in grid])
        before = read_shard(path)

        # Simulate an interrupted concurrent writer: a stray temp file next
        # to the real shard, deliberately corrupt/incomplete.
        stray = os.path.join(os.path.dirname(path), ".tmp-shard-stray.csv")
        with open(stray, "w") as f:
            f.write("not,a,valid,shard\n")

        check("previous complete shard is unaffected by a stray temp file",
              read_shard(path) == before)
        check("shard is still complete", shard_is_complete(path, grid))
        os.remove(stray)


def test_merge_is_deterministic_and_excludes_incomplete():
    from runner.shards import merge_shards, shard_path, write_shard_atomic

    with tempfile.TemporaryDirectory() as tmp:
        grid = [0, 5, 10]
        # Two complete repetitions for algorithm A, one complete for B.
        write_shard_atomic(shard_path(tmp, "EgeExp8", "AlgoA", 0),
                          [_make_row("AlgoA", 0, t, grid) for t in grid])
        write_shard_atomic(shard_path(tmp, "EgeExp8", "AlgoA", 1),
                          [_make_row("AlgoA", 1, t, grid) for t in grid])
        write_shard_atomic(shard_path(tmp, "EgeExp8", "AlgoB", 0),
                          [_make_row("AlgoB", 0, t, grid) for t in grid])
        # An incomplete shard for AlgoB repetition 1 (missing budget 10):
        # must be excluded from the merge entirely.
        write_shard_atomic(shard_path(tmp, "EgeExp8", "AlgoB", 1),
                          [_make_row("AlgoB", 1, t, grid) for t in [0, 5]])

        out_path = os.path.join(tmp, "merged.csv")
        expected_grids = {"AlgoA": grid, "AlgoB": grid}
        n_rows = merge_shards(tmp, "EgeExp8", out_path, expected_grids=expected_grids,
                              validate_metrics=True)
        check("merge row count excludes the incomplete shard", n_rows == 3 * len(grid),
              detail=f"got {n_rows}")

        import csv
        with open(out_path) as f:
            rows = list(csv.DictReader(f))
        check("merged file row count matches return value", len(rows) == n_rows)
        pairs = [(r["algorithm"], int(r["repetition"])) for r in rows]
        check("incomplete (AlgoB, rep=1) is absent from merged output",
              ("AlgoB", 1) not in pairs, detail=str(sorted(set(pairs))))

        ordering = [(r["algorithm"], int(r["repetition"]), int(r["arm_pull"])) for r in rows]
        check("merge output is deterministically sorted",
              ordering == sorted(ordering), detail=str(ordering))

        # Re-running the merge must be byte-identical (idempotent/deterministic).
        out_path_2 = os.path.join(tmp, "merged_2.csv")
        merge_shards(tmp, "EgeExp8", out_path_2, expected_grids=expected_grids,
                    validate_metrics=True)
        with open(out_path) as f1, open(out_path_2) as f2:
            check("repeated merge is byte-identical", f1.read() == f2.read())


def test_merge_raises_on_metric_tampering():
    from runner.shards import merge_shards, shard_path, write_shard_atomic

    with tempfile.TemporaryDirectory() as tmp:
        grid = [0, 5]
        rows = [_make_row("AlgoA", 0, t, grid) for t in grid]
        rows[0]["bernoulli"] = 1 - int(rows[0]["bernoulli"])  # tamper
        write_shard_atomic(shard_path(tmp, "EgeExp8", "AlgoA", 0), rows)

        out_path = os.path.join(tmp, "merged.csv")
        try:
            merge_shards(tmp, "EgeExp8", out_path, validate_metrics=True)
            check("merge raises on tampered metrics", False)
        except ValueError:
            check("merge raises on tampered metrics", True)


if __name__ == "__main__":
    run("atomic write/read roundtrip", test_atomic_write_and_read_roundtrip)
    run("shard completeness detection", test_shard_completeness_detection)
    run("needs_recompute matches completeness", test_needs_recompute_matches_completeness)
    run("atomic replace leaves no partial state", test_atomic_replace_leaves_no_partial_state)
    run("merge is deterministic and excludes incomplete shards",
        test_merge_is_deterministic_and_excludes_incomplete)
    run("merge raises on metric tampering", test_merge_raises_on_metric_tampering)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)/test(s) failed: {FAILURES}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)
