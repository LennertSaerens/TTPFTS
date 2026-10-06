#!/usr/bin/env python3
"""Regression gate for the doubling-trick (D-EGE) publication figures."""

from __future__ import annotations

import csv
import shutil
import tempfile
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from bandits.ege_kone import EGE_SH, EGE_SR
from experiments.ege_doubling.build import build, doubling_path
from experiments.ege_doubling.config import (
    REPO_ROOT,
    assert_safe_output,
    default_output_root,
    environment_specs,
    load_config,
    verify_protected_hashes,
    verify_source_hashes,
)
from experiments.ege_doubling.plot import render
from experiments.ege_doubling.run import run, stage_results_path
from experiments.ege_doubling.schedule import doubling_schedule, evaluation_points, stage_at
from experiments.ege_doubling.validate import independent_stage, validate
from momab_main import ENVIRONMENTS, _construct_environment, _run_repetition_ege
from runner.ege_allocation import expected_ege_draws
from runner.seeding import seed_rngs


FAILURES: list[str] = []
SMOKE_ENVIRONMENTS = ["EgeExp2", "EgeExp7"]
SMOKE_RUNS = 3
SMOKE_REPRODUCTION = {"repetitions": [0], "grid_indices": [1, 100]}


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def execute(name: str, fn) -> None:
    print(f"\n=== {name} ===")
    try:
        fn()
    except Exception as error:  # noqa: BLE001 - report every gate failure
        traceback.print_exc()
        check(f"{name} raised no exception", False, f"{type(error).__name__}: {error}")


def raises(fn, exception) -> bool:
    try:
        fn()
    except exception:
        return True
    return False


def test_schedule_arithmetic() -> None:
    schedule = doubling_schedule(22, 7000)
    check("EgeExp7 endpoints match the requested example",
          [stage.start for stage in schedule] == [22, 66, 154, 330, 682, 1386, 2794, 5610])
    check("EgeExp7 stage budgets are 2^(s-1) K",
          [stage.budget for stage in schedule] == [22 * 2 ** s for s in range(8)])
    check("stages tile [K, B] without gaps",
          all(a.end + 1 == b.start for a, b in zip(schedule, schedule[1:])) and schedule[-1].end == 7000)
    check("no stage before K", stage_at(schedule, 21) is None)
    check("stage boundaries hold the previous completed run",
          stage_at(schedule, 22).index == 1 and stage_at(schedule, 65).index == 1
          and stage_at(schedule, 66).index == 2 and stage_at(schedule, 7000).index == 8)
    check("stage lookup equals the validator's integer formula on every pull",
          all(stage_at(schedule, t).index == independent_stage(t, 22) for t in range(22, 7001)))
    grid = [round(k * 7000 / 100) for k in range(101)]
    points = evaluation_points(schedule, grid, 7000)
    expected = sorted({s.start for s in schedule} | {s.start - 1 for s in schedule[1:]}
                      | {g for g in grid if g >= 22} | {7000})
    check("evaluation points contain both sides of every jump and the shared grid", points == expected)
    check("evaluation points start at K and end at B", points[0] == 22 and points[-1] == 7000)
    check("invalid schedules are rejected",
          raises(lambda: doubling_schedule(1, 100), ValueError) and raises(lambda: doubling_schedule(50, 49), ValueError))
    config = load_config()
    for spec in environment_specs(config):
        stages = doubling_schedule(spec["num_arms"], spec["budget"])
        last = stages[-1].start
        check(f"{spec['name']}: final completed stage brackets B",
              last <= spec["budget"] < (2 ** (len(stages) + 1) - 1) * spec["num_arms"])


def test_protection_and_provenance() -> None:
    config = load_config()
    check("output root is config-hash namespaced", default_output_root(config).name.startswith("results_ege_doubling_"))
    for protected in (
        REPO_ROOT / "figures" / "publication" / "publication_jaccard.png",
        REPO_ROOT / config["source_results_dir"] / "EgeExp1.csv",
        REPO_ROOT / "results" / "x.csv",
        REPO_ROOT / "results_p_sweep_baf5064aa5a5" / "x.csv",
    ):
        check(f"refuses to write {protected.relative_to(REPO_ROOT)}",
              raises(lambda p=protected: assert_safe_output(p, config), PermissionError))
    check("allows the new figure directory",
          assert_safe_output(REPO_ROOT / config["figures_dir"] / "x.png", config) is not None)
    check("allows temporary roots", assert_safe_output(Path(tempfile.gettempdir()) / "x", config) is not None)
    verify_source_hashes(config)
    verify_protected_hashes(config)
    check("preregistered sources and publication artifacts match their recorded hashes", True)


def test_actual_draw_accounting() -> None:
    environment = _construct_environment(ENVIRONMENTS["EgeExp7"][0], "gaussian")
    seed_rngs(0, "EgeExp7|gaussian", "env", 0)
    environment.reset()
    original = environment.sample
    for name, function in (("EGE_SH", EGE_SH), ("EGE_SR", EGE_SR)):
        counts = []
        for stage in doubling_schedule(22, 7000):
            drawn = [0]

            def counting(arms, _drawn=drawn):
                _drawn[0] += len(arms)
                return original(arms)

            environment.sample = counting
            seed_rngs(0, "EgeExp7|gaussian", name, 0, "budget", stage.budget)
            function(T=stage.budget, K=22, D=2, environment=environment)
            counts.append(drawn[0] == expected_ege_draws(name, stage.budget, 22) <= stage.budget)
        environment.sample = original
        check(f"{name}: recorded actual draws equal counted draws and never exceed nominal stage budgets", all(counts))


def _smoke(root: Path, n_jobs: int, environments=SMOKE_ENVIRONMENTS) -> None:
    run(root, environments=environments, num_runs=SMOKE_RUNS, n_jobs=n_jobs, allow_dirty=True)


def _validate(root: Path, **kwargs) -> dict:
    return validate(root, environments=SMOKE_ENVIRONMENTS, num_runs=SMOKE_RUNS, require_clean=False,
                    reproduction=SMOKE_REPRODUCTION, **kwargs)


def test_pipeline_resume_validation_and_plotting() -> None:
    config = load_config()
    with tempfile.TemporaryDirectory() as directory:
        root, serial_root, figures = Path(directory) / "root", Path(directory) / "serial", Path(directory) / "figs"
        _smoke(root, n_jobs=2)
        shards = sorted((root / "stage_runs" / "run" / "shards").rglob("*.csv"))
        stamps = {path: path.stat().st_mtime_ns for path in shards}
        merged = {env: stage_results_path(root, env).read_bytes() for env in SMOKE_ENVIRONMENTS}
        _smoke(root, n_jobs=2)
        check("resume recomputes no complete shard", all(path.stat().st_mtime_ns == stamps[path] for path in shards))
        check("resume reproduces merged stage files byte-for-byte",
              all(stage_results_path(root, env).read_bytes() == merged[env] for env in SMOKE_ENVIRONMENTS))
        _smoke(serial_root, n_jobs=1, environments=["EgeExp7"])
        check("serial and parallel stage runs are byte-identical",
              stage_results_path(serial_root, "EgeExp7").read_bytes() == merged["EgeExp7"])

        with stage_results_path(root, "EgeExp7").open(newline="") as handle:
            stage_row = next(row for row in csv.DictReader(handle)
                             if row["algorithm"] == "EGE_SR" and row["repetition"] == "1" and row["arm_pull"] == "704")
        environment = _construct_environment(ENVIRONMENTS["EgeExp7"][0], "gaussian")
        direct = _run_repetition_ege(environment, "EgeExp7", "gaussian", "EGE_SR", EGE_SR, 7000, [704], 0, 1)[0]
        check("a stage run equals the preregistered runner's fixed-budget output at that budget",
              stage_row["recommendation"] == direct["recommendation"])

        build(root, environments=SMOKE_ENVIRONMENTS, num_runs=SMOKE_RUNS)
        check("render is refused before validation",
              raises(lambda: render(root, environments=SMOKE_ENVIRONMENTS, figures_dir=figures, allow_dirty=True),
                     RuntimeError))
        report = _validate(root)
        check("validator passes clean smoke data", report["status"] == "pass", "; ".join(report["problems"][:5]))
        expected_reproduced = 2 * len(SMOKE_REPRODUCTION["repetitions"]) * len(SMOKE_REPRODUCTION["grid_indices"])
        check("validator reproduces preregistered EGE rows",
              all(summary["reproduced_preregistered_rows"] == expected_reproduced
                  for summary in report["summaries"].values()))

        path = doubling_path(root, "EgeExp7")
        original = path.read_bytes()
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            header, rows = list(reader.fieldnames), list(reader)

        def write_rows(mutated: list[dict]) -> None:
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=header)
                writer.writeheader()
                writer.writerows(mutated)

        def semantic_failure(fragment: str) -> bool:
            result = _validate(root, write_report=False)
            return result["status"] == "fail" and any(fragment in problem for problem in result["problems"])

        write_rows([dict(row, stage=str(int(row["stage"]) + 1)) if i == 5 else row for i, row in enumerate(rows)])
        check("validator rejects a wrong stage index", semantic_failure(": stage="))
        write_rows(rows[:-1] + [dict(rows[-1], recommendation="[0]")])
        check("validator rejects a recommendation that is not held from its stage",
              semantic_failure("recommendation differs from stage"))
        write_rows(rows[:-1])
        check("validator rejects a missing trajectory point", semantic_failure("evaluation points differ"))
        path.write_bytes(original)
        check("validator passes again after restoring data", _validate(root)["status"] == "pass")

        render(root, environments=SMOKE_ENVIRONMENTS, figures_dir=figures, allow_dirty=True)
        report = _validate(root, figures_dir=figures, check_figures=True)
        check("figures keep anytime curves identical to the existing publication tables and D-EGE means exact",
              report["status"] == "pass", "; ".join(report["problems"][:5]))
        check("existing doubling figures are never overwritten implicitly",
              raises(lambda: render(root, environments=SMOKE_ENVIRONMENTS, figures_dir=figures, allow_dirty=True),
                     FileExistsError))
        verify_protected_hashes(config)
        verify_source_hashes(config)
        check("protected sources and publication artifacts are unchanged after the gate", True)
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    execute("Doubling schedule arithmetic", test_schedule_arithmetic)
    execute("Overwrite protection and source provenance", test_protection_and_provenance)
    execute("Native EGE actual-draw accounting", test_actual_draw_accounting)
    execute("Resume, serial/parallel identity, validation, corruption, and plotting",
            test_pipeline_resume_validation_and_plotting)
    print("\n" + "=" * 80)
    if FAILURES:
        print("FAILED:", ", ".join(FAILURES))
        raise SystemExit(1)
    print("ALL TESTS PASSED")
