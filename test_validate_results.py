#!/usr/bin/env python3
"""Standalone regression tests for validate_results.py."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from runner.schema import SCHEMA_COLUMNS

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"
WORK = ROOT / ".validation_test_work"
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


def reset_work() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)


def run_cmd(cmd: list[str], *, expect_ok: bool) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if (proc.returncode == 0) != expect_ok:
        print(proc.stdout)
    return proc


def generate_dataset(path: Path, algorithms: str = "EGE_SR,EGE_SH,Uniform") -> Path:
    cmd = [str(PYTHON), "momab_main.py",
           "--algorithms", algorithms,
           "--environments", "EgeExp7",
           "--budgets", "12",
           "--eval_points", "3",
           "--num_runs", "2",
           "--seed", "0",
           "--results_dir", str(path)]
    proc = run_cmd(cmd, expect_ok=True)
    if proc.returncode != 0:
        raise RuntimeError("dataset generation failed")
    return path


def validator(path: Path, *extra: str, expect_ok: bool) -> subprocess.CompletedProcess:
    cmd = [str(PYTHON), "validate_results.py", str(path), "--python", str(PYTHON), *extra]
    return run_cmd(cmd, expect_ok=expect_ok)


def read_rows(path: Path) -> list[dict[str, str]]:
    with (path / "EgeExp7.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with (path / "EgeExp7.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def copy_case(name: str) -> Path:
    dst = WORK / name
    shutil.copytree(WORK / "good", dst)
    return dst


def assert_validator_catches(case: str, phrase: str, mutator) -> None:
    path = copy_case(case)
    mutator(path)
    proc = validator(path, "--skip-ege-invariance", expect_ok=False)
    check(f"{case} is rejected", proc.returncode != 0)
    check(f"{case} reports {phrase!r}", phrase in proc.stdout, proc.stdout[-1000:])


def test_happy_path_real_dataset() -> None:
    reset_work()
    generate_dataset(WORK / "good")
    proc = validator(WORK / "good", expect_ok=True)
    check("happy-path real dataset validates", proc.returncode == 0)
    check("happy path runs EGE invariance", "EGE grid invariance" in proc.stdout)


def test_missing_repetition() -> None:
    def mutate(path: Path) -> None:
        rows = [r for r in read_rows(path) if not (r["algorithm"] == "Uniform" and r["repetition"] == "1")]
        write_rows(path, rows)
    assert_validator_catches("missing_repetition", "repetitions", mutate)


def test_missing_grid_point() -> None:
    def mutate(path: Path) -> None:
        rows = [r for r in read_rows(path) if not (r["algorithm"] == "EGE_SR" and r["repetition"] == "0" and r["arm_pull"] == "4")]
        write_rows(path, rows)
    assert_validator_catches("missing_grid_point", "grid", mutate)


def test_duplicate_key() -> None:
    def mutate(path: Path) -> None:
        rows = read_rows(path)
        rows.append(dict(rows[0]))
        write_rows(path, rows)
    assert_validator_catches("duplicate_key", "duplicate key", mutate)


def test_out_of_range_metric() -> None:
    def mutate(path: Path) -> None:
        rows = read_rows(path)
        rows[0]["jaccard"] = "1.5"
        write_rows(path, rows)
    assert_validator_catches("out_of_range_metric", "outside finite [0,1]", mutate)


def test_tampered_metric() -> None:
    def mutate(path: Path) -> None:
        rows = read_rows(path)
        rows[0]["bernoulli"] = "0" if rows[0]["bernoulli"] == "1" else "1"
        write_rows(path, rows)
    assert_validator_catches("tampered_metric", "recomputed", mutate)


def test_mismatched_instance_hash() -> None:
    def mutate(path: Path) -> None:
        rows = read_rows(path)
        for r in rows:
            if r["algorithm"] == "Uniform":
                r["instance_hash"] = "badbadbadbadbad1"
                break
        write_rows(path, rows)
    assert_validator_catches("mismatched_instance_hash", "instance_hash", mutate)


def test_mixed_manifest() -> None:
    def mutate(path: Path) -> None:
        source = next((path / "run" / "manifests").glob("*.json"))
        original = json.loads(source.read_text())
        mixed_dir = path / "other_run" / "manifests"
        mixed_dir.mkdir(parents=True)
        original["num_runs"] = 99
        (mixed_dir / "EgeExpX.json").write_text(json.dumps(original, indent=2, sort_keys=True) + "\n")
    assert_validator_catches("mixed_manifest", "mixed manifests", mutate)


def test_clobbered_manifest_hides_environments() -> None:
    """A shared run/ directory written by one array task per environment.

    Every task writes run/manifest.json, so the last finisher clobbers the
    others. Validation must not silently cover only the surviving environment
    while other result files sit unchecked.
    """
    def mutate(path: Path) -> None:
        manifests = path / "run" / "manifests"
        keep = sorted(manifests.glob("*.json"))[0]
        for extra in sorted(manifests.glob("*.json"))[1:]:
            extra.unlink()
        # Simulate the surviving combined manifest naming only one environment.
        data = json.loads(keep.read_text())
        (path / "run" / "manifest.json").write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n")
        # A second environment's results exist but are now undeclared.
        shutil.copy(path / "EgeExp7.csv", path / "EgeExp2.csv")
    assert_validator_catches("clobbered_manifest", "not described by any manifest", mutate)


if __name__ == "__main__":
    run("happy path on generated real dataset", test_happy_path_real_dataset)
    run("detect missing repetition", test_missing_repetition)
    run("detect missing grid point", test_missing_grid_point)
    run("detect duplicate key", test_duplicate_key)
    run("detect out-of-range metric", test_out_of_range_metric)
    run("detect tampered metric", test_tampered_metric)
    run("detect mismatched instance hash", test_mismatched_instance_hash)
    run("detect mixed manifest", test_mixed_manifest)
    run("detect clobbered manifest hiding environments", test_clobbered_manifest_hides_environments)

    print("\n" + "=" * 80)
    if WORK.exists():
        shutil.rmtree(WORK)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("ALL TESTS PASSED")
    sys.exit(0)
