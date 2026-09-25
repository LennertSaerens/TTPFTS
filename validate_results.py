#!/usr/bin/env python3
"""Phase 8 result-integrity and reproducibility validator for MOMAB results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

from momab_main import ENVIRONMENTS, _construct_environment
from runner.instance import compute_instance_hash
from runner.schema import SCHEMA_COLUMNS

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EGE_ALGS = ["EGE_SR", "EGE_SH"]


class CheckReport:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.passes: list[str] = []

    def pass_(self, name: str, detail: str = "") -> None:
        self.passes.append(name)
        print(f"[PASS] {name}" + (f" -- {detail}" if detail else ""))

    def fail(self, name: str, detail: str) -> None:
        self.failures.append(f"{name}: {detail}")
        print(f"[FAIL] {name} -- {detail}")


def canonical_manifest(data: dict[str, Any]) -> dict[str, Any]:
    ignored = {"created_at", "results_dir", "run_dir", "cli_args", "n_jobs"}
    return {k: data[k] for k in sorted(data) if k not in ignored}


def find_manifests(results_dir: Path) -> list[Path]:
    return sorted(p for p in results_dir.rglob("manifest.json") if p.is_file())


def load_manifest(results_dir: Path, report: CheckReport) -> dict[str, Any] | None:
    manifests = find_manifests(results_dir)
    if not manifests:
        report.fail("manifest agreement", f"no manifest.json found under {results_dir}")
        return None
    loaded = []
    for path in manifests:
        try:
            loaded.append((path, json.loads(path.read_text())))
        except Exception as exc:
            report.fail("manifest agreement", f"cannot parse {path}: {exc}")
            return None
    canonical = canonical_manifest(loaded[0][1])
    mixed = [str(path.relative_to(results_dir)) for path, data in loaded
             if canonical_manifest(data) != canonical]
    if mixed:
        report.fail("manifest agreement", "mixed manifests detected: " + ", ".join(mixed))
    else:
        report.pass_("manifest agreement", f"{len(loaded)} consistent manifest(s)")
    return loaded[0][1]


def parse_int_set(encoded: Any, *, field: str, num_arms: int | None) -> tuple[set[int] | None, str | None]:
    try:
        values = json.loads(encoded)
    except Exception as exc:
        return None, f"{field} is not valid JSON: {exc}"
    if not isinstance(values, list):
        return None, f"{field} must encode a JSON list"
    out: list[int] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, int):
            return None, f"{field} contains non-integer value {v!r}"
        if v < 0:
            return None, f"{field} contains negative arm {v}"
        if num_arms is not None and v >= num_arms:
            return None, f"{field} contains arm {v} outside [0,{num_arms - 1}]"
        out.append(v)
    if out != sorted(out):
        return None, f"{field} must be sorted for canonical set encoding"
    if len(out) != len(set(out)):
        return None, f"{field} contains duplicate arms"
    return set(out), None


def independent_metrics(recommendation: set[int], true_pareto: set[int], num_arms: int) -> dict[str, float]:
    union = recommendation | true_pareto
    symmetric_diff = recommendation.symmetric_difference(true_pareto)
    return {
        "bernoulli": float(int(recommendation == true_pareto)),
        "jaccard": float(len(recommendation & true_pareto) / len(union) if union else 1.0),
        "misidentification": float(len(symmetric_diff) / num_arms if num_arms else 0.0),
    }


def read_rows(results_dir: Path, manifest: dict[str, Any], report: CheckReport) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for env in manifest.get("environments", []):
        path = results_dir / f"{env}.csv"
        if not path.exists():
            report.fail("load merged data", f"missing {path.name}")
            continue
        try:
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != SCHEMA_COLUMNS:
                    report.fail("load merged data", f"{path.name} schema mismatch: {reader.fieldnames}")
                    continue
                env_rows = list(reader)
        except Exception as exc:
            report.fail("load merged data", f"cannot read {path.name}: {exc}")
            continue
        rows.extend(env_rows)
    if rows:
        report.pass_("load merged data", f"{len(rows)} rows")
    return rows


def expected_env_info(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    dist = manifest.get("dist", "gaussian")
    for env_name in manifest.get("environments", []):
        if env_name not in ENVIRONMENTS:
            continue
        env_cls, _ = ENVIRONMENTS[env_name]
        env = _construct_environment(env_cls, dist)
        info[env_name] = {
            "num_arms": int(env.num_arms),
            "true_pareto": set(int(v) for v in env.get_top_arms()),
            "instance_hash": compute_instance_hash(env),
        }
    return info


def check_completeness_and_uniqueness(rows: list[dict[str, str]], manifest: dict[str, Any], report: CheckReport) -> None:
    problems: list[str] = []
    algorithms = list(manifest.get("algorithms", []))
    environments = list(manifest.get("environments", []))
    n = int(manifest.get("num_runs", -1))
    grids = {e: [int(x) for x in manifest.get("evaluation_grids", {}).get(e, [])] for e in environments}
    budgets = {e: int(manifest.get("budgets", {}).get(e, -1)) for e in environments}
    expected_seed = "" if manifest.get("master_seed") is None else str(int(manifest.get("master_seed")))
    expected_rho = str(manifest.get("rho"))

    seen: set[tuple[str, str, int, int]] = set()
    by_ea: dict[tuple[str, str], dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows, start=2):
        try:
            env, alg = row["environment"], row["algorithm"]
            rep, pull = int(row["repetition"]), int(row["arm_pull"])
        except Exception as exc:
            problems.append(f"row {i}: malformed key fields: {exc}")
            continue
        if env not in environments:
            problems.append(f"row {i}: environment {env!r} not listed in manifest")
        if alg not in algorithms:
            problems.append(f"row {i}: algorithm {alg!r} not listed in manifest")
        if row.get("master_seed", "") != expected_seed:
            problems.append(f"row {i}: master_seed {row.get('master_seed')!r} != manifest {expected_seed!r}")
        if row.get("rho", "") not in {"", expected_rho}:
            problems.append(f"row {i}: rho {row.get('rho')!r} != manifest {expected_rho!r}")
        key = (env, alg, rep, pull)
        if key in seen:
            problems.append(f"duplicate key {key}")
        seen.add(key)
        by_ea[(env, alg)][rep].append(row)

    for env in environments:
        grid = grids.get(env, [])
        if not grid or grid[0] != 0 or grid[-1] != budgets.get(env):
            problems.append(f"manifest grid for {env} does not include endpoints 0 and max_budget")
        encoded_grid = json.dumps(grid, separators=(",", ":"))
        for alg in algorithms:
            reps = by_ea.get((env, alg), {})
            actual_reps = sorted(reps)
            expected_reps = list(range(n))
            if actual_reps != expected_reps:
                problems.append(f"{env}/{alg}: repetitions {actual_reps}, expected {expected_reps}")
            for rep in expected_reps:
                rep_rows = reps.get(rep, [])
                pulls = sorted(int(r["arm_pull"]) for r in rep_rows if str(r.get("arm_pull", "")).lstrip("-").isdigit())
                if pulls != grid:
                    problems.append(f"{env}/{alg}/rep{rep}: grid {pulls}, expected {grid}")
                for r in rep_rows:
                    if r.get("evaluation_grid") != encoded_grid:
                        problems.append(f"{env}/{alg}/rep{rep}/t={r.get('arm_pull')}: evaluation_grid mismatch")
                    try:
                        if int(r.get("max_budget", -1)) != budgets[env]:
                            problems.append(f"{env}/{alg}/rep{rep}/t={r.get('arm_pull')}: max_budget mismatch")
                    except ValueError:
                        problems.append(f"{env}/{alg}/rep{rep}/t={r.get('arm_pull')}: malformed max_budget")

    expected_count = len(environments) * len(algorithms) * n * sum(0 for _ in [0])
    expected_count = sum(len(grids[e]) * len(algorithms) * n for e in environments)
    if len(rows) != expected_count:
        problems.append(f"row count {len(rows)}, expected {expected_count}")

    if problems:
        report.fail("completeness and uniqueness", "; ".join(problems[:20]) + (" ..." if len(problems) > 20 else ""))
    else:
        report.pass_("completeness and uniqueness", f"{expected_count} expected rows present")


def check_metrics_and_row_values(rows: list[dict[str, str]], info: dict[str, dict[str, Any]], report: CheckReport) -> None:
    problems: list[str] = []
    for idx, row in enumerate(rows, start=2):
        env = row.get("environment", "")
        if env not in info:
            problems.append(f"row {idx}: unknown environment {env!r}")
            continue
        num_arms = info[env]["num_arms"]
        true_pareto = info[env]["true_pareto"]
        rec, err = parse_int_set(row.get("recommendation"), field="recommendation", num_arms=num_arms)
        if err:
            problems.append(f"row {idx}: {err}")
            continue
        stored_true, err = parse_int_set(row.get("true_pareto"), field="true_pareto", num_arms=num_arms)
        if err:
            problems.append(f"row {idx}: {err}")
            continue
        if stored_true != true_pareto:
            problems.append(f"row {idx}: stored true_pareto {sorted(stored_true)} != frozen {sorted(true_pareto)}")
        expected = independent_metrics(rec or set(), true_pareto, num_arms)
        for key in ("bernoulli", "jaccard", "misidentification"):
            try:
                stored = float(row[key])
            except Exception:
                problems.append(f"row {idx}: {key} is not numeric")
                continue
            if not math.isfinite(stored) or not (0.0 <= stored <= 1.0):
                problems.append(f"row {idx}: {key}={stored} outside finite [0,1]")
                continue
            if stored != expected[key]:
                problems.append(f"row {idx}: {key} stored={stored!r} recomputed={expected[key]!r}")
    if problems:
        report.fail("row values and independent metrics", "; ".join(problems[:20]) + (" ..." if len(problems) > 20 else ""))
    else:
        report.pass_("row values and independent metrics", f"{len(rows)} rows cross-checked")


def check_instance_agreement(rows: list[dict[str, str]], manifest: dict[str, Any], info: dict[str, dict[str, Any]], report: CheckReport) -> None:
    problems: list[str] = []
    rows_by_env_alg: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        rows_by_env_alg[(row.get("environment", ""), row.get("algorithm", ""))].add(row.get("instance_hash", ""))
    for env in manifest.get("environments", []):
        env_hashes: set[str] = set()
        for alg in manifest.get("algorithms", []):
            hashes = rows_by_env_alg.get((env, alg), set())
            if len(hashes) != 1:
                problems.append(f"{env}/{alg}: expected one instance_hash, got {sorted(hashes)}")
            env_hashes.update(hashes)
        if len(env_hashes) != 1:
            problems.append(f"{env}: algorithms disagree on instance_hash {sorted(env_hashes)}")
        expected_hash = info.get(env, {}).get("instance_hash")
        if expected_hash is not None and env_hashes and env_hashes != {expected_hash}:
            problems.append(f"{env}: instance_hash {sorted(env_hashes)} != environment manifest/hash {expected_hash}")
    frozen_manifest_path = REPO_ROOT / "environments" / "instances" / "manifest.json"
    if frozen_manifest_path.exists():
        frozen = json.loads(frozen_manifest_path.read_text()).get("environments", {})
        for env in manifest.get("environments", []):
            if env in frozen and env not in info:
                problems.append(f"{env}: listed in frozen instance manifest but cannot construct environment")
    if problems:
        report.fail("instance agreement", "; ".join(problems[:20]) + (" ..." if len(problems) > 20 else ""))
    else:
        report.pass_("instance agreement", "one hash per environment and matches frozen instance where applicable")


def rows_digest(path: Path, shared_pulls: set[int]) -> dict[tuple[str, int, int], str]:
    out: dict[tuple[str, int, int], str] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            pull = int(row["arm_pull"])
            if pull not in shared_pulls:
                continue
            row_for_compare = dict(row)
            # The provenance grid necessarily differs between dense and sparse runs;
            # all algorithm outputs and metric bytes must be identical.
            row_for_compare.pop("evaluation_grid", None)
            key = (row["algorithm"], int(row["repetition"]), pull)
            payload = json.dumps(row_for_compare, sort_keys=True, separators=(",", ":"))
            out[key] = hashlib.sha256(payload.encode()).hexdigest()
    return out


def run_ege_grid_invariance(report: CheckReport, python: str, keep_work: bool = False) -> None:
    work = REPO_ROOT / ".validation_ege_invariance_work"
    if work.exists():
        shutil.rmtree(work)
    dense = work / "dense"
    sparse = work / "sparse"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    base = [python, str(REPO_ROOT / "momab_main.py"),
            "--algorithms", ",".join(DEFAULT_EGE_ALGS),
            "--environments", "EgeExp7",
            "--budgets", "12",
            "--num_runs", "2",
            "--seed", "0",
            "--n_jobs", "1"]
    try:
        for out_dir, points in [(sparse, "3"), (dense, "6")]:
            cmd = base + ["--eval_points", points, "--results_dir", str(out_dir)]
            subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        sparse_grid = json.loads((sparse / "run" / "manifest.json").read_text())["evaluation_grids"]["EgeExp7"]
        dense_grid = json.loads((dense / "run" / "manifest.json").read_text())["evaluation_grids"]["EgeExp7"]
        shared = set(map(int, sparse_grid)) & set(map(int, dense_grid))
        sd = rows_digest(sparse / "EgeExp7.csv", shared)
        dd = rows_digest(dense / "EgeExp7.csv", shared)
        if sd == dd and sd:
            report.pass_("EGE grid invariance", f"{len(sd)} shared rows match for pulls {sorted(shared)}")
        else:
            missing = sorted(set(sd) ^ set(dd))[:10]
            mismatched = sorted(k for k in set(sd) & set(dd) if sd[k] != dd[k])[:10]
            report.fail("EGE grid invariance", f"missing={missing}, mismatched={mismatched}")
    except Exception as exc:
        report.fail("EGE grid invariance", str(exc))
    finally:
        if work.exists() and not keep_work:
            shutil.rmtree(work)


def print_summary(rows: list[dict[str, str]], manifest: dict[str, Any]) -> None:
    print("\nSUMMARY: endpoint means and uncertainty (mean ± 1.96*SE over all repetitions)")
    metrics = ["bernoulli", "jaccard", "misidentification"]
    for env in manifest.get("environments", []):
        max_budget = int(manifest.get("budgets", {}).get(env, -1))
        for alg in manifest.get("algorithms", []):
            subset = [r for r in rows if r.get("environment") == env and r.get("algorithm") == alg
                      and int(r.get("arm_pull", -1)) == max_budget]
            reps = sorted({int(r["repetition"]) for r in subset})
            pieces = []
            for metric in metrics:
                vals = [float(r[metric]) for r in subset]
                if not vals:
                    pieces.append(f"{metric}=NA")
                    continue
                se = (stdev(vals) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
                pieces.append(f"{metric}={mean(vals):.6g}±{1.96 * se:.3g}")
            print(f"  {env}/{alg} endpoint={max_budget} n={len(reps)}: " + ", ".join(pieces))


def validate(results_dir: Path, *, python: str, skip_ege_invariance: bool, summary_only: bool, keep_work: bool) -> int:
    report = CheckReport()
    manifest = load_manifest(results_dir, report)
    if manifest is None:
        return 1
    rows = read_rows(results_dir, manifest, report)
    if not rows:
        return 1
    info = expected_env_info(manifest)
    if not summary_only:
        check_completeness_and_uniqueness(rows, manifest, report)
        check_metrics_and_row_values(rows, info, report)
        check_instance_agreement(rows, manifest, info, report)
        if skip_ege_invariance:
            report.pass_("EGE grid invariance", "skipped by CLI")
        else:
            run_ege_grid_invariance(report, python=python, keep_work=keep_work)
    print_summary(rows, manifest)
    if report.failures:
        print("\nVALIDATION FAILED")
        return 1
    print("\nVALIDATION PASSED")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--python", default=str(REPO_ROOT / ".venv" / "bin" / "python"),
                        help="Python executable used for the EGE invariance smoke run")
    parser.add_argument("--skip-ege-invariance", action="store_true",
                        help="Skip the runner re-execution check (for focused fixture tests only)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Only print endpoint summaries after loading a consistent manifest/data set")
    parser.add_argument("--keep-work", action="store_true",
                        help="Keep validator-generated reproducibility work directories for debugging")
    args = parser.parse_args(argv)
    return validate(args.results_dir.resolve(), python=args.python,
                    skip_ege_invariance=args.skip_ege_invariance,
                    summary_only=args.summary_only, keep_work=args.keep_work)


if __name__ == "__main__":
    sys.exit(main())
