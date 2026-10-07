#!/usr/bin/env python3
"""Independent validator for the information-matched data and figures.

* APE-UV: exact coverage, row bindings and metrics, plus bitwise re-runs of a
  deterministic sample of repetitions with a directly constructed bandit.
* Empirical-mean D-EGE: every completed stage must reproduce the stored
  naive stage recommendation; draw counts, stage clock, evaluation points and
  cumulative sample counts are recomputed from integer arithmetic; a
  deterministic sample of trajectories is recomputed pull by pull with an
  independent Pareto implementation (``paretoset``) in both pull orders.
* Plot inputs must contain every reused row verbatim.
* Figures: reused curves must equal the existing publication and doubling
  plot tables, new curves must equal means recomputed from the trajectories.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from paretoset import paretoset

from bandits.APE_UV import APEUnknownVarianceBandit
from momab_main import EGE_ALGORITHMS, ENVIRONMENTS, _construct_environment, _run_repetition_anytime
from plotting import bounded_mean_interval, wilson_interval
from runner.ege_allocation import expected_ege_draws
from runner.instance import compute_instance_hash
from runner.schema import SCHEMA_COLUMNS, encode_int_set
from runner.seeding import seed_rngs

from .build import ALL_ALGORITHMS, PLOT_COLUMNS, build_manifest_path, plot_input_path
from .config import (
    EMPIRICAL_ALGORITHMS,
    METRICS,
    NAIVE_ALGORITHMS,
    NEW_ANYTIME_ALGORITHM,
    PREREGISTERED_ALGORITHMS,
    REPO_ROOT,
    config_hash,
    default_output_root,
    doubling_dir,
    environment_specs,
    load_config,
    preregistered_dir,
    sha256_file,
    verify_protected,
    verify_sources,
)
from .fileio import atomic_json, read_csv
from .run_ape_uv import manifest_path as ape_uv_manifest_path
from .run_ape_uv import results_path as ape_uv_results_path
from .run_empirical import STAGE_COLUMNS, TRAJECTORY_COLUMNS
from .run_empirical import manifest_path as empirical_manifest_path
from .run_empirical import results_path as empirical_results_path

MAX_REPORTED = 200


def report_path(output_root: Path) -> Path:
    return Path(output_root) / "validation_report.json"


def figure_paths(figures_dir: Path, set_name: str, metric: str) -> tuple[Path, Path]:
    stem = Path(figures_dir) / f"publication_{metric}_{set_name}"
    return stem.with_suffix(".png"), Path(f"{stem}_plot_table.csv")


def figure_manifest_path(figures_dir: Path) -> Path:
    return Path(figures_dir) / "figure_manifest.json"


class Problems(list):
    def add(self, message: str) -> None:
        self.append(message)


def _live_environment(spec: dict, config: dict):
    environment = _construct_environment(ENVIRONMENTS[spec["name"]][0], config["dist"])
    seed_rngs(int(config["master_seed"]), f"{spec['name']}|{config['dist']}", "env", 0)
    environment.reset()
    return environment


def _metric_problem(row: dict, environment) -> str | None:
    recommendation = sorted(json.loads(row["recommendation"]))
    expected = {
        "bernoulli": float(environment.bernoulli_metric(recommendation)),
        "jaccard": float(environment.jaccard_metric(recommendation)),
        "misidentification": float(environment.mis_id_metric(recommendation)),
    }
    for metric, value in expected.items():
        if abs(float(row[metric]) - value) > 1e-12:
            return f"{metric} stored={row[metric]} recomputed={value}"
    return None


def _check_manifest(path: Path, digest: str, require_clean: bool, name: str, problems: Problems) -> dict:
    with Path(path).open() as handle:
        manifest = json.load(handle)
    if manifest.get("config_hash") != digest:
        problems.add(f"{name}: manifest config hash mismatch ({path.name})")
    if require_clean and manifest.get("source_dirty"):
        problems.add(f"{name}: produced from an uncommitted source tree ({path.parent.parent.name})")
    return manifest


def endpoints(num_arms: int, budget: int) -> list[int]:
    out, s = [], 1
    while (2**s - 1) * num_arms <= budget:
        out.append((2**s - 1) * num_arms)
        s += 1
    return out


# --------------------------------------------------------------------------- APE-UV
def validate_ape_uv(output_root: Path, spec: dict, config: dict, repetitions: list[int], require_clean: bool,
                    samples: dict, problems: Problems) -> dict:
    name, digest = spec["name"], config_hash(config)
    manifest = _check_manifest(ape_uv_manifest_path(output_root, name), digest, require_clean, name, problems)
    path = ape_uv_results_path(output_root, name)
    if manifest["result_file"]["sha256"] != sha256_file(path):
        problems.add(f"{name}: APE-UV results differ from their manifest")
    columns, rows = read_csv(path)
    if columns != SCHEMA_COLUMNS:
        problems.add(f"{name}: APE-UV columns differ from the runner schema")
    environment = _live_environment(spec, config)
    true_pareto = encode_int_set(environment.get_top_arms())
    encoded_grid = encode_int_set(spec["grid"])
    keys = set()
    for row in rows:
        key = (int(row["repetition"]), int(row["arm_pull"]))
        if key in keys:
            problems.add(f"{name}: duplicate APE-UV row {key}")
        keys.add(key)
        expected = {"environment": name, "algorithm": NEW_ANYTIME_ALGORITHM, "instance_hash": spec["instance_hash"],
                    "max_budget": str(spec["budget"]), "evaluation_grid": encoded_grid, "posterior_model": "",
                    "rho": "", "master_seed": str(config["master_seed"]), "true_pareto": true_pareto}
        for field, value in expected.items():
            if row[field] != value:
                problems.add(f"{name}/APE_UV/{key}: {field}={row[field]!r}, expected {value!r}")
                break
        problem = _metric_problem(row, environment)
        if problem:
            problems.add(f"{name}/APE_UV/{key}: {problem}")
    if keys != {(rep, t) for rep in repetitions for t in spec["grid"]}:
        problems.add(f"{name}: APE-UV coverage mismatch")
    reran = 0
    if name in samples["environments"]:
        index = {(int(r["repetition"]), int(r["arm_pull"])): r for r in rows}
        for repetition in samples["repetitions"]:
            if repetition not in repetitions:
                continue
            fresh = _construct_environment(ENVIRONMENTS[name][0], config["dist"])
            bandit = APEUnknownVarianceBandit(fresh.num_arms, fresh.num_objectives, delta=0.1, num_warmup_pulls=4)
            for row in _run_repetition_anytime(fresh, name, config["dist"], NEW_ANYTIME_ALGORITHM, bandit,
                                               spec["budget"], spec["grid"], int(config["master_seed"]), repetition):
                reran += 1
                if index[(repetition, row["arm_pull"])]["recommendation"] != row["recommendation"]:
                    problems.add(f"{name}/APE_UV/rep{repetition}/t={row['arm_pull']}: re-run differs")
    return {"rows": len(rows), "reran_rows": reran, "sha256": sha256_file(path)}


# ------------------------------------------------------------------ empirical D-EGE
def _independent_trajectory(spec: dict, config: dict, base: str, repetition: int, points: list[int],
                            order: str, pool_samples: bool = True) -> list[tuple[int, str, int, int]]:
    """Pull-by-pull reference: own recorder, explicit loops, ``paretoset`` Pareto sets."""
    name, K, B = spec["name"], spec["num_arms"], spec["budget"]
    env_tag, master = f"{name}|{config['dist']}", int(config["master_seed"])
    environment = _construct_environment(ENVIRONMENTS[name][0], config["dist"])
    seed_rngs(master, env_tag, "env", repetition)
    environment.reset()
    D = environment.num_objectives
    stage_ends = endpoints(K, B)
    budgets = [2 ** (s - 1) * K for s in range(1, len(stage_ends) + 2)]
    starts = [0, *stage_ends]
    limits = [*stage_ends, B]
    timeline: list[tuple[int, int, np.ndarray]] = []

    class Recorder:
        def __init__(self):
            self.calls = []

        def sample(self, arms):
            rewards = environment.sample(arms)
            self.calls.append((list(np.asarray(arms).tolist()), np.array(rewards, dtype=float)))
            return rewards

    for budget, start, limit in zip(budgets, starts, limits):
        seed_rngs(master, env_tag, base, repetition, "budget", budget)
        recorder = Recorder()
        EGE_ALGORITHMS[base](T=budget, K=K, D=D, environment=recorder)
        sequence = []
        for arms, rewards in recorder.calls:
            if order == "block":
                sequence.extend(zip(arms, rewards))
                continue
            active, per_arm = [], {}
            for arm, reward in zip(arms, rewards):
                if arm not in per_arm:
                    active.append(arm)
                    per_arm[arm] = []
                per_arm[arm].append(reward)
            for k in range(len(per_arm[active[0]])):
                for arm in active:
                    sequence.append((arm, per_arm[arm][k]))
        for offset, (arm, reward) in enumerate(sequence[: limit - start]):
            timeline.append((start + 1 + offset, arm, reward))

    sums = np.zeros((K, D))
    counts = np.zeros(K, dtype=int)
    out, position, previous_start = [], 0, 0
    for t in points:
        if not pool_samples:
            current_start = max((end for end in stage_ends if end < t), default=0)
            if current_start != previous_start:
                sums.fill(0.0)
                counts.fill(0)
                while position < len(timeline) and timeline[position][0] <= current_start:
                    position += 1
                previous_start = current_start
        while position < len(timeline) and timeline[position][0] <= t:
            _, arm, reward = timeline[position]
            sums[arm] += reward
            counts[arm] += 1
            position += 1
        means = np.zeros((K, D))
        for arm in range(K):
            if counts[arm]:
                means[arm] = sums[arm] / counts[arm]
        mask = paretoset(means, sense=["max"] * D, distinct=False)
        out.append((t, encode_int_set(np.flatnonzero(mask)), position, int(np.sum(counts == 0))))
    return out


def validate_empirical(output_root: Path, spec: dict, config: dict, repetitions: list[int], require_clean: bool,
                       samples: dict, problems: Problems) -> dict:
    name, K, B, digest = spec["name"], spec["num_arms"], spec["budget"], config_hash(config)
    manifest = _check_manifest(empirical_manifest_path(output_root, name), digest, require_clean, name, problems)
    files = {kind: empirical_results_path(output_root, name, kind) for kind in ("trajectory", "block", "stages")}
    for kind, path in files.items():
        if manifest["files"][kind]["sha256"] != sha256_file(path):
            problems.add(f"{name}: empirical {kind} file differs from its manifest")
    environment = _live_environment(spec, config)
    if compute_instance_hash(environment) != spec["instance_hash"]:
        problems.add(f"{name}: live instance differs from preregistration")
    true_pareto = encode_int_set(environment.get_top_arms())
    stage_ends = endpoints(K, B)
    S = len(stage_ends)
    points = sorted(set(spec["grid"]) | set(stage_ends))
    if config["empirical_doubling"].get("sample_memory", "all_stages") == "current_stage":
        points = sorted(set(points) | {end + 1 for end in stage_ends if end < B})
    encoded_points = encode_int_set(points)

    # stages
    columns, stage_rows = read_csv(files["stages"])
    if columns != STAGE_COLUMNS:
        problems.add(f"{name}: stage columns differ")
    naive = {(r["algorithm"], int(r["repetition"]), int(r["arm_pull"])): r["recommendation"]
             for r in read_csv(doubling_dir(config) / "stage_runs" / f"{name}.csv")[1]}
    used: dict[tuple[str, int], list[tuple[int, int]]] = {}
    seen, reproduced = set(), 0
    for row in stage_rows:
        algorithm, base, rep, s = row["algorithm"], row["base_algorithm"], int(row["repetition"]), int(row["stage"])
        seen.add((algorithm, rep, s))
        budget = 2 ** (s - 1) * K
        start = (2 ** (s - 1) - 1) * K
        end = B if s == S + 1 else (2**s - 1) * K
        actual = expected_ege_draws(base, budget, K)
        expected = {"base_algorithm": EMPIRICAL_ALGORITHMS.get(algorithm), "stage_budget": str(budget),
                    "stage_start": str(start), "stage_end": str(end), "completed": str(int(s <= S)),
                    "actual_draws": str(actual), "used_draws": str(min(actual, end - start)), "config_hash": digest}
        for field, value in expected.items():
            if row[field] != value:
                problems.add(f"{name}/{algorithm}/rep{rep}/stage{s}: {field}={row[field]!r}, expected {value!r}")
                break
        used.setdefault((algorithm, rep), []).append((start, min(actual, end - start)))
        if s <= S:
            reproduced += 1
            if naive.get((base, rep, budget)) != row["recommendation"]:
                problems.add(f"{name}/{algorithm}/rep{rep}/stage{s}: does not reproduce the naive stage run")
    if seen != {(a, r, s) for a in EMPIRICAL_ALGORITHMS for r in repetitions for s in range(1, S + 2)}:
        problems.add(f"{name}: stage coverage mismatch")

    # trajectories (both pull orders)
    sample_reps = [r for r in samples["repetitions"] if r in repetitions]
    recomputed = 0
    for kind, suffix in (("trajectory", ""), ("block", "_BLOCK")):
        columns, rows = read_csv(files[kind])
        if columns != TRAJECTORY_COLUMNS:
            problems.add(f"{name}: {kind} columns differ")
        trajectories: dict[tuple[str, int], list[dict]] = {}
        for row in rows:
            trajectories.setdefault((row["algorithm"], int(row["repetition"])), []).append(row)
        if set(trajectories) != {(f"{a}{suffix}", r) for a in EMPIRICAL_ALGORITHMS for r in repetitions}:
            problems.add(f"{name}: {kind} trajectory coverage mismatch")
        for (algorithm, rep), trajectory in trajectories.items():
            base_key = algorithm.removesuffix(suffix) if suffix else algorithm
            if [int(r["arm_pull"]) for r in trajectory] != points:
                problems.add(f"{name}/{algorithm}/rep{rep}: evaluation points differ")
                continue
            stages = used.get((base_key, rep), [])
            for row in trajectory:
                t = int(row["arm_pull"])
                drawn = sum(min(count, max(0, t - start)) for start, count in stages)
                expected = {"samples_drawn": str(drawn), "max_budget": str(B), "evaluation_grid": encoded_points,
                            "instance_hash": spec["instance_hash"], "true_pareto": true_pareto,
                            "base_algorithm": EMPIRICAL_ALGORITHMS[base_key], "config_hash": digest}
                for field, value in expected.items():
                    if row[field] != value:
                        problems.add(f"{name}/{algorithm}/rep{rep}/t={t}: {field}={row[field]!r}, expected {value!r}")
                        break
                active = 0 if t == 0 else next(
                    (s for s in range(1, S + 2) if (2 ** (s - 1) - 1) * K < t <= (B if s == S + 1 else (2**s - 1) * K)), -1)
                if row["active_stage"] != str(active):
                    problems.add(f"{name}/{algorithm}/rep{rep}/t={t}: active_stage={row['active_stage']}, expected {active}")
                problem = _metric_problem(row, environment)
                if problem:
                    problems.add(f"{name}/{algorithm}/rep{rep}/t={t}: {problem}")
            if rep in sample_reps:
                order = "block" if suffix else "round_robin"
                reference = _independent_trajectory(
                    spec, config, EMPIRICAL_ALGORITHMS[base_key], rep, points, order,
                    pool_samples=config["empirical_doubling"].get("sample_memory", "all_stages") != "current_stage",
                )
                for row, (t, recommendation, drawn, unsampled) in zip(trajectory, reference):
                    recomputed += 1
                    if (row["recommendation"], row["samples_drawn"], row["unsampled_arms"]) != (
                            recommendation, str(drawn), str(unsampled)):
                        problems.add(f"{name}/{algorithm}/rep{rep}/t={t}: independent recomputation differs")
                        break
    return {"stage_rows": len(stage_rows), "reproduced_naive_stages": reproduced,
            "independently_recomputed_points": recomputed,
            "sha256": {kind: sha256_file(path) for kind, path in files.items()}}


# ------------------------------------------------------------------ plot inputs
def validate_plot_inputs(output_root: Path, spec: dict, config: dict, problems: Problems) -> str:
    name = spec["name"]
    path = plot_input_path(output_root, name)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != PLOT_COLUMNS:
        problems.add(f"{name}: plot input columns differ")
        return sha256_file(path)
    if sorted(frame["algorithm"].unique()) != sorted(ALL_ALGORITHMS):
        problems.add(f"{name}: plot input algorithms differ")
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    pre = read(preregistered_dir(config) / f"{name}.csv")
    naive = read(doubling_dir(config) / "doubling" / f"{name}.csv")
    expected = {
        **{a: pre[pre["algorithm"] == a] for a in PREREGISTERED_ALGORITHMS},
        **{a: naive[naive["algorithm"] == src].assign(algorithm=a) for a, src in NAIVE_ALGORITHMS.items()},
        NEW_ANYTIME_ALGORITHM: read(ape_uv_results_path(output_root, name)),
    }
    trajectory = read(empirical_results_path(output_root, name, "trajectory"))
    expected.update({a: trajectory[trajectory["algorithm"] == a] for a in EMPIRICAL_ALGORITHMS})
    for algorithm, source in expected.items():
        got = frame[frame["algorithm"] == algorithm].reset_index(drop=True)
        want = source.loc[:, PLOT_COLUMNS].reset_index(drop=True)
        if not got.equals(want):
            problems.add(f"{name}/{algorithm}: plot input rows are not verbatim source rows")
    return sha256_file(path)


# ---------------------------------------------------------------------- figures
def validate_figures(output_root: Path, figures_dir: Path, specs: list[dict], config: dict, num_runs: int,
                     problems: Problems, reused_algorithms: tuple[str, ...] = ()) -> dict:
    names = [spec["name"] for spec in specs]
    key = ["environment", "algorithm", "arm_pull"]
    existing = {m: pd.read_csv(REPO_ROOT / "figures/publication" / f"publication_{m}_plot_table.csv") for m in METRICS}
    doubling = {m: pd.read_csv(REPO_ROOT / "figures/publication_ege_doubling" / f"publication_{m}_ege_doubling_plot_table.csv")
                for m in METRICS}
    inputs = {spec["name"]: pd.read_csv(plot_input_path(output_root, spec["name"])) for spec in specs}
    hashes = {}
    for set_name, algorithms in config["figure_sets"].items():
        for metric in METRICS:
            png, table_path = figure_paths(figures_dir, set_name, metric)
            if not png.is_file() or not table_path.is_file():
                problems.add(f"{set_name}/{metric}: missing figure or plot table")
                continue
            hashes[png.name], hashes[table_path.name] = sha256_file(png), sha256_file(table_path)
            table = pd.read_csv(table_path)
            if set(table["algorithm"]) != set(algorithms) or set(table["environment"]) != set(names):
                problems.add(f"{set_name}/{metric}: plotted algorithms or environments are unexpected")
                continue
            for algorithm in algorithms:
                got = table[table["algorithm"] == algorithm].sort_values(key).reset_index(drop=True)
                if algorithm in ("Uniform", "APE", "TTPFTS_UNKNOWN_VAR"):
                    want = existing[metric]
                    want = want[(want["algorithm"] == algorithm) & want["environment"].isin(names)]
                elif algorithm in NAIVE_ALGORITHMS:
                    want = doubling[metric]
                    want = want[(want["algorithm"] == NAIVE_ALGORITHMS[algorithm]) & want["environment"].isin(names)]
                    want = want.assign(algorithm=algorithm)
                else:
                    want = None
                if want is not None:
                    try:
                        pd.testing.assert_frame_equal(got, want.sort_values(key).reset_index(drop=True), check_exact=True)
                    except AssertionError:
                        problems.add(f"{set_name}/{metric}/{algorithm}: curve differs from the existing publication table")
                    continue
                for spec in specs:
                    rows = inputs[spec["name"]]
                    rows = rows[rows["algorithm"] == algorithm]
                    grouped = rows.groupby("arm_pull")[metric]
                    curve = got[got["environment"] == spec["name"]]
                    means = grouped.mean()
                    if list(curve["arm_pull"]) != list(means.index) or not np.allclose(
                            curve["mean"].to_numpy(), means.to_numpy(), rtol=0, atol=1e-12):
                        problems.add(f"{set_name}/{metric}/{spec['name']}/{algorithm}: plotted means differ")
                        continue
                    expected_n = int(config["num_runs"]) if algorithm in (
                        *PREREGISTERED_ALGORITHMS, *reused_algorithms
                    ) else num_runs
                    if set(curve["n"]) != {expected_n} or float(curve["normalized_budget"].iloc[-1]) != 1.0:
                        problems.add(f"{set_name}/{metric}/{spec['name']}/{algorithm}: wrong n or endpoint")
                    for (_, point), values in zip(curve.iterrows(), (g.to_numpy() for _, g in grouped)):
                        if metric == "bernoulli":
                            lower, upper = wilson_interval(int(np.sum(values)), len(values))
                        else:
                            lower, upper = bounded_mean_interval(values)
                        if not (math.isclose(point["lower"], lower, abs_tol=1e-12)
                                and math.isclose(point["upper"], upper, abs_tol=1e-12)):
                            problems.add(f"{set_name}/{metric}/{spec['name']}/{algorithm}: interval differs")
                            break
    manifest = figure_manifest_path(figures_dir)
    if not manifest.is_file():
        problems.add("missing figure manifest")
    else:
        with manifest.open() as handle:
            payload = json.load(handle)
        if payload.get("config_hash") != config_hash(config) or payload.get("files") != hashes:
            problems.add("figure manifest does not match the rendered figures")
    return hashes


def validate(output_root: Path | None = None, config: dict | None = None, environments: list[str] | None = None,
             repetitions: list[int] | None = None, require_clean: bool = True, samples: dict | None = None,
             figures_dir: Path | None = None, check_figures: bool = False, write_report: bool = True) -> dict:
    config = load_config() if config is None else config
    output_root = Path(default_output_root(config) if output_root is None else output_root)
    figures_dir = Path(REPO_ROOT / config["figures_dir"] if figures_dir is None else figures_dir)
    repetitions = list(range(int(config["num_runs"]))) if repetitions is None else list(repetitions)
    samples = samples or config["validation_samples"]
    problems = Problems()
    try:
        verify_sources(config)
        verify_protected(config)
    except ValueError as error:
        problems.add(str(error))
    specs = [s for s in environment_specs(config) if environments is None or s["name"] in environments]
    summaries = {}
    for spec in specs:
        try:
            summaries[spec["name"]] = {
                "ape_uv": validate_ape_uv(output_root, spec, config, repetitions, require_clean,
                                          samples["ape_uv_rerun"], problems),
                "empirical": validate_empirical(output_root, spec, config, repetitions, require_clean,
                                                samples["empirical_recompute"], problems),
                "plot_input_sha256": validate_plot_inputs(output_root, spec, config, problems),
            }
        except (OSError, KeyError, ValueError) as error:
            problems.add(f"{spec['name']}: {type(error).__name__}: {error}")
    try:
        with build_manifest_path(output_root).open() as handle:
            build_manifest = json.load(handle)
        for name, summary in summaries.items():
            if build_manifest["files"][name]["sha256"] != summary["plot_input_sha256"]:
                problems.add(f"{name}: plot input differs from its build manifest")
    except (OSError, KeyError, ValueError) as error:
        problems.add(f"build manifest: {type(error).__name__}: {error}")
    figure_hashes = None
    if check_figures:
        figure_hashes = validate_figures(output_root, figures_dir, specs, config, len(repetitions), problems)
    report = {
        "schema_version": "information-matched-validation-v1",
        "status": "pass" if not problems else "fail",
        "problem_count": len(problems),
        "problems": list(problems[:MAX_REPORTED]),
        "config_hash": config_hash(config),
        "environments": [s["name"] for s in specs],
        "repetitions": repetitions,
        "require_clean": require_clean,
        "samples": samples,
        "summaries": summaries,
        "figures_checked": check_figures,
        "figure_files": figure_hashes,
    }
    if write_report:
        atomic_json(report_path(output_root), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--environments", default=None)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--allow-dirty-source", action="store_true")
    args = parser.parse_args()
    report = validate(
        output_root=Path(args.output_root) if args.output_root else None,
        environments=[e.strip() for e in args.environments.split(",")] if args.environments else None,
        require_clean=not args.allow_dirty_source,
        check_figures=args.figures,
    )
    for problem in report["problems"]:
        print(f"FAIL: {problem}")
    for name, summary in report["summaries"].items():
        print(name, {"ape_uv_rows": summary["ape_uv"]["rows"], "ape_uv_reran": summary["ape_uv"]["reran_rows"],
                     "naive_stages_reproduced": summary["empirical"]["reproduced_naive_stages"],
                     "recomputed_points": summary["empirical"]["independently_recomputed_points"]})
    print(f"Validation {report['status'].upper()} ({report['problem_count']} problem(s))")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
