#!/usr/bin/env python3
"""Independent validator for the doubling-trick EGE data and figures.

The stage index is recomputed here from integer arithmetic,
``s = bit_length(t // K + 1) - 1``, rather than through ``schedule.py``, so a
schedule bug cannot validate itself. Metrics are recomputed with the live
environment's own metric methods, and a deterministic sample of preregistered
EGE rows is re-run locally to prove that the stage runs share the exact
algorithm/seed contract of the published fixed-budget curves.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from momab_main import EGE_ALGORITHMS, ENVIRONMENTS, _construct_environment, _run_repetition_ege
from runner.ege_allocation import expected_ege_draws
from runner.instance import compute_instance_hash
from runner.schema import SCHEMA_COLUMNS, encode_int_set
from runner.seeding import seed_rngs

from .build import (
    DOUBLING_COLUMNS,
    PLOT_COLUMNS,
    build_manifest_path,
    doubling_path,
    plot_input_path,
)
from .config import (
    ANYTIME_ALGORITHMS,
    DOUBLING_ALGORITHMS,
    METRICS,
    REPO_ROOT,
    config_hash,
    default_output_root,
    environment_specs,
    load_config,
    sha256_file,
    source_results_dir,
    verify_protected_hashes,
    verify_source_hashes,
)
from .run import stage_manifest_path, stage_results_path, write_json_atomic


REFERENCE_ENDPOINTS = {"EgeExp7": [22, 66, 154, 330, 682, 1386, 2794, 5610]}
MAX_REPORTED_PROBLEMS = 200


def report_path(output_root: Path) -> Path:
    return Path(output_root) / "validation_report.json"


def figure_paths(figures_dir: Path, metric: str) -> tuple[Path, Path]:
    stem = Path(figures_dir) / f"publication_{metric}_ege_doubling"
    return stem.with_suffix(".png"), Path(f"{stem}_plot_table.csv")


def figure_manifest_path(figures_dir: Path) -> Path:
    return Path(figures_dir) / "figure_manifest.json"


def independent_stage(arm_pull: int, num_arms: int) -> int:
    """Largest s with (2**s - 1) K <= t, i.e. 2**s <= t // K + 1."""
    return (arm_pull // num_arms + 1).bit_length() - 1


def independent_endpoints(num_arms: int, budget: int) -> list[int]:
    endpoints, s = [], 1
    while (2**s - 1) * num_arms <= budget:
        endpoints.append((2**s - 1) * num_arms)
        s += 1
    return endpoints


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


class Problems(list):
    def add(self, message: str) -> None:
        self.append(message)


def _metric_mismatch(row: dict, environment) -> str | None:
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


def validate_environment(
    output_root: Path,
    spec: dict,
    config: dict,
    algorithms: list[str],
    num_runs: int,
    require_clean: bool,
    reproduction: dict,
    problems: Problems,
) -> dict:
    name, num_arms, budget = spec["name"], spec["num_arms"], spec["budget"]
    digest = config_hash(config)
    dist, master_seed = config["dist"], int(config["master_seed"])
    bases = {algorithm: DOUBLING_ALGORITHMS[algorithm] for algorithm in algorithms}

    endpoints = independent_endpoints(num_arms, budget)
    stage_count = len(endpoints)
    budgets = [2 ** (s - 1) * num_arms for s in range(1, stage_count + 1)]
    if not endpoints or not endpoints[-1] <= budget < (2 ** (stage_count + 1) - 1) * num_arms:
        problems.add(f"{name}: final completed stage does not bracket B")
    if [sum(budgets[:s]) for s in range(1, stage_count + 1)] != endpoints:
        problems.add(f"{name}: stage endpoints are not cumulative stage budgets")
    if name in REFERENCE_ENDPOINTS and endpoints != REFERENCE_ENDPOINTS[name]:
        problems.add(f"{name}: endpoints {endpoints} differ from the requested example")
    points = sorted(
        set(endpoints)
        | {value - 1 for value in endpoints[1:]}
        | {value for value in spec["grid"] if value >= num_arms}
        | {budget}
    )

    environment = _construct_environment(ENVIRONMENTS[name][0], dist)
    seed_rngs(master_seed, f"{name}|{dist}", "env", 0)
    environment.reset()
    if compute_instance_hash(environment) != spec["instance_hash"]:
        problems.add(f"{name}: live instance hash differs from preregistration")
    true_pareto = encode_int_set(environment.get_top_arms())

    # ---- stage runs ----------------------------------------------------------
    stage_file = stage_results_path(output_root, name)
    with stage_manifest_path(output_root, name).open() as handle:
        stage_manifest = json.load(handle)
    if stage_manifest["doubling_config_hash"] != digest:
        problems.add(f"{name}: stage manifest config hash mismatch")
    if stage_manifest["num_runs"] != num_runs:
        problems.add(f"{name}: stage manifest repetition count mismatch")
    if require_clean and stage_manifest["source_dirty"]:
        problems.add(f"{name}: stage runs were produced from an uncommitted source tree")
    if stage_manifest["result_file"]["sha256"] != sha256_file(stage_file):
        problems.add(f"{name}: stage result file differs from its manifest")
    columns, stage_rows = _read_csv_rows(stage_file)
    if columns != SCHEMA_COLUMNS:
        problems.add(f"{name}: stage result columns differ from the runner schema")
    stage_index: dict[tuple[str, int, int], dict] = {}
    for row in stage_rows:
        key = (row["algorithm"], int(row["repetition"]), int(row["arm_pull"]))
        if key in stage_index:
            problems.add(f"{name}: duplicate stage row {key}")
        stage_index[key] = row
    expected_keys = {(base, rep, b) for base in bases.values() for rep in range(num_runs) for b in budgets}
    if set(stage_index) != expected_keys:
        problems.add(
            f"{name}: stage coverage mismatch (missing={len(expected_keys - set(stage_index))}, "
            f"extra={len(set(stage_index) - expected_keys)})"
        )
    encoded_budgets = encode_int_set(budgets)
    for key, row in stage_index.items():
        if (row["environment"], row["instance_hash"], int(row["max_budget"])) != (name, spec["instance_hash"], budget):
            problems.add(f"{name}: stage row {key} has wrong environment binding")
        if row["evaluation_grid"] != encoded_budgets or row["master_seed"] != str(master_seed):
            problems.add(f"{name}: stage row {key} has wrong stage grid or seed")
        if row["rho"] != "" or row["posterior_model"] != "":
            problems.add(f"{name}: stage row {key} is not a fixed-budget EGE row")
        if row["true_pareto"] != true_pareto:
            problems.add(f"{name}: stage row {key} has wrong true Pareto set")
        mismatch = _metric_mismatch(row, environment)
        if mismatch:
            problems.add(f"{name}: stage row {key}: {mismatch}")

    # ---- doubling trajectories -----------------------------------------------
    doubling_file = doubling_path(output_root, name)
    columns, doubling = _read_csv_rows(doubling_file)
    if columns != DOUBLING_COLUMNS:
        problems.add(f"{name}: doubling columns differ from the declared schema")
    encoded_points = encode_int_set(points)
    trajectories: dict[tuple[str, int], list[int]] = {}
    for row in doubling:
        algorithm, repetition, arm_pull = row["algorithm"], int(row["repetition"]), int(row["arm_pull"])
        trajectories.setdefault((algorithm, repetition), []).append(arm_pull)
        if algorithm not in bases:
            problems.add(f"{name}: unexpected doubling algorithm {algorithm}")
            continue
        base = bases[algorithm]
        s = independent_stage(arm_pull, num_arms)
        if s < 1 or arm_pull > budget:
            problems.add(f"{name}/{algorithm}/rep{repetition}: row at t={arm_pull} lies outside [K, B]")
            continue
        stage_budget = 2 ** (s - 1) * num_arms
        stage_end = budget if s == stage_count else (2 ** (s + 1) - 1) * num_arms - 1
        expected_meta = {
            "base_algorithm": base, "stage": str(s), "stage_budget": str(stage_budget),
            "stage_start": str((2**s - 1) * num_arms), "stage_end": str(stage_end),
            "stage_actual_draws": str(expected_ege_draws(base, stage_budget, num_arms)),
            "doubling_config_hash": digest, "max_budget": str(budget), "evaluation_grid": encoded_points,
            "environment": name, "instance_hash": spec["instance_hash"], "true_pareto": true_pareto,
        }
        for field, value in expected_meta.items():
            if row[field] != value:
                problems.add(f"{name}/{algorithm}/rep{repetition}/t={arm_pull}: {field}={row[field]!r}, expected {value!r}")
        source = stage_index.get((base, repetition, stage_budget))
        if source is None:
            problems.add(f"{name}/{algorithm}/rep{repetition}/t={arm_pull}: no stage-{s} run")
            continue
        for field in ("recommendation", *METRICS):
            if row[field] != source[field]:
                problems.add(f"{name}/{algorithm}/rep{repetition}/t={arm_pull}: {field} differs from stage {s}")
    expected_trajectories = {(algorithm, rep) for algorithm in bases for rep in range(num_runs)}
    if set(trajectories) != expected_trajectories:
        problems.add(f"{name}: doubling trajectory coverage mismatch")
    for key, pulls in trajectories.items():
        if pulls != points:
            problems.add(f"{name}/{key}: evaluation points differ from exact step points")

    # ---- slim plot inputs ------------------------------------------------------
    with build_manifest_path(output_root, name).open() as handle:
        build_manifest = json.load(handle)
    plot_file = plot_input_path(output_root, name)
    if build_manifest["doubling_config_hash"] != digest:
        problems.add(f"{name}: build manifest config hash mismatch")
    if build_manifest["doubling_file"]["sha256"] != sha256_file(doubling_file):
        problems.add(f"{name}: doubling file differs from its build manifest")
    if build_manifest["plot_input_file"]["sha256"] != sha256_file(plot_file):
        problems.add(f"{name}: plot input differs from its build manifest")
    if build_manifest["stage_result_sha256"] != sha256_file(stage_file):
        problems.add(f"{name}: doubling was built from a different stage file")
    source = pd.read_csv(
        source_results_dir(config) / f"{name}.csv",
        usecols=[*PLOT_COLUMNS, "recommendation"], dtype=str, keep_default_na=False,
    )
    expected_anytime = source[source["algorithm"].isin(ANYTIME_ALGORITHMS)].loc[:, PLOT_COLUMNS].reset_index(drop=True)
    plot_frame = pd.read_csv(plot_file, dtype=str, keep_default_na=False)
    if list(plot_frame.columns) != PLOT_COLUMNS:
        problems.add(f"{name}: plot input columns differ")
    else:
        anytime = plot_frame[plot_frame["algorithm"].isin(ANYTIME_ALGORITHMS)].reset_index(drop=True)
        doubling_part = plot_frame[plot_frame["algorithm"].isin(bases)].reset_index(drop=True)
        if len(anytime) + len(doubling_part) != len(plot_frame):
            problems.add(f"{name}: plot input contains unexpected algorithms")
        if not anytime.equals(expected_anytime):
            problems.add(f"{name}: anytime plot rows are not verbatim preregistered rows")
        expected_doubling = pd.DataFrame(
            [{column: row[column] for column in PLOT_COLUMNS} for row in doubling], columns=PLOT_COLUMNS
        ).astype(str)
        if not doubling_part.equals(expected_doubling):
            problems.add(f"{name}: doubling plot rows differ from the doubling trajectories")

    # ---- native seed-contract reproduction -------------------------------------
    preregistered = source[source["algorithm"].isin(bases.values())].set_index(["algorithm", "repetition", "arm_pull"])
    reproduced = 0
    for base in bases.values():
        selected = [spec["grid"][index] for index in reproduction["grid_indices"]]
        for repetition in reproduction["repetitions"]:
            rows = _run_repetition_ege(
                environment, name, dist, base, EGE_ALGORITHMS[base], budget, selected, master_seed, repetition
            )
            for row in rows:
                stored = preregistered.loc[(base, str(repetition), str(row["arm_pull"])), "recommendation"]
                reproduced += 1
                if stored != row["recommendation"]:
                    problems.add(f"{name}/{base}/rep{repetition}/t={row['arm_pull']}: local EGE does not reproduce preregistered row")

    return {
        "stage_rows": len(stage_rows),
        "doubling_rows": len(doubling),
        "plot_rows": len(plot_frame),
        "stages": stage_count,
        "endpoints": endpoints,
        "evaluation_points": len(points),
        "reproduced_preregistered_rows": reproduced,
        "data_sha256": {
            "stage": sha256_file(stage_file),
            "doubling": sha256_file(doubling_file),
            "plot_input": sha256_file(plot_file),
        },
    }


def validate_figures(
    output_root: Path, figures_dir: Path, specs: list[dict], algorithms: list[str], num_runs: int,
    config: dict, problems: Problems,
) -> dict:
    protected_dir = REPO_ROOT / config["protected_publication_dir"]
    names = [spec["name"] for spec in specs]
    figure_hashes = {}
    for metric in METRICS:
        png, table_path = figure_paths(figures_dir, metric)
        if not png.is_file() or png.stat().st_size == 0 or not table_path.is_file():
            problems.add(f"{metric}: missing doubling figure or plot table")
            continue
        figure_hashes[png.name] = sha256_file(png)
        figure_hashes[table_path.name] = sha256_file(table_path)
        table = pd.read_csv(table_path)
        existing = pd.read_csv(protected_dir / f"publication_{metric}_plot_table.csv")
        key = ["environment", "algorithm", "arm_pull"]
        new_anytime = table[table["algorithm"].isin(ANYTIME_ALGORITHMS)].sort_values(key).reset_index(drop=True)
        old_anytime = existing[
            existing["algorithm"].isin(ANYTIME_ALGORITHMS) & existing["environment"].isin(names)
        ].sort_values(key).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(new_anytime, old_anytime, check_exact=True)
        except AssertionError as error:
            problems.add(f"{metric}: anytime curves differ from the existing publication table: {error}")
        if set(table["algorithm"]) != set(ANYTIME_ALGORITHMS) | set(algorithms):
            problems.add(f"{metric}: plotted algorithms {sorted(set(table['algorithm']))} are unexpected")
        for spec in specs:
            frame = pd.read_csv(doubling_path(output_root, spec["name"]))
            for algorithm in algorithms:
                curve = table[(table["environment"] == spec["name"]) & (table["algorithm"] == algorithm)].sort_values("arm_pull")
                rows = frame[frame["algorithm"] == algorithm]
                expected = rows.groupby("arm_pull")[metric].mean().sort_index()
                if list(curve["arm_pull"]) != list(expected.index):
                    problems.add(f"{metric}/{spec['name']}/{algorithm}: plotted points differ from trajectories")
                    continue
                if not np.allclose(curve["mean"].to_numpy(), expected.to_numpy(), rtol=0, atol=1e-12):
                    problems.add(f"{metric}/{spec['name']}/{algorithm}: plotted means differ from trajectories")
                if int(curve["arm_pull"].iloc[0]) != spec["num_arms"] or float(curve["normalized_budget"].iloc[-1]) != 1.0:
                    problems.add(f"{metric}/{spec['name']}/{algorithm}: curve must span [K/B, 1]")
                if set(curve["n"]) != {num_runs}:
                    problems.add(f"{metric}/{spec['name']}/{algorithm}: plotted repetition count differs")
    manifest_file = figure_manifest_path(figures_dir)
    if not manifest_file.is_file():
        problems.add("missing figure manifest")
    else:
        with manifest_file.open() as handle:
            manifest = json.load(handle)
        if manifest.get("doubling_config_hash") != config_hash(config):
            problems.add("figure manifest config hash mismatch")
        if manifest.get("files") != figure_hashes:
            problems.add("figure files differ from the figure manifest")
    return figure_hashes


def validate(
    output_root: Path | None = None,
    config: dict | None = None,
    environments: list[str] | None = None,
    algorithms: list[str] | None = None,
    num_runs: int | None = None,
    require_clean: bool = True,
    reproduction: dict | None = None,
    figures_dir: Path | None = None,
    check_figures: bool = False,
    write_report: bool = True,
) -> dict:
    config = load_config() if config is None else config
    output_root = Path(default_output_root(config) if output_root is None else output_root)
    figures_dir = Path(REPO_ROOT / config["figures_dir"] if figures_dir is None else figures_dir)
    algorithms = algorithms or list(DOUBLING_ALGORITHMS)
    num_runs = int(config["num_runs"] if num_runs is None else num_runs)
    reproduction = reproduction or config["reproduction_check"]
    problems = Problems()
    try:
        verify_source_hashes(config)
        verify_protected_hashes(config)
    except ValueError as error:
        problems.add(str(error))
    specs = [spec for spec in environment_specs(config) if environments is None or spec["name"] in environments]
    summaries = {}
    for spec in specs:
        try:
            summaries[spec["name"]] = validate_environment(
                output_root, spec, config, algorithms, num_runs, require_clean, reproduction, problems
            )
        except (OSError, KeyError, ValueError) as error:
            problems.add(f"{spec['name']}: {type(error).__name__}: {error}")
    figure_hashes = None
    if check_figures:
        figure_hashes = validate_figures(output_root, figures_dir, specs, algorithms, num_runs, config, problems)
    report = {
        "schema_version": "ege-doubling-validation-v1",
        "status": "pass" if not problems else "fail",
        "problem_count": len(problems),
        "problems": list(problems[:MAX_REPORTED_PROBLEMS]),
        "doubling_config_hash": config_hash(config),
        "environments": [spec["name"] for spec in specs],
        "algorithms": algorithms,
        "num_runs": num_runs,
        "require_clean": require_clean,
        "reproduction_check": reproduction,
        "summaries": summaries,
        "figures_checked": check_figures,
        "figure_files": figure_hashes,
    }
    if write_report:
        write_json_atomic(report_path(output_root), report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--environments", default=None)
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--figures", action="store_true", help="Also validate the rendered figures/tables.")
    parser.add_argument("--allow-dirty-source", action="store_true")
    args = parser.parse_args()
    report = validate(
        output_root=Path(args.output_root) if args.output_root else None,
        environments=[item.strip() for item in args.environments.split(",")] if args.environments else None,
        num_runs=args.num_runs,
        require_clean=not args.allow_dirty_source,
        check_figures=args.figures,
    )
    for problem in report["problems"]:
        print(f"FAIL: {problem}")
    counts = {name: {k: v for k, v in summary.items() if k != "data_sha256"} for name, summary in report["summaries"].items()}
    print(json.dumps(counts, indent=2))
    print(f"Validation {report['status'].upper()} ({report['problem_count']} problem(s))")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
