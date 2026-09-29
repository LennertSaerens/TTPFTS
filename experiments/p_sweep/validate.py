#!/usr/bin/env python3
"""Independently validate p-sweep coverage, provenance, metrics, and p=0.5 identity."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from experiments.final_config_loader import load_final_config
from momab_main import ENVIRONMENTS, _construct_environment
from runner.schema import decode_int_set, encode_int_set, validate_row_metrics
from runner.seeding import seed_rngs

from .config import (
    algorithm_label,
    config_hash,
    default_output_root,
    environment_config,
    expected_environments,
    load_config,
    p_tag,
)
from .run import SWEEP_COLUMNS, shard_is_complete, shard_path


IDENTITY_COLUMNS = (
    "environment",
    "instance_hash",
    "posterior_model",
    "repetition",
    "arm_pull",
    "max_budget",
    "rho",
    "master_seed",
    "bernoulli",
    "jaccard",
    "misidentification",
    "recommendation",
    "true_pareto",
)


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _add(problems: list[str], message: str) -> None:
    problems.append(message)


def _same_float(actual: str, expected: float, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _validate_row(
    row: dict,
    *,
    environment: str,
    p: float,
    repetition: int,
    arm_pull: int,
    details: dict,
    digest: str,
    true_pareto: set[int],
    master_seed: int,
    problems: list[str],
) -> None:
    label = f"{environment}/{p_tag(p)}/rep{repetition:05d}/t={arm_pull}"
    if set(row) != set(SWEEP_COLUMNS):
        _add(problems, f"{label}: schema mismatch")
        return
    expected = {
        "environment": environment,
        "instance_hash": details["instance_hash"],
        "algorithm": algorithm_label(p),
        "posterior_model": "TPosterior",
        "repetition": repetition,
        "arm_pull": arm_pull,
        "max_budget": details["budget"],
        "evaluation_grid": encode_int_set(details["grid"]),
        "master_seed": master_seed,
        "sweep_config_hash": digest,
    }
    for key, value in expected.items():
        if str(row[key]) != str(value):
            _add(problems, f"{label}: {key}={row[key]!r}, expected {value!r}")
    if not _same_float(row["rho"], p):
        _add(problems, f"{label}: rho={row['rho']!r}, expected {p}")
    if not _same_float(row["normalized_budget"], arm_pull / details["budget"]):
        _add(problems, f"{label}: normalized budget mismatch")
    try:
        recommendation = decode_int_set(row["recommendation"])
        stored_pareto = decode_int_set(row["true_pareto"])
    except (TypeError, ValueError) as exc:
        _add(problems, f"{label}: invalid arm-set encoding: {exc}")
        return
    valid_arms = set(range(details["num_arms"]))
    if not recommendation <= valid_arms:
        _add(problems, f"{label}: recommendation contains invalid arm indices")
    if stored_pareto != true_pareto:
        _add(problems, f"{label}: true Pareto set changes within the fixed environment")
    metric_problems = validate_row_metrics(row, details["num_arms"])
    for problem in metric_problems:
        _add(problems, f"{label}: {problem}")


def _expected_true_pareto(environment_name: str, config: dict) -> set[int]:
    """Reconstruct the frozen true Pareto set independently of saved result rows."""
    final_config = load_final_config()
    environment_class, _legacy_budget = ENVIRONMENTS[environment_name]
    environment = _construct_environment(environment_class, final_config["reward_model"]["distribution"])
    environment_tag = f"{environment_name}|{final_config['reward_model']['distribution']}"
    seed_rngs(config["master_seed"], environment_tag, "env", 0)
    environment.reset()
    return set(environment.get_top_arms())


def validate(
    run_dir: Path,
    results_dir: Path,
    final_results_dir: Path,
    *,
    environments: list[str] | None = None,
    p_values: list[float] | None = None,
    repetitions: list[int] | None = None,
) -> list[str]:
    """Return all detected violations; an empty list means validation passed."""
    config = load_config()
    digest = config_hash(config)
    environments = [item["name"] for item in expected_environments()] if environments is None else environments
    p_values = [float(value) for value in config["p_values"]] if p_values is None else p_values
    repetitions = list(range(config["num_runs"])) if repetitions is None else repetitions
    problems: list[str] = []

    for environment in environments:
        details = environment_config(environment, config)
        result_path = results_dir / f"{environment}.csv"
        if not result_path.is_file():
            _add(problems, f"{environment}: missing merged result CSV")
            continue
        merged_rows = _rows(result_path)
        expected_keys = {
            (round(float(p), 1), repetition, arm_pull)
            for p in p_values
            for repetition in repetitions
            for arm_pull in details["grid"]
        }
        observed: dict[tuple[float, int, int], dict] = {}
        for row in merged_rows:
            try:
                key = (round(float(row["rho"]), 1), int(row["repetition"]), int(row["arm_pull"]))
            except (KeyError, ValueError):
                _add(problems, f"{environment}: invalid merged key")
                continue
            if key in observed:
                _add(problems, f"{environment}: duplicate merged row {key}")
            observed[key] = row
        if set(observed) != expected_keys:
            missing, extra = expected_keys - set(observed), set(observed) - expected_keys
            if missing:
                _add(problems, f"{environment}: missing merged rows, first={sorted(missing)[:3]}")
            if extra:
                _add(problems, f"{environment}: unexpected merged rows, first={sorted(extra)[:3]}")

        final_by_key: dict[tuple[int, int], dict] = {}
        if 0.5 in p_values:
            final_path = final_results_dir / f"{environment}.csv"
            if not final_path.is_file():
                _add(problems, f"{environment}: missing final results for p=0.5 identity")
            else:
                final_by_key = {
                    (int(row["repetition"]), int(row["arm_pull"])): row
                    for row in _rows(final_path)
                    if row["algorithm"] == "TTPFTS_UNKNOWN_VAR"
                }

        true_pareto = _expected_true_pareto(environment, config)
        for p in p_values:
            for repetition in repetitions:
                path = shard_path(run_dir, environment, p, repetition)
                if not shard_is_complete(run_dir, environment, p, repetition, details["grid"], digest):
                    _add(problems, f"{environment}/{p_tag(p)}/rep{repetition:05d}: incomplete or stale shard")
                    continue
                shard_rows = _rows(path)
                if len(shard_rows) != len(details["grid"]):
                    _add(problems, f"{environment}/{p_tag(p)}/rep{repetition:05d}: shard row count mismatch")
                    continue
                for row, arm_pull in zip(shard_rows, details["grid"]):
                    _validate_row(
                        row,
                        environment=environment,
                        p=p,
                        repetition=repetition,
                        arm_pull=arm_pull,
                        details=details,
                        digest=digest,
                        true_pareto=true_pareto,
                        master_seed=config["master_seed"],
                        problems=problems,
                    )
                    merged = observed.get((round(float(p), 1), repetition, arm_pull))
                    if merged != row:
                        _add(problems, f"{environment}/{p_tag(p)}/rep{repetition:05d}/t={arm_pull}: merged row mismatch")
                    if p == 0.5:
                        final = final_by_key.get((repetition, arm_pull))
                        if final is None:
                            _add(problems, f"{environment}/p0.5/rep{repetition:05d}/t={arm_pull}: final row missing")
                        else:
                            for column in IDENTITY_COLUMNS:
                                if row[column] != final[column]:
                                    _add(problems, f"{environment}/p0.5/rep{repetition:05d}/t={arm_pull}: final identity mismatch in {column}")
    return problems


def _parse_values(spec: str | None, transform) -> list | None:
    return None if spec is None else [transform(value) for value in spec.split(",") if value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--final-results-dir", type=Path, default=Path("results_preregistered_final_ff29cc456351")
    )
    parser.add_argument("--environments", default=None)
    parser.add_argument("--p-values", default=None)
    parser.add_argument("--repetitions", default=None)
    args = parser.parse_args()

    config = load_config()
    output_root = default_output_root(config)
    problems = validate(
        args.run_dir or output_root / "run",
        args.results_dir or output_root / "results",
        args.final_results_dir,
        environments=_parse_values(args.environments, str),
        p_values=_parse_values(args.p_values, float),
        repetitions=_parse_values(args.repetitions, int),
    )
    if problems:
        print("P-SWEEP VALIDATION FAILED")
        for problem in problems[:100]:
            print(f"- {problem}")
        return 1
    print("P-SWEEP VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
