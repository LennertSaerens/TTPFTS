#!/usr/bin/env python3
"""Run the approved paired heteroscedastic pilot after a local preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .pilot import environment_key, load_pilot_config, pilot_candidates, pilot_config_hash, run_candidate_condition


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"


def projected_local_core_hours(config: dict) -> float:
    """Conservative all-algorithm, full-grid serial workload estimate.

    The probe does not cover every (K, D) pair in the approved cohort.  Until a
    fitted and validated runtime model exists, use each algorithm's slowest
    observed steady-state pull rate rather than extrapolating a nearest stratum.
    """
    probe = pd.read_csv(REPORTS_DIR / "runtime_probe.csv")
    candidates = pilot_candidates(config)
    slowest_rate = probe.groupby("algorithm")["seconds_per_pull_conservative"].max()
    missing = set(config["algorithms"]) - set(slowest_rate.index)
    if missing:
        raise ValueError(f"runtime probe lacks algorithms: {sorted(missing)}")
    total_budget = sum(candidate.budget for candidate in candidates)
    # Anytime algorithms produce every grid checkpoint in one B-pull run.  EGE
    # restarts at each checkpoint, whose approved grid sums to 5.5 B.
    grid_multiplier = {
        algorithm: 5.5 if algorithm.startswith("EGE_") else 1.0
        for algorithm in config["algorithms"]
    }
    seconds_per_condition = sum(
        total_budget * grid_multiplier[algorithm] * slowest_rate[algorithm]
        for algorithm in config["algorithms"]
    )
    # Matched heterogeneous/control conditions are separate independent runs.
    return 2.0 * seconds_per_condition / 3600.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--override-local-preflight",
        action="store_true",
        help="Required to run locally when the measured conservative projection exceeds the approved limit.",
    )
    args = parser.parse_args()
    config = load_pilot_config()
    projected_hours = projected_local_core_hours(config)
    limit = float(config["execution_policy"]["local_max_projected_core_hours"])
    if projected_hours > limit and not args.override_local_preflight:
        raise SystemExit(
            f"REFUSING LOCAL PILOT: projected all-algorithm paired workload is {projected_hours:.1f} core-hours, "
            f"above approved limit {limit:.1f}. Per approved policy, stop and request HPC; "
            "do not override without an explicit updated decision."
        )

    digest = pilot_config_hash(config)[:12]
    results_dir = args.results_dir or ROOT.parent.parent / f"results_heteroscedastic_pilot_{digest}"
    run_dir = results_dir / "run"
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "heteroscedastic-pilot-results-v1",
        "pilot_config_hash": pilot_config_hash(config),
        "algorithms": config["algorithms"],
        "num_runs": config["num_runs"],
        "conditions": config["conditions"],
        "projected_local_core_hours_conservative": projected_hours,
        "instances": {},
    }
    for candidate in pilot_candidates(config):
        key = environment_key(candidate)
        manifest["instances"][key] = {
            "instance_id": candidate.instance_id,
            "condition": candidate.condition,
            "budget": candidate.budget,
            "grid": list(candidate.grid),
            "hardness": candidate.hardness.hardness,
            "num_arms": candidate.means.shape[0],
            "num_objectives": candidate.means.shape[1],
        }
        rows = run_candidate_condition(candidate, config, results_dir, run_dir, args.n_jobs)
        print(f"{key}: {rows} rows")
    (run_dir / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
