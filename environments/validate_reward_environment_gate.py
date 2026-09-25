"""
Standalone validation + report script for the reward/environment/hardness
gate covering the 8 final EgeExp benchmarking environments.

Checks performed (all independent of the code paths they validate):
  1. Reward covariance: every EgeExp environment's Gaussian reward std is
     exactly 0.5 per arm per objective (diagonal covariance sigma^2 I_D with
     sigma^2 = 0.25), as specified in the manuscript.
  2. Pareto set: an independently reimplemented O(K^2 D) pairwise-domination
     check is cross-referenced against each environment's own cached
     `pareto_indices`.
  3. Hardness: per-arm gaps Delta_a are computed via the direct definition
     (case-split on Pareto-set membership) AND via the equivalent, unified
     Lemma 2.1.2 formula, and cross-checked arm-by-arm. H(nu) = sum 1/Delta_a^2
     is reported, together with zero-gap arms (which make H infinite) and the
     resulting change relative to the current flat budget of 5000.
  4. EgeExp1/EgeExp4/EgeExp5 frozen-instance integrity: reload via the
     checksum-verifying loader and confirm two successive instantiations +
     reset() produce byte-identical arm means (no redraw in "paper mode").

Source-verification limitation: the direct/lemma gap definitions and H(nu)
formula are taken from the single local source document
"/Users/lennertsaerens/Desktop/PhD/TTPFTS/Bounds Workspace clean/Ground Truth
and Knowledge/TTPFTS MOMAB PSI.md". No repository citation for these exact
definitions exists inside MOTS itself (they are external, unpublished
theory notes); this script cannot cross-check them against a second
independent authoritative source, only against each other (Lemma 2.1.2 is
claimed there to be *equivalent* to the direct definition, which this script
verifies empirically per environment/arm).

Usage:
    python -m environments.validate_reward_environment_gate
Writes:
    environments/reports/hardness_report.json
    environments/reports/hardness_report.csv   (per-arm gaps)
    environments/reports/hardness_summary.csv  (per-environment summary)
"""

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from environments import (
    EgeExp1, EgeExp2, EgeExp3, EgeExp4, EgeExp5, EgeExp6, EgeExp7, EgeExp8,
)
from environments.hardness import compute_hardness
from environments.instances.frozen_instance_loader import load_frozen_arms

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

EGE_ENV_CLASSES = {
    "EgeExp1": EgeExp1, "EgeExp2": EgeExp2, "EgeExp3": EgeExp3, "EgeExp4": EgeExp4,
    "EgeExp5": EgeExp5, "EgeExp6": EgeExp6, "EgeExp7": EgeExp7, "EgeExp8": EgeExp8,
}
FROZEN_ENVS = {"EgeExp1", "EgeExp4", "EgeExp5"}
CURRENT_FLAT_BUDGET = 5000
EXPECTED_GAUSSIAN_STD = 0.5
EXPECTED_GAUSSIAN_VARIANCE = 0.25


def check_reward_covariance(env_name: str, env) -> dict:
    stds = np.array([d.std for d in env.distributions])
    ok = bool(np.allclose(stds, EXPECTED_GAUSSIAN_STD))
    return {
        "env": env_name,
        "std_min": float(stds.min()),
        "std_max": float(stds.max()),
        "expected_std": EXPECTED_GAUSSIAN_STD,
        "expected_variance": EXPECTED_GAUSSIAN_VARIANCE,
        "pass": ok,
    }


def check_frozen_instance_stability(env_name: str, env_cls) -> dict:
    """Confirm __init__ + reset() do not redraw means for frozen environments."""
    env = env_cls()
    means_a = np.array([d.mean for d in env.distributions]).copy()
    env.reset()
    means_b = np.array([d.mean for d in env.distributions])
    stable = bool(np.array_equal(means_a, means_b))

    env2 = env_cls()
    means_c = np.array([d.mean for d in env2.distributions])
    identical_across_instances = bool(np.array_equal(means_a, means_c))

    frozen_csv_arms = load_frozen_arms(env_name)
    matches_frozen_csv = bool(np.array_equal(means_a, frozen_csv_arms))

    return {
        "env": env_name,
        "stable_across_reset": stable,
        "identical_across_new_instantiations": identical_across_instances,
        "matches_frozen_csv": matches_frozen_csv,
        "pass": stable and identical_across_instances and matches_frozen_csv,
    }


def run_gate() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    reward_checks = []
    frozen_checks = []
    hardness_rows = []
    per_arm_rows = []

    for env_name, env_cls in EGE_ENV_CLASSES.items():
        env = env_cls()
        arms = np.array([d.mean for d in env.distributions])

        reward_checks.append(check_reward_covariance(env_name, env))
        if env_name in FROZEN_ENVS:
            frozen_checks.append(check_frozen_instance_stability(env_name, env_cls))

        result = compute_hardness(env_name, arms, env.pareto_indices)

        for arm_idx in range(result.num_arms):
            per_arm_rows.append({
                "env": env_name,
                "arm": arm_idx,
                "in_pareto_set": bool(arm_idx in set(result.pareto_indices_independent.tolist())),
                "gap_direct": result.gaps_direct[arm_idx],
                "gap_lemma": result.gaps_lemma[arm_idx],
                "abs_mismatch": abs(result.gaps_direct[arm_idx] - result.gaps_lemma[arm_idx]),
                "zero_gap": bool(arm_idx in result.zero_gap_arms.tolist()),
            })

        hardness_rows.append({
            "env": env_name,
            "num_arms": result.num_arms,
            "num_pareto_arms": len(result.pareto_indices_independent),
            "pareto_sets_match": result.pareto_sets_match,
            "gaps_match_direct_vs_lemma": result.gaps_match,
            "max_abs_gap_mismatch": result.max_abs_gap_mismatch,
            "num_zero_gap_arms": len(result.zero_gap_arms),
            "zero_gap_arm_indices": result.zero_gap_arms.tolist(),
            "hardness_H": result.hardness,
            "current_flat_budget": CURRENT_FLAT_BUDGET,
            "budget_change_vs_5000": (
                None if not np.isfinite(result.hardness)
                else result.hardness - CURRENT_FLAT_BUDGET
            ),
        })

    with open(REPORTS_DIR / "hardness_report.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_arm_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_arm_rows)

    with open(REPORTS_DIR / "hardness_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(hardness_rows[0].keys()))
        writer.writeheader()
        writer.writerows(hardness_rows)

    full_report = {
        "reward_covariance_checks": reward_checks,
        "frozen_instance_checks": frozen_checks,
        "hardness_summary": hardness_rows,
        "gate2_reward_covariance_pass": all(c["pass"] for c in reward_checks),
        "gate2_frozen_instance_pass": all(c["pass"] for c in frozen_checks),
        "gate3_pareto_and_hardness_pass": all(
            r["pareto_sets_match"] and r["gaps_match_direct_vs_lemma"]
            for r in hardness_rows
        ),
    }
    (REPORTS_DIR / "hardness_report.json").write_text(json.dumps(full_report, indent=2, default=str))

    return full_report


def print_report(report: dict) -> None:
    print("=" * 100)
    print("REWARD COVARIANCE (Gate 2)")
    print("=" * 100)
    for c in report["reward_covariance_checks"]:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"  [{status}] {c['env']}: std in [{c['std_min']:.4f}, {c['std_max']:.4f}] "
              f"(expected {c['expected_std']}, i.e. variance {c['expected_variance']})")

    print()
    print("=" * 100)
    print("FROZEN INSTANCE STABILITY (EgeExp1/4/5, Gate 2)")
    print("=" * 100)
    for c in report["frozen_instance_checks"]:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"  [{status}] {c['env']}: stable_across_reset={c['stable_across_reset']} "
              f"identical_across_new_instantiations={c['identical_across_new_instantiations']} "
              f"matches_frozen_csv={c['matches_frozen_csv']}")

    print()
    print("=" * 100)
    print("PARETO SET + HARDNESS (Gate 3)")
    print("=" * 100)
    header = (f"{'env':<10}{'K':>5}{'|S*|':>6}{'pareto_ok':>11}{'gaps_ok':>9}"
              f"{'max_mismatch':>14}{'#zero_gap':>11}{'H(nu)':>14}{'H - 5000':>14}")
    print(header)
    for r in report["hardness_summary"]:
        h_str = "inf" if not np.isfinite(r["hardness_H"]) else f"{r['hardness_H']:.2f}"
        d_str = "n/a (zero gap)" if r["budget_change_vs_5000"] is None else f"{r['budget_change_vs_5000']:+.2f}"
        print(f"{r['env']:<10}{r['num_arms']:>5}{r['num_pareto_arms']:>6}"
              f"{str(r['pareto_sets_match']):>11}{str(r['gaps_match_direct_vs_lemma']):>9}"
              f"{r['max_abs_gap_mismatch']:>14.2e}{r['num_zero_gap_arms']:>11}"
              f"{h_str:>14}{d_str:>14}")
        if r["num_zero_gap_arms"] > 0:
            print(f"    -> zero-gap arm indices: {r['zero_gap_arm_indices']}")

    print()
    print(f"Gate 2 (reward covariance) pass: {report['gate2_reward_covariance_pass']}")
    print(f"Gate 2 (frozen instance stability) pass: {report['gate2_frozen_instance_pass']}")
    print(f"Gate 3 (Pareto set + hardness cross-check) pass: {report['gate3_pareto_and_hardness_pass']}")


if __name__ == "__main__":
    report = run_gate()
    print_report(report)
