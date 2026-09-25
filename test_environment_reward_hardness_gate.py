"""
Standalone regression tests for the reward/environment/hardness gate.

No pytest dependency (matches the repo's existing convention, e.g.
test_ape_equivalence.py / the former verify_fixes.py): run directly with

    python test_environment_reward_hardness_gate.py

and check for "ALL TESTS PASSED" at the end / a non-zero exit code on failure.

Scope: environment files, environment instance data, and the new
reward/hardness validation scripts under environments/. Does NOT touch
momab_main.py, bandits/, plotting, or HPC files, and does NOT test the
non-Ege environments (N50VS, CovBoost, PoissonExp), which were intentionally
left untouched.
"""

import sys
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


# ---------------------------------------------------------------------------
# 1. Gaussian reward covariance: all 8 EgeExp environments use std=0.5
#    (i.e. diagonal covariance sigma^2 I_D with sigma^2=0.25).
# ---------------------------------------------------------------------------
def test_gaussian_std():
    from environments import (
        EgeExp1, EgeExp2, EgeExp3, EgeExp4, EgeExp5, EgeExp6, EgeExp7, EgeExp8,
    )
    for cls in [EgeExp1, EgeExp2, EgeExp3, EgeExp4, EgeExp5, EgeExp6, EgeExp7, EgeExp8]:
        env = cls()
        stds = np.array([d.std for d in env.distributions])
        check(f"{cls.__name__} Gaussian std == 0.5",
              bool(np.allclose(stds, 0.5)),
              f"got std range [{stds.min()}, {stds.max()}]")
        check(f"{cls.__name__} implied variance == 0.25",
              bool(np.allclose(stds ** 2, 0.25)))


def test_std_from_variance_helper():
    from environments.distributions import std_from_variance
    check("std_from_variance(0.25) == 0.5", np.isclose(std_from_variance(0.25), 0.5))
    check("std_from_variance(1.0) == 1.0", np.isclose(std_from_variance(1.0), 1.0))


# ---------------------------------------------------------------------------
# 2. Frozen EgeExp1/EgeExp4/EgeExp5 instances: checksum-verified, reproducible
#    from master seed 0 without seed-shopping, and no longer redrawn on reset().
# ---------------------------------------------------------------------------
def test_frozen_instances_reproducible_from_master_seed():
    from environments.instances.freeze_instances import freeze_all
    from environments.instances.frozen_instance_loader import load_frozen_arms

    regenerated = freeze_all(write=False)
    for env_name, arms in regenerated.items():
        on_disk = load_frozen_arms(env_name)
        check(f"{env_name} regenerated arms match committed frozen CSV (master seed 0)",
              bool(np.array_equal(arms, on_disk)))


def test_frozen_instances_checksums_valid():
    from environments.instances.frozen_instance_loader import load_frozen_arms
    for env_name in ["EgeExp1", "EgeExp4", "EgeExp5"]:
        # Raises FrozenInstanceError on checksum mismatch; success == no exception.
        arms = load_frozen_arms(env_name)
        check(f"{env_name} frozen CSV loads and checksum-verifies", arms.ndim == 2)


def test_reset_does_not_redraw_means():
    from environments import EgeExp1, EgeExp4, EgeExp5
    for cls in [EgeExp1, EgeExp4, EgeExp5]:
        env = cls()
        means_before = np.array([d.mean for d in env.distributions]).copy()
        env.reset()
        means_after = np.array([d.mean for d in env.distributions])
        check(f"{cls.__name__}.reset() preserves arm means (paper mode)",
              bool(np.array_equal(means_before, means_after)))

        env2 = cls()
        means_new_instance = np.array([d.mean for d in env2.distributions])
        check(f"{cls.__name__} two independent instantiations agree (single frozen instance)",
              bool(np.array_equal(means_before, means_new_instance)))


def test_expected_arm_counts():
    from environments import EgeExp1, EgeExp4, EgeExp5
    expectations = {"EgeExp1": (60, 2), "EgeExp4": (50, 10), "EgeExp5": (20, 2)}
    for cls in [EgeExp1, EgeExp4, EgeExp5]:
        env = cls()
        expected_k, expected_d = expectations[cls.__name__]
        check(f"{cls.__name__} has {expected_k} arms x {expected_d} objectives",
              env.num_arms == expected_k and env.num_objectives == expected_d)


# ---------------------------------------------------------------------------
# 3. Hardness module correctness on a hand-verified toy instance.
# ---------------------------------------------------------------------------
def test_hardness_hand_verified_example():
    from environments.hardness import compute_hardness, compute_pareto_mask

    # A=(1,0), B=(0,1): mutually non-dominated (Pareto). C=(-1,-1): dominated by both.
    # Hand-derived (see task notes): Delta_A = Delta_B = 1, Delta_C = 1, H = 3.
    arms = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    mask = compute_pareto_mask(arms)
    check("toy instance Pareto set == {A, B}", list(np.where(mask)[0]) == [0, 1])

    result = compute_hardness("toy", arms, np.where(mask)[0])
    check("toy Delta_A == 1", np.isclose(result.gaps_direct[0], 1.0))
    check("toy Delta_B == 1", np.isclose(result.gaps_direct[1], 1.0))
    check("toy Delta_C == 1", np.isclose(result.gaps_direct[2], 1.0))
    check("toy direct gaps match lemma gaps", result.gaps_match,
          f"max mismatch {result.max_abs_gap_mismatch}")
    check("toy H(nu) == 3", np.isclose(result.hardness, 3.0), f"got {result.hardness}")
    check("toy has no zero-gap arms", len(result.zero_gap_arms) == 0)


def test_hardness_zero_gap_detection():
    from environments.hardness import compute_hardness, compute_pareto_mask

    # D=(1,0), E=(0,0): E is tied with D on objective 1 (both 0 there... wait use
    # a clean exact-tie construction: D=(1,0) Pareto, E=(1,-1) dominated by D but
    # tied with D on objective 1 -> Delta_E should be 0 (m(E,D) = min(0, 1) = 0).
    arms = np.array([[1.0, 0.0], [1.0, -1.0]])
    mask = compute_pareto_mask(arms)
    result = compute_hardness("toy_zero_gap", arms, np.where(mask)[0])
    check("toy_zero_gap arm 1 (E) has Delta == 0", np.isclose(result.gaps_direct[1], 0.0))
    check("toy_zero_gap flags arm 1 as zero-gap", 1 in result.zero_gap_arms.tolist())
    check("toy_zero_gap H(nu) is infinite", not np.isfinite(result.hardness))
    check("toy_zero_gap direct/lemma agree even in degenerate case", result.gaps_match)


# ---------------------------------------------------------------------------
# 4. Independent Pareto-set validation + direct/lemma gap cross-check across
#    all 8 EgeExp environments, and the EgeExp3 zero-gap finding specifically.
# ---------------------------------------------------------------------------
def test_all_egeexp_pareto_and_hardness_crosscheck():
    from environments import (
        EgeExp1, EgeExp2, EgeExp3, EgeExp4, EgeExp5, EgeExp6, EgeExp7, EgeExp8,
    )
    import importlib
    EgeExp3_module = importlib.import_module("environments.EgeExp3")
    from environments.hardness import compute_hardness, objective_value_ties

    for cls in [EgeExp1, EgeExp2, EgeExp3, EgeExp4, EgeExp5, EgeExp6, EgeExp7, EgeExp8]:
        env = cls()
        arms = np.array([d.mean for d in env.distributions])
        result = compute_hardness(cls.__name__, arms, env.pareto_indices)

        check(f"{cls.__name__} independent Pareto set matches env.pareto_indices",
              result.pareto_sets_match)
        check(f"{cls.__name__} direct gaps match lemma gaps (max mismatch {result.max_abs_gap_mismatch:.2e})",
              result.gaps_match)

        check(f"{cls.__name__} has no zero-gap arms", len(result.zero_gap_arms) == 0,
              f"zero-gap arms: {result.zero_gap_arms.tolist()}")
        check(f"{cls.__name__} has finite hardness", np.isfinite(result.hardness))

        if cls.__name__ == "EgeExp3":
            # The original construction sampled the dominated arc's closed
            # endpoint 345 deg, which is exactly the reflection of Pareto arm 0
            # (15 deg) about the x-axis, so cos(345 deg) == cos(15 deg) to full
            # float precision. Arm 199 was then only weakly dominated, giving
            # Delta_0 = Delta_199 = 0 and H = +inf. The arc is now inset from
            # both critical boundaries by half the Pareto angular step.
            check("EgeExp3 has no exact objective-value ties",
                  objective_value_ties(arms) == [], f"ties: {objective_value_ties(arms)}")
            check("EgeExp3 dominated arc ends strictly inside the admissible interval",
                  EgeExp3_module.DOMINATED_END_DEG < EgeExp3_module.CRITICAL_HIGH_DEG
                  and EgeExp3_module.DOMINATED_START_DEG > EgeExp3_module.CRITICAL_LOW_DEG)
            check("EgeExp3 arc inset equals half the Pareto angular step",
                  np.isclose(EgeExp3_module.CRITICAL_HIGH_DEG - EgeExp3_module.DOMINATED_END_DEG,
                             EgeExp3_module.PARETO_STEP_DEG / 2.0))


# ---------------------------------------------------------------------------
# 5. Non-Ege environments must be untouched / unaffected.
# ---------------------------------------------------------------------------
def test_non_ege_environments_unaffected():
    from environments.N50VS import N50VS
    from environments.CovBoost import CovBoost

    n50 = N50VS()
    check("N50VS still uses uniform per-arm stds in [1, 3] (untouched)",
          bool(np.all(n50.stds >= 1.0) and np.all(n50.stds <= 3.0)))

    cb = CovBoost()
    check("CovBoost still uses its explicit [0.7, 0.83, 1.54] std (untouched)",
          bool(np.allclose(np.sort(cb.stds), np.sort([0.7, 0.83, 1.54]))))


def test_end_to_end_gate_report_runs():
    from environments.validate_reward_environment_gate import run_gate
    report = run_gate()
    check("gate report: reward covariance passes for all 8 EgeExp envs",
          report["gate2_reward_covariance_pass"])
    check("gate report: frozen-instance stability passes for EgeExp1/4/5",
          report["gate2_frozen_instance_pass"])
    check("gate report: Pareto set matches and gaps cross-check for all envs",
          all(r["pareto_sets_match"] and r["gaps_match_direct_vs_lemma"]
              for r in report["hardness_summary"]))
    check("gate report: every environment now has finite hardness",
          all(np.isfinite(r["hardness_H"]) for r in report["hardness_summary"]),
          f'infinite: {[r["env"] for r in report["hardness_summary"] if not np.isfinite(r["hardness_H"])]}')


if __name__ == "__main__":
    run("Gaussian reward std/variance", test_gaussian_std)
    run("std_from_variance helper", test_std_from_variance_helper)
    run("Frozen instances reproducible from master seed 0", test_frozen_instances_reproducible_from_master_seed)
    run("Frozen instance checksums valid", test_frozen_instances_checksums_valid)
    run("reset() does not redraw means (paper mode)", test_reset_does_not_redraw_means)
    run("Expected arm counts for frozen environments", test_expected_arm_counts)
    run("Hardness module: hand-verified toy example", test_hardness_hand_verified_example)
    run("Hardness module: zero-gap detection", test_hardness_zero_gap_detection)
    run("All 8 EgeExp: Pareto + hardness cross-check", test_all_egeexp_pareto_and_hardness_crosscheck)
    run("Non-Ege environments unaffected", test_non_ege_environments_unaffected)
    run("End-to-end gate report runs and passes", test_end_to_end_gate_report_runs)

    print("\n" + "=" * 80)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)
