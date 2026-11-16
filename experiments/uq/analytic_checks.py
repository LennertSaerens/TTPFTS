#!/usr/bin/env python3
"""Analytic verification suite for Baseline Eq. (10), Variant 1 (V1), and Variant 2 (V2).

Addresses Issue #2 analytic evidence requirements:
  - Reproduce the two-arm (10s, 0.01s) vs (0, 0) calculation
  - Independent sweep of large and small coordinate gaps
  - Clearly resolved dominance, confidently incomparable arms, within-F1 ties
  - Pure dilution stress test (1 ambiguous pair among M=2, 10, 50 distant pairs)
  - Under-sampled arm below F2 (moderate and extreme variance ratios)
  - Empty F2 under diffuse vs concentrated posteriors
  - Shared-dominator and shared-challenger dependence
  - Wall-clock runtime comparison against direct Monte Carlo disagreement
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from paretoset import paretoset
from scipy.special import ndtr

from experiments.uq.metrics import (
    uncertainty_from_moments,
    uq_directional_bc_from_moments,
    uq_dominance_risk_from_moments,
    uq_mc_disagreement_from_moments,
)


REPORT_PATH = Path(__file__).resolve().parent / "reports" / "analytic_checks.json"


def _eq10_with_zero_fallback(means: np.ndarray, stds: np.ndarray) -> tuple[float, bool]:
    res = uncertainty_from_moments(means, stds)
    if not res.available or res.bhattacharyya is None:
        return 0.0, False
    return float(res.bhattacharyya), True


def _two_arm_exact_disagreement(means: np.ndarray, stds: np.ndarray) -> float:
    z = (means[0] - means[1]) / np.sqrt(stds[0] ** 2 + stds[1] ** 2)
    p01 = float(np.prod(ndtr(z)))
    p10 = float(np.prod(ndtr(-z)))
    f1 = paretoset(means, sense=["max"] * means.shape[1], distinct=False)
    if f1.all():
        # Both arms in F1 -> disagree iff one dominates the other
        return float(p01 + p10)
    if f1[0]:
        return float(1.0 - p01)
    return float(1.0 - p10)


def _distant_filler_pairs(n_filler: int, s: float) -> tuple[np.ndarray, np.ndarray]:
    """Create n_filler/2 distant Pareto arms and n_filler/2 dominated arms
    separated by >= 100s on every gap so exact q_t stays unchanged."""
    half = n_filler // 2
    if half == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    # First filler block at offset (+1000s, -1000s) spaced by 100s along anti-diagonal
    # Wait: keep arm 0 at (10s, 0.01s) and arm 1 at (0, 0) strictly in positive quadrant,
    # and place filler Pareto arms far along high-X, negative-Y OR negative-X, high-Y?
    # Better: place Pareto fillers at (1000s + 100k*s, -1000s - 100k*s) — Wait! Then (10s, 0.01s)
    # still dominates negative? No, 10s > 1000s is false, 0.01s > -1000s is true, so incomparable!
    idx = np.arange(1, half + 1, dtype=np.float64)
    p_arms = np.column_stack([1000.0 * s + 100.0 * s * idx, -1000.0 * s - 100.0 * s * idx])
    # Dominated fillers strictly below BOTH filler Pareto arms by 100s on both coordinates
    d_arms = p_arms - 100.0 * s
    # Keep arm 0 `(10s, 0.01s)` undominated and dominating only arm 1 `(0,0)`:
    # Note for p_arms: coord1 >= 1100s > 10s, coord2 <= -1100s < 0 => incomparable to (10s, 0.01s) and (0,0)!
    #Wait: could d_arms be dominated by (10s, 0.01s)? No, d_arms has coord1 >= 1000s > 10s! Only p_arms dominates d_arms!
    all_means = np.vstack([p_arms, d_arms])
    all_stds = np.full_like(all_means, s)
    return all_means, all_stds


def run_checks() -> dict:
    s = 0.05
    results: dict = {"unit_cases": [], "gap_sweep": [], "timings_ms": []}

    # Exact two-arm formula sanity
    m_tie = np.array([[10.0 * s, 0.01 * s], [0.0, 0.0]])
    sd_tie = np.full((2, 2), s)
    p_same = float(ndtr(10.0 / np.sqrt(2.0)) * ndtr(0.01 / np.sqrt(2.0)))
    assert abs(p_same - 0.5028204) < 1e-4, f"unexpected p_same={p_same}"

    # Pure dilution filler sets
    f10_m, f10_s = _distant_filler_pairs(8, s)
    f50_m, f50_s = _distant_filler_pairs(48, s)

    cases = [
        ("two_arm_coordinate_tie_10s_0.01s",
         m_tie, sd_tie,
         {"eq10_max": 1e-5, "v1_min": 0.45, "v2_min": 0.45, "mc_target": 1.0 - p_same, "mc_tol": 0.02}),
        ("resolved_dominance_10s_10s",
         np.array([[10.0 * s, 10.0 * s], [0.0, 0.0]]), np.full((2, 2), s),
         {"eq10_max": 1e-6, "v1_max": 1e-6, "v2_max": 1e-6}),
        ("confidently_incomparable",
         np.array([[10.0 * s, -10.0 * s], [-10.0 * s, 10.0 * s]]), np.full((2, 2), s),
         {"eq10_max": 1e-6, "v1_max": 1e-6, "v2_max": 1e-6}),
        ("uncertain_within_F1_empty_F2",
         np.array([[10.0 * s, -0.01 * s], [0.0, 0.0]]), np.full((2, 2), s),
         {"eq10_avail": False, "v1_min": 0.45, "v2_min": 0.45, "mc_target": 1.0 - p_same, "mc_tol": 0.02}),
        ("dilution_M2_baseline",
         m_tie, sd_tie,
         {"v1_min": 0.45, "v2_min": 0.45}),
        ("dilution_M10_distant_pairs",
         np.vstack([m_tie, f10_m]), np.vstack([sd_tie, f10_s]),
         {"v1_min": 0.45, "v2_min": 0.45, "mc_target": 1.0 - p_same, "mc_tol": 0.025}),
        ("dilution_M50_distant_pairs",
         np.vstack([m_tie, f50_m]), np.vstack([sd_tie, f50_s]),
         {"v1_min": 0.45, "v2_min": 0.45, "mc_target": 1.0 - p_same, "mc_tol": 0.025}),
        ("undersampled_below_F2_moderate",
         np.array([[1.0, 1.0], [0.7, 0.7], [0.6, 0.6]]),
         np.array([[0.05, 0.05], [0.05, 0.05], [0.5, 0.5]]),
         {"eq10_max": 1e-3, "v1_min": 0.10, "v2_min": 0.30, "v2_tol_mc": 0.03}),
        ("undersampled_below_F2_extreme_counterexample",
         np.array([[1.0, 1.0], [0.7, 0.7], [0.95, 0.95]]),
         np.array([[0.0005, 0.0005], [0.0005, 0.0005], [0.5, 0.5]]),
         {"eq10_max": 0.01, "v1_max_counterexample": 0.02, "v2_min": 0.65, "v2_tol_mc": 0.03}),
        ("empty_F2_diffuse",
         np.array([[0.1, 0.0], [0.0, 0.1]]), np.full((2, 2), 0.5),
         {"eq10_avail": False, "v1_min": 0.40, "v2_min": 0.40, "v2_tol_mc": 0.03}),
        ("empty_F2_concentrated",
         np.array([[0.1, 0.0], [0.0, 0.1]]), np.full((2, 2), 0.001),
         {"eq10_avail": False, "v1_max": 1e-6, "v2_max": 1e-6}),
        ("shared_dominator_20_aligned",
         np.vstack([[[0.8, 0.8]], np.tile([0.5, 0.5], (20, 1))]),
         np.vstack([[[0.158, 0.158]], np.tile([0.005, 0.005], (20, 1))]),
         {"v2_max": 0.12, "v2_tol_mc": 0.03}),
        ("shared_dominator_orthogonal_coords",
         np.array([[0.8, 0.8], [0.79, 0.0], [0.0, 0.79]]),
         np.array([[0.2, 0.2], [0.001, 0.001], [0.001, 0.001]]),
         {"v2_min": 0.68, "v2_tol_mc": 0.03}),
        ("shared_noisy_challenger_10_optimal",
         np.vstack([
             np.tile([0.5, 0.5], (10, 1)) + np.column_stack([np.linspace(-0.01, 0.01, 10), np.linspace(0.01, -0.01, 10)]),
             [[0.4, 0.4]],
         ]),
         np.vstack([np.tile([0.002, 0.002], (10, 1)), [[0.5, 0.5]]]),
         {"v1_min": 0.90, "v2_min": 0.95, "v2_tol_mc": 0.02}),
    ]

    header = f"{'Case':<44} | {'Eq10':>8} | {'V1':>8} | {'V2':>8} | {'MC_q (20k)':>14}"
    print(header)
    print("-" * len(header))

    for name, means, stds, checks in cases:
        eq10_val, eq10_avail = _eq10_with_zero_fallback(means, stds)
        v1 = uq_directional_bc_from_moments(means, stds)
        v2 = uq_dominance_risk_from_moments(means, stds)
        mc_q, mc_se = uq_mc_disagreement_from_moments(means, stds, n_draws=20000, seed=20261100)

        if "eq10_max" in checks:
            assert eq10_val <= checks["eq10_max"], f"{name}: eq10={eq10_val}"
        if "eq10_avail" in checks:
            assert eq10_avail is checks["eq10_avail"], f"{name}: eq10_avail={eq10_avail}"
        if "v1_min" in checks:
            assert v1 >= checks["v1_min"], f"{name}: v1={v1}"
        if "v1_max" in checks:
            assert v1 <= checks["v1_max"], f"{name}: v1={v1}"
        if "v1_max_counterexample" in checks:
            assert v1 <= checks["v1_max_counterexample"], f"{name}: expected counterexample v1<={checks['v1_max_counterexample']}, got {v1}"
        if "v2_min" in checks:
            assert v2 >= checks["v2_min"], f"{name}: v2={v2}"
        if "v2_max" in checks:
            assert v2 <= checks["v2_max"], f"{name}: v2={v2}"
        if "v2_tol_mc" in checks:
            assert abs(v2 - mc_q) <= checks["v2_tol_mc"], f"{name}: |v2-mc|={abs(v2-mc_q)}"
        if "mc_target" in checks:
            assert abs(mc_q - checks["mc_target"]) <= checks["mc_tol"], f"{name}: mc={mc_q}"

        results["unit_cases"].append({
            "name": name,
            "k": int(means.shape[0]),
            "d": int(means.shape[1]),
            "eq10": eq10_val,
            "eq10_available": eq10_avail,
            "v1": v1,
            "v2": v2,
            "mc_q": mc_q,
            "mc_se": mc_se,
        })
        print(f"{name:<44} | {eq10_val:8.5f} | {v1:8.5f} | {v2:8.5f} | {mc_q:6.4f}±{1.96*mc_se:6.4f}")

    # Independent sweep over large gap d1/s in [0, 10] and small gap d2/s in [-10, 10]
    for z1 in (0.0, 2.0, 5.0, 10.0):
        for z2 in (-10.0, -2.0, -0.01, 0.01, 0.5, 2.0, 10.0):
            m = np.array([[z1 * s, z2 * s], [0.0, 0.0]])
            sd = np.full((2, 2), s)
            eq10_val, eq10_avail = _eq10_with_zero_fallback(m, sd)
            v1 = uq_directional_bc_from_moments(m, sd)
            v2 = uq_dominance_risk_from_moments(m, sd)
            exact_q = _two_arm_exact_disagreement(m, sd)
            assert abs(v2 - exact_q) < 1e-12, f"Two-arm exact mismatch at ({z1},{z2}): {v2} vs {exact_q}"
            results["gap_sweep"].append({
                "z1": z1, "z2": z2, "eq10": eq10_val, "eq10_available": eq10_avail,
                "v1": v1, "v2": v2, "exact_q": exact_q,
            })

    # Wall-clock cost scaling across K = 10, 60, 200 (d = 2) and K = 50 (d = 10)
    rng = np.random.default_rng(20261100)
    print("\nTiming comparison (ms per single checkpoint call):")
    for k, d in ((10, 2), (60, 2), (200, 2), (50, 10)):
        m = rng.uniform(0.0, 1.0, size=(k, d))
        sd = rng.uniform(0.01, 0.25, size=(k, d))
        repeats = 200 if k < 200 else 50
        t0 = time.perf_counter()
        for _ in range(repeats):
            _eq10_with_zero_fallback(m, sd)
        t_eq10 = (time.perf_counter() - t0) * 1000.0 / repeats

        t0 = time.perf_counter()
        for _ in range(repeats):
            uq_directional_bc_from_moments(m, sd)
        t_v1 = (time.perf_counter() - t0) * 1000.0 / repeats

        t0 = time.perf_counter()
        for _ in range(repeats):
            uq_dominance_risk_from_moments(m, sd)
        t_v2 = (time.perf_counter() - t0) * 1000.0 / repeats

        t0 = time.perf_counter()
        for _ in range(max(10, repeats // 4)):
            uq_mc_disagreement_from_moments(m, sd, n_draws=512, seed=1)
        t_mc512 = (time.perf_counter() - t0) * 1000.0 / max(10, repeats // 4)

        results["timings_ms"].append({
            "k": k, "d": d, "eq10_ms": t_eq10, "v1_ms": t_v1, "v2_ms": t_v2, "mc512_ms": t_mc512,
        })
        print(f"  K={k:3d}, d={d:2d}: Eq10={t_eq10:6.3f}ms | V1={t_v1:6.3f}ms | V2={t_v2:6.3f}ms | MC512={t_mc512:7.3f}ms")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nAll analytic assertions PASSED. Report saved to {REPORT_PATH}")
    return results


if __name__ == "__main__":
    run_checks()
