#!/usr/bin/env python3
"""Held-out random-instance evaluation (checkpoints, stopping rules, and trajectory correlations)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from paretoset import paretoset
from scipy.special import ndtr

from bandits.Posteriors import NormalPosterior
from bandits.TTPFTS import TTPFTSBandit
from experiments.uq.metrics import (
    uncertainty_from_moments,
    uq_directional_bc_from_moments,
    uq_dominance_risk_from_moments,
    uq_mc_disagreement_from_moments,
)
from runner.schema import recompute_metrics

from .config import load_config


def _random_means(rng: np.random.Generator, shape: tuple[int, ...], law: str) -> np.ndarray:
    if law == "uniform":
        return rng.uniform(0.0, 1.0, shape)
    if law == "normal":
        return rng.normal(0.5, 1.0 / math.sqrt(12.0), shape)
    raise ValueError(f"unknown mean law: {law}")


def _pareto_mask_2d(means: np.ndarray) -> np.ndarray:
    return paretoset(means, sense=["max"] * means.shape[1], distinct=False)


def _wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def _eq10_score(means: np.ndarray, stds: np.ndarray) -> tuple[float, bool]:
    res = uncertainty_from_moments(means, stds)
    if not res.available or res.bhattacharyya is None:
        return 0.0, False
    return float(res.bhattacharyya), True


# ---------------------------------------------------------------------------
# Part R1: Held-out Checkpoint & Stopping-Rule Benchmark (seed = 20261116)
# ---------------------------------------------------------------------------

R1_SCENARIOS = [
    {"name": "uniform_K2_d2_s0.5", "law": "uniform", "k": 2, "d": 2, "sigma": 0.5, "repeats": 50000},
    {"name": "normal_K2_d2_s0.5", "law": "normal", "k": 2, "d": 2, "sigma": 0.5, "repeats": 50000},
    {"name": "uniform_K10_d2_s0.5", "law": "uniform", "k": 10, "d": 2, "sigma": 0.5, "repeats": 3000},
    {"name": "normal_K10_d2_s0.5", "law": "normal", "k": 10, "d": 2, "sigma": 0.5, "repeats": 3000},
    {"name": "uniform_K10_d4_s0.5", "law": "uniform", "k": 10, "d": 4, "sigma": 0.5, "repeats": 3000},
    {"name": "uniform_K20_d2_s0.5", "law": "uniform", "k": 20, "d": 2, "sigma": 0.5, "repeats": 2000},
    {"name": "uniform_K10_d2_s0.25", "law": "uniform", "k": 10, "d": 2, "sigma": 0.25, "repeats": 3000},
    {"name": "uniform_K50_d2_s0.5", "law": "uniform", "k": 50, "d": 2, "sigma": 0.5, "repeats": 1000},
]
R1_CHECKPOINTS = (10, 25, 100)  # pulls per arm


def _run_two_arm_r1(scenario: dict[str, Any], seed: int, mc_draws: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    law = scenario["law"]
    sigma = float(scenario["sigma"])
    repeats = int(scenario["repeats"])
    out_checkpoints: dict[str, Any] = {}

    for per_arm in R1_CHECKPOINTS:
        truth_mu = _random_means(rng, (repeats, 2, 2), law)
        pulls = 2 * per_arm
        n0 = 1 + rng.binomial(pulls - 2, 0.5, repeats)
        counts = np.column_stack((n0, pulls - n0))
        vars_ = (sigma ** 2) / counts
        stds = np.sqrt(vars_)
        centers = truth_mu + rng.normal(size=truth_mu.shape) * stds[..., None]

        # Vectorized truth & estimate fronts
        f_hat = np.ones((repeats, 2), dtype=bool)
        dom01 = np.all(centers[:, 0] >= centers[:, 1], axis=-1) & np.any(centers[:, 0] > centers[:, 1], axis=-1)
        dom10 = np.all(centers[:, 1] >= centers[:, 0], axis=-1) & np.any(centers[:, 1] > centers[:, 0], axis=-1)
        f_hat[dom01, 1] = False
        f_hat[dom10, 0] = False

        f_true = np.ones((repeats, 2), dtype=bool)
        tdom01 = np.all(truth_mu[:, 0] >= truth_mu[:, 1], axis=-1) & np.any(truth_mu[:, 0] > truth_mu[:, 1], axis=-1)
        tdom10 = np.all(truth_mu[:, 1] >= truth_mu[:, 0], axis=-1) & np.any(truth_mu[:, 1] > truth_mu[:, 0], axis=-1)
        f_true[tdom01, 1] = False
        f_true[tdom10, 0] = False

        jaccard = (f_hat & f_true).sum(axis=1) / (f_hat | f_true).sum(axis=1)
        errors = (jaccard < 1.0).astype(int)
        has_second = ~f_hat.all(axis=1)

        # Eq 10 (returns 0.0 when empty second front; track raw separately)
        v_avg = 0.5 * (vars_[:, 0] + vars_[:, 1])
        diff2 = np.sum((centers[:, 0] - centers[:, 1]) ** 2, axis=-1)
        db = 0.125 * diff2 / v_avg + 2.0 * 0.5 * np.log(v_avg / (stds[:, 0] * stds[:, 1]))
        bc_pair = np.exp(-db)
        eq10_scores = np.where(has_second, bc_pair, 0.0)

        # Exact 2-arm MC reference (closed form q_t under Gaussian posterior)
        z = (centers[:, 0] - centers[:, 1]) / np.sqrt(vars_.sum(axis=1))[:, None]
        p01 = ndtr(z).prod(axis=1)
        p10 = ndtr(-z).prod(axis=1)
        p_correct = np.where(f_hat.all(axis=1), 1.0 - p01 - p10, np.where(f_hat[:, 0], p01, p10))
        q_exact = 1.0 - p_correct

        # Vectorized V1 (2 arms)
        sd_broad = stds[..., None] * np.ones((1, 1, 2))
        s_prod = sd_broad[:, 0] * sd_broad[:, 1]
        v_sum = vars_.sum(axis=1)[:, None]
        bc_1d = np.sqrt((2.0 * s_prod) / v_sum) * np.exp(-0.25 * ((centers[:, 0] - centers[:, 1]) ** 2) / v_sum)
        q_c = np.clip(0.5 * (bc_1d ** 2), 0.0, 0.5)
        esc_01 = 1.0 - np.prod(1.0 - q_c, axis=-1)
        def01 = centers[:, 0] <= centers[:, 1]
        def10 = centers[:, 1] <= centers[:, 0]
        p_01_v1 = np.prod(np.where(def01, q_c, 1.0 - q_c), axis=-1)
        p_10_v1 = np.prod(np.where(def10, q_c, 1.0 - q_c), axis=-1)
        v1_scores = np.where(has_second, esc_01, np.minimum(1.0, p_01_v1 + p_10_v1))
        # V2 on 2 arms equals exact q_t analytically
        v2_scores = q_exact.copy()

        out_checkpoints[str(per_arm)] = _summarize_checkpoint(
            scores={"eq10": eq10_scores, "uq_v1": v1_scores, "uq_v2": v2_scores, "uq_mc": q_exact},
            has_second=has_second,
            errors=errors,
            jaccard=jaccard,
        )

    return {"scenario": scenario["name"], "type": "two_arm_analytic", "checkpoints": out_checkpoints}


def _summarize_checkpoint(
    *,
    scores: dict[str, np.ndarray],
    has_second: np.ndarray,
    errors: np.ndarray,
    jaccard: np.ndarray,
) -> dict[str, Any]:
    n = int(len(errors))
    wrong = int(errors.sum())
    err_rate, lo, hi = _wilson_ci(wrong, n)
    out: dict[str, Any] = {
        "n": n,
        "wrong": wrong,
        "error_rate": err_rate,
        "error_ci_low": lo,
        "error_ci_high": hi,
        "mean_jaccard_loss": float(1.0 - jaccard.mean()),
        "empty_second_count": int((~has_second).sum()),
        "empty_second_wrong": int(errors[~has_second].sum()),
        "metrics": {},
    }

    bins = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 1.01)]
    for m_name, arr in scores.items():
        m_info: dict[str, Any] = {"bins": [], "coverage": {}, "eq10_legacy_thresholds": {}}
        for b_lo, b_hi in bins:
            if b_lo == 0.0:
                mask = (arr >= 0.0) & (arr <= b_hi)
            else:
                mask = (arr > b_lo) & (arr <= b_hi)
            bn = int(mask.sum())
            bw = int(errors[mask].sum())
            ber, blo, bhi = _wilson_ci(bw, bn)
            m_info["bins"].append({
                "range": f"[{b_lo:.2f},{min(b_hi, 1.0):.2f}]" if b_lo == 0 else f"({b_lo:.2f},{min(b_hi, 1.0):.2f}]",
                "n": bn,
                "wrong": bw,
                "error_rate": ber if bn else None,
                "error_ci_low": blo if bn else None,
                "error_ci_high": bhi if bn else None,
                "mean_jaccard_loss": float(1.0 - jaccard[mask].mean()) if bn else None,
            })

        if m_name == "eq10":
            for thr in (0.01, 0.0001, 0.000004):
                for mode, mask in (("all", arr <= thr), ("nonempty_second", has_second & (arr <= thr))):
                    tn = int(mask.sum())
                    tw = int(errors[mask].sum())
                    ter, tlo, thi = _wilson_ci(tw, tn)
                    m_info["eq10_legacy_thresholds"][f"{mode}_le_{thr}"] = {
                        "n": tn, "wrong": tw, "error_rate": ter if tn else None,
                        "ci_low": tlo if tn else None, "ci_high": thi if tn else None,
                        "mean_jaccard_loss": float(1.0 - jaccard[mask].mean()) if tn else None,
                    }

        # Matched-coverage accuracy: lowest-score q fraction of runs
        order = np.argsort(arr, kind="mergesort")
        for cov in (0.10, 0.25, 0.50):
            top_k = max(1, int(round(cov * n)))
            idx = order[:top_k]
            cw = int(errors[idx].sum())
            cer, clo, chi = _wilson_ci(cw, top_k)
            m_info["coverage"][f"{cov:.2f}"] = {
                "n": top_k,
                "wrong": cw,
                "accuracy": 1.0 - cer,
                "error_rate": cer,
                "score_cutoff": float(arr[idx[-1]]),
                "mean_jaccard_loss": float(1.0 - jaccard[idx].mean()),
            }
        out["metrics"][m_name] = m_info
    return out


def _adaptive_r1_worker(
    chunk_id: int,
    truth_mu_chunk: np.ndarray,
    truth_mask_chunk: np.ndarray,
    k: int,
    d: int,
    sigma: float,
    max_pulls: int,
    mc_draws: int,
    seed: int,
) -> dict[str, Any]:
    traj_seed, mc_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(traj_seed)
    mc_rng = np.random.default_rng(mc_seed)
    n_runs = truth_mu_chunk.shape[0]
    checkpoints_raw = [c * k for c in R1_CHECKPOINTS]
    cp_set = set(checkpoints_raw)

    stop_thresholds = {
        "eq10": [0.01, 0.0001, 0.000004],
        "uq_v1": [0.10, 0.05, 0.01],
        "uq_v2": [0.10, 0.05, 0.01],
        "uq_mc": [0.10, 0.05, 0.01],
    }

    cp_records: dict[int, dict[str, list]] = {
        cp: {
            "eq10": [], "has_second": [], "uq_v1": [], "uq_v2": [], "uq_mc": [],
            "errors": [], "jaccard": [], "t_eq10": [], "t_v1": [], "t_v2": [], "t_mc": [],
        }
        for cp in checkpoints_raw
    }

    stopping_records: dict[str, dict[float, dict[str, list]]] = {
        m: {
            thr: {"crossed": [], "error": [], "jaccard": [], "pulls": []}
            for thr in thrs
        }
        for m, thrs in stop_thresholds.items()
    }

    for r in range(n_runs):
        mu_true = truth_mu_chunk[r]
        true_mask = truth_mask_chunk[r]
        counts = np.ones(k, dtype=int)
        centers = mu_true + rng.normal(0.0, sigma, size=(k, d))

        crossed_flags = {m: {thr: False for thr in thrs} for m, thrs in stop_thresholds.items()}
        final_err = 0
        final_jac = 1.0

        for pulls in range(k, max_pulls + 1):
            if pulls % k == 0:
                stds = (sigma / np.sqrt(counts))[:, None] * np.ones((1, d))
                f_hat = _pareto_mask_2d(centers)
                jac = float(np.sum(f_hat & true_mask) / np.sum(f_hat | true_mask))
                err = int(jac < 1.0)
                final_err, final_jac = err, jac

                t0 = time.perf_counter()
                eq10_val, eq10_nonempty = _eq10_score(centers, stds)
                t_eq10 = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                v1_val = uq_directional_bc_from_moments(centers, stds)
                t_v1 = (time.perf_counter() - t0) * 1000.0

                t0 = time.perf_counter()
                v2_val = uq_dominance_risk_from_moments(centers, stds)
                t_v2 = (time.perf_counter() - t0) * 1000.0

                # Evaluate MC at monitoring points for stopping comparisons (smaller batch for non-cp if K<=20 else 256)
                t0 = time.perf_counter()
                mc_val, _ = uq_mc_disagreement_from_moments(
                    centers, stds, n_draws=(mc_draws if pulls in cp_set else min(mc_draws, 256)), rng=mc_rng
                )
                t_mc = (time.perf_counter() - t0) * 1000.0

                vals = {"eq10": eq10_val, "uq_v1": v1_val, "uq_v2": v2_val, "uq_mc": mc_val}

                if pulls in cp_set:
                    rec = cp_records[pulls]
                    rec["eq10"].append(eq10_val)
                    rec["has_second"].append(eq10_nonempty)
                    rec["uq_v1"].append(v1_val)
                    rec["uq_v2"].append(v2_val)
                    rec["uq_mc"].append(mc_val)
                    rec["errors"].append(err)
                    rec["jaccard"].append(jac)
                    rec["t_eq10"].append(t_eq10)
                    rec["t_v1"].append(t_v1)
                    rec["t_v2"].append(t_v2)
                    rec["t_mc"].append(t_mc)

                for m_name, thrs in stop_thresholds.items():
                    score_val = vals[m_name]
                    for thr in thrs:
                        if not crossed_flags[m_name][thr]:
                            # Require non-empty second front for Eq10 stopping rule so empty-front 0 doesn't stop spuriously
                            cond = (score_val <= thr) and (eq10_nonempty if m_name == "eq10" else True)
                            if cond:
                                crossed_flags[m_name][thr] = True
                                srec = stopping_records[m_name][thr]
                                srec["crossed"].append(1)
                                srec["error"].append(err)
                                srec["jaccard"].append(jac)
                                srec["pulls"].append(pulls)

            if pulls == max_pulls:
                break

            stds_arm = (sigma / np.sqrt(counts))[:, None]
            draw = centers + rng.normal(size=(k, d)) * stds_arm
            f1_d = _pareto_mask_2d(draw)
            if rng.random() < 0.5 or np.all(f1_d):
                arm = int(rng.choice(np.flatnonzero(f1_d)))
            else:
                rem = np.flatnonzero(~f1_d)
                f2_mask = _pareto_mask_2d(draw[rem])
                arm = int(rng.choice(rem[f2_mask]))
            reward = mu_true[arm] + rng.normal(0.0, sigma, size=d)
            counts[arm] += 1
            centers[arm] += (reward - centers[arm]) / counts[arm]

        for m_name, thrs in stop_thresholds.items():
            for thr in thrs:
                if not crossed_flags[m_name][thr]:
                    srec = stopping_records[m_name][thr]
                    srec["crossed"].append(0)
                    srec["error"].append(final_err)
                    srec["jaccard"].append(final_jac)
                    srec["pulls"].append(max_pulls)

    return {"cp_records": cp_records, "stopping_records": stopping_records}


def run_r1_suite(output_dir: Path, n_jobs: int, smoke: bool = False) -> dict[str, Any]:
    cfg = load_config()
    seed = int(cfg["heldout_seeds"]["checkpoint_stopping_seed"])
    mc_draws = 128 if smoke else int(cfg["mc_draws"])
    scenarios = []
    for item in R1_SCENARIOS:
        sc = dict(item)
        if smoke:
            sc["repeats"] = 100 if sc["k"] == 2 else 30
        scenarios.append(sc)

    sub_seeds = np.random.SeedSequence(seed).spawn(len(scenarios))
    results: dict[str, Any] = {"seed": seed, "smoke": smoke, "scenarios": {}}

    for scenario, s_seed in zip(scenarios, sub_seeds):
        name = scenario["name"]
        k, d, sigma, repeats = scenario["k"], scenario["d"], scenario["sigma"], scenario["repeats"]
        print(f"[R1] Running {name} (K={k}, d={d}, sigma={sigma}, repeats={repeats})...", flush=True)
        if k == 2:
            results["scenarios"][name] = _run_two_arm_r1(scenario, int(s_seed.generate_state(1)[0]), mc_draws)
            continue

        n_chunks = min(repeats, max(1, n_jobs * 2))
        indices = np.array_split(np.arange(repeats), n_chunks)
        spawned = s_seed.spawn(len(indices) + 1)
        rng = np.random.default_rng(spawned[0])
        truth_mu = _random_means(rng, (repeats, k, d), scenario["law"])
        left = truth_mu[:, :, None, :]
        right = truth_mu[:, None, :, :]
        truth_mask = ~np.any(np.all(left >= right, axis=-1) & np.any(left > right, axis=-1), axis=1)

        max_pulls = k * max(R1_CHECKPOINTS)
        chunk_seeds = spawned[1:]

        if n_jobs == 1:
            worker_out = [
                _adaptive_r1_worker(
                    idx_c, truth_mu[idx], truth_mask[idx], k, d, sigma, max_pulls, mc_draws,
                    int(cs.generate_state(1)[0]),
                )
                for idx_c, (idx, cs) in enumerate(zip(indices, chunk_seeds))
            ]
        else:
            worker_out = Parallel(n_jobs=n_jobs)(
                delayed(_adaptive_r1_worker)(
                    idx_c, truth_mu[idx], truth_mask[idx], k, d, sigma, max_pulls, mc_draws,
                    int(cs.generate_state(1)[0]),
                )
                for idx_c, (idx, cs) in enumerate(zip(indices, chunk_seeds))
            )

        cp_summaries: dict[str, Any] = {}
        for cp_mul in R1_CHECKPOINTS:
            cp = cp_mul * k
            merged = {
                key: np.concatenate([np.asarray(w["cp_records"][cp][key]) for w in worker_out])
                for key in worker_out[0]["cp_records"][cp]
            }
            summary = _summarize_checkpoint(
                scores={
                    "eq10": merged["eq10"],
                    "uq_v1": merged["uq_v1"],
                    "uq_v2": merged["uq_v2"],
                    "uq_mc": merged["uq_mc"],
                },
                has_second=merged["has_second"].astype(bool),
                errors=merged["errors"].astype(int),
                jaccard=merged["jaccard"].astype(float),
            )
            summary["timings_ms"] = {
                "eq10": float(merged["t_eq10"].mean()),
                "uq_v1": float(merged["t_v1"].mean()),
                "uq_v2": float(merged["t_v2"].mean()),
                "uq_mc": float(merged["t_mc"].mean()),
            }
            cp_summaries[str(cp_mul)] = summary

        stop_summaries: dict[str, Any] = {}
        for m_name in worker_out[0]["stopping_records"]:
            stop_summaries[m_name] = {}
            for thr in worker_out[0]["stopping_records"][m_name]:
                crossed = np.concatenate([np.asarray(w["stopping_records"][m_name][thr]["crossed"]) for w in worker_out]).astype(bool)
                err = np.concatenate([np.asarray(w["stopping_records"][m_name][thr]["error"]) for w in worker_out]).astype(int)
                jac = np.concatenate([np.asarray(w["stopping_records"][m_name][thr]["jaccard"]) for w in worker_out]).astype(float)
                pulls = np.concatenate([np.asarray(w["stopping_records"][m_name][thr]["pulls"]) for w in worker_out]).astype(int)

                n_c = int(crossed.sum())
                w_c = int(err[crossed].sum())
                er_c, lo_c, hi_c = _wilson_ci(w_c, n_c)

                n_nc = int((~crossed).sum())
                w_nc = int(err[~crossed].sum())
                er_nc, _, _ = _wilson_ci(w_nc, n_nc)

                er_all, lo_all, hi_all = _wilson_ci(int(err.sum()), len(err))
                stop_summaries[m_name][str(thr)] = {
                    "n_total": int(len(err)),
                    "n_crossed": n_c,
                    "frac_crossed": float(n_c / len(err)),
                    "error_at_crossing": er_c if n_c else None,
                    "error_at_crossing_ci_low": lo_c if n_c else None,
                    "error_at_crossing_ci_high": hi_c if n_c else None,
                    "mean_jaccard_at_crossing": float(jac[crossed].mean()) if n_c else None,
                    "mean_pulls_crossed": float(pulls[crossed].mean()) if n_c else None,
                    "mean_pulls_all": float(pulls.mean()),
                    "never_crossed_frac": float(n_nc / len(err)),
                    "error_among_never_crossed": er_nc if n_nc else None,
                    "overall_stopped_or_horizon_error": er_all,
                    "overall_error_ci_low": lo_all,
                    "overall_error_ci_high": hi_all,
                }

        results["scenarios"][name] = {
            "scenario": name,
            "type": "adaptive_ttpts",
            "k": k, "d": d, "sigma": sigma, "max_pulls": max_pulls,
            "checkpoints": cp_summaries,
            "stopping": stop_summaries,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "random_checkpoint_stopping.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved R1 results -> {out_json}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Part R2: Held-out Random-Instance Trajectory Correlations (seed = 20261117)
# ---------------------------------------------------------------------------

TRAJ_COLUMNS = [
    "family", "law", "num_arms", "num_objectives", "sigma", "repetition",
    "arm_pull", "max_budget", "normalized_budget",
    "bernoulli", "jaccard", "misidentification",
    "uq_available", "eq10_nonempty_second", "bhattacharyya", "bhattacharyya_raw",
    "uq_v1", "uq_v2", "uq_mc", "uq_mc_se",
    "time_eq10_ms", "time_v1_ms", "time_v2_ms", "time_mc_ms",
]


def _run_r2_repetition(
    family: dict[str, Any],
    repetition: int,
    eval_points: int,
    mc_draws: int,
    seq: np.random.SeedSequence,
) -> list[dict]:
    fam_seed_children = seq.spawn(2)
    rng = np.random.default_rng(fam_seed_children[0])
    mc_rng = np.random.default_rng(fam_seed_children[1])
    k = int(family["k"])
    d = int(family["d"])
    sigma = float(family["sigma"])
    budget = int(family["budget"])
    law = str(family["law"])

    truth_mu = _random_means(rng, (k, d), law)
    true_mask = _pareto_mask_2d(truth_mu)
    true_pareto = set(np.flatnonzero(true_mask).tolist())

    grid = sorted({0, budget, *(round(i * budget / eval_points) for i in range(eval_points + 1))})
    grid_set = set(grid)

    # Known-variance Gaussian posterior with 4-pull warm-up matching paper protocol
    # Drive bandit decisions with local rng-seeded numpy/random state so workers are parallel-safe
    warmup = 4
    counts = np.zeros((k, d), dtype=int)
    means = np.zeros((k, d), dtype=np.float64)
    warmup_pulls = np.zeros(k, dtype=int)
    current_warmup = 0

    rows = []

    def record(pull: int) -> None:
        if pull == 0 or np.any(counts <= 0):
            # Initial point: uniform recommendation if unpulled
            rec = set(range(k)) if pull == 0 else set(np.flatnonzero(_pareto_mask_2d(means)).tolist())
            m_dict = recompute_metrics(rec, true_pareto, k)
            rows.append({
                "family": family["name"], "law": law, "num_arms": k, "num_objectives": d,
                "sigma": sigma, "repetition": repetition, "arm_pull": pull, "max_budget": budget,
                "normalized_budget": pull / budget,
                "bernoulli": m_dict["bernoulli"], "jaccard": m_dict["jaccard"],
                "misidentification": m_dict["misidentification"],
                "uq_available": 0, "eq10_nonempty_second": 0,
                "bhattacharyya": "", "bhattacharyya_raw": "",
                "uq_v1": "", "uq_v2": "", "uq_mc": "", "uq_mc_se": "",
                "time_eq10_ms": "", "time_v1_ms": "", "time_v2_ms": "", "time_mc_ms": "",
            })
            return

        stds = sigma / np.sqrt(counts)
        rec = set(np.flatnonzero(_pareto_mask_2d(means)).tolist())
        m_dict = recompute_metrics(rec, true_pareto, k)

        t0 = time.perf_counter()
        eq10_val, eq10_nonempty = _eq10_score(means, stds)
        t_eq10 = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        v1_val = uq_directional_bc_from_moments(means, stds)
        t_v1 = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        v2_val = uq_dominance_risk_from_moments(means, stds)
        t_v2 = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        mc_val, mc_se = uq_mc_disagreement_from_moments(means, stds, n_draws=mc_draws, rng=mc_rng)
        t_mc = (time.perf_counter() - t0) * 1000.0

        rows.append({
            "family": family["name"], "law": law, "num_arms": k, "num_objectives": d,
            "sigma": sigma, "repetition": repetition, "arm_pull": pull, "max_budget": budget,
            "normalized_budget": pull / budget,
            "bernoulli": m_dict["bernoulli"], "jaccard": m_dict["jaccard"],
            "misidentification": m_dict["misidentification"],
            "uq_available": 1, "eq10_nonempty_second": int(eq10_nonempty),
            "bhattacharyya": eq10_val,
            "bhattacharyya_raw": eq10_val if eq10_nonempty else "",
            "uq_v1": v1_val, "uq_v2": v2_val, "uq_mc": mc_val, "uq_mc_se": mc_se,
            "time_eq10_ms": t_eq10, "time_v1_ms": t_v1, "time_v2_ms": t_v2, "time_mc_ms": t_mc,
        })

    if 0 in grid_set:
        record(0)

    for t in range(1, budget + 1):
        if np.any(warmup_pulls < warmup):
            arm = current_warmup
            warmup_pulls[arm] += 1
            current_warmup = (current_warmup + 1) % k
        else:
            stds = sigma / np.sqrt(counts)
            draw = means + rng.normal(size=(k, d)) * stds
            f1 = _pareto_mask_2d(draw)
            if rng.random() < 0.5 or np.all(f1):
                arm = int(rng.choice(np.flatnonzero(f1)))
            else:
                rem = np.flatnonzero(~f1)
                f2_mask = _pareto_mask_2d(draw[rem])
                arm = int(rng.choice(rem[f2_mask]))

        reward = truth_mu[arm] + rng.normal(0.0, sigma, size=d)
        n_old = counts[arm]
        means[arm] = (means[arm] * n_old + reward) / (n_old + 1)
        counts[arm] += 1

        if t in grid_set:
            record(t)

    return rows


def run_r2_suite(output_dir: Path, n_jobs: int, smoke: bool = False) -> Path:
    cfg = load_config()
    seed = int(cfg["heldout_seeds"]["trajectory_correlation_seed"])
    eval_points = int(cfg["eval_points"])
    mc_draws = 128 if smoke else int(cfg["mc_draws"])
    families = cfg["random_trajectory_families"]

    all_rows: list[dict] = []
    root_ss = np.random.SeedSequence(seed)
    fam_seeds = root_ss.spawn(len(families))

    for fam, fam_ss in zip(families, fam_seeds):
        fam_use = dict(fam)
        if smoke:
            fam_use["num_runs"] = 5
            fam_use["budget"] = min(int(fam["budget"]), 2000)
        n_runs = int(fam_use["num_runs"])
        rep_seeds = fam_ss.spawn(n_runs)
        print(f"[R2] Running {fam_use['name']} (K={fam_use['k']}, d={fam_use['d']}, sigma={fam_use['sigma']}, B={fam_use['budget']}, n={n_runs})...", flush=True)
        if n_jobs == 1:
            batches = [_run_r2_repetition(fam_use, r, eval_points, mc_draws, rep_seeds[r]) for r in range(n_runs)]
        else:
            batches = Parallel(n_jobs=n_jobs)(
                delayed(_run_r2_repetition)(fam_use, r, eval_points, mc_draws, rep_seeds[r]) for r in range(n_runs)
            )
        for batch in batches:
            all_rows.extend(batch)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "random_trajectories.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAJ_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved R2 trajectory dataset ({len(all_rows)} rows) -> {out_csv}", flush=True)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["all", "r1", "r2"], default="all")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.mode in ("all", "r1"):
        run_r1_suite(args.output_dir, args.n_jobs, smoke=args.smoke)
    if args.mode in ("all", "r2"):
        run_r2_suite(args.output_dir, args.n_jobs, smoke=args.smoke)


if __name__ == "__main__":
    main()
