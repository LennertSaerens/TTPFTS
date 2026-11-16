"""Analysis summaries for KV 8-env replay, decision calibration, stopping rules, and random instances."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from experiments.final_config_loader import load_final_config

from .config import load_config


UQ_MEASURES = ("bhattacharyya", "bhattacharyya_raw", "uq_v1", "uq_v2", "uq_mc")
GT_METRICS = ("jaccard", "bernoulli", "misidentification")


def _wilson_ci(wrong: int, total: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    p = wrong / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return float(p), float(max(0.0, center - half)), float(min(1.0, center + half))


def load_kv_results(results_dir: Path) -> pd.DataFrame:
    environments = [item["name"] for item in load_final_config()["environments"]]
    frames = []
    for env in environments:
        path = results_dir / f"{env}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path, keep_default_na=True))
    return pd.concat(frames, ignore_index=True)


def evolution_summary(rows: pd.DataFrame, group_col: str, measure: str) -> pd.DataFrame:
    z = NormalDist().inv_cdf(0.975)
    out = []
    for (group_val, pull, norm_b), group in rows.groupby([group_col, "arm_pull", "normalized_budget"], sort=True):
        if int(pull) == 0:
            continue
        valid = pd.to_numeric(group.loc[group["uq_available"].astype(int) == 1, measure], errors="coerce").dropna()
        n_valid = len(valid)
        n_total = len(group)
        if n_valid:
            mean = float(valid.mean())
            std = float(valid.std(ddof=1)) if n_valid > 1 else 0.0
            se = std / (n_valid ** 0.5)
            lower, upper = mean - z * se, mean + z * se
        else:
            mean = std = se = lower = upper = float("nan")
        out.append({
            group_col: group_val,
            "arm_pull": int(pull),
            "normalized_budget": float(norm_b),
            "measure": measure,
            "mean": mean,
            "std": std,
            "lower": lower,
            "upper": upper,
            "n_valid": n_valid,
            "n_total": n_total,
        })
    return pd.DataFrame(out)


def trajectory_correlations(rows: pd.DataFrame, group_col: str, measures=UQ_MEASURES, gt_metrics=GT_METRICS) -> pd.DataFrame:
    out = []
    for (group_val, rep), group in rows.groupby([group_col, "repetition"], sort=True):
        pos = group[(group["arm_pull"] > 0) & (group["uq_available"].astype(int) == 1)].sort_values("normalized_budget")
        for measure in measures:
            sub = pos[pd.to_numeric(pos[measure], errors="coerce").notna()].copy()
            sub[measure] = pd.to_numeric(sub[measure], errors="coerce")
            for gt in gt_metrics:
                sub_gt = sub[pd.to_numeric(sub[gt], errors="coerce").notna()]
                n_valid = len(sub_gt)
                r = float("nan")
                reason = "available"
                if n_valid < 3:
                    reason = "fewer_than_three_valid"
                elif sub_gt[measure].nunique() < 2:
                    reason = "constant_uq"
                elif sub_gt[gt].nunique() < 2:
                    reason = f"constant_{gt}"
                else:
                    r = float(sub_gt[measure].corr(sub_gt[gt], method="pearson"))
                    if np.isnan(r):
                        reason = "undefined_pearson"
                out.append({
                    group_col: group_val,
                    "repetition": int(rep),
                    "measure": measure,
                    "gt_metric": gt,
                    "pearson_r": r,
                    "status": reason,
                    "n_valid": n_valid,
                })
    return pd.DataFrame(out)


def correlation_summary_table(corr_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    out = []
    for (group_val, measure, gt), grp in corr_df.groupby([group_col, "measure", "gt_metric"], sort=True):
        valid = grp["pearson_r"].dropna()
        reasons = grp.loc[grp["pearson_r"].isna(), "status"].value_counts().to_dict()
        out.append({
            group_col: group_val,
            "measure": measure,
            "gt_metric": gt,
            "n_total": len(grp),
            "n_valid": len(valid),
            "n_excluded": len(grp) - len(valid),
            "mean": float(valid.mean()) if len(valid) else float("nan"),
            "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
            "q1": float(valid.quantile(0.25)) if len(valid) else float("nan"),
            "median": float(valid.median()) if len(valid) else float("nan"),
            "q3": float(valid.quantile(0.75)) if len(valid) else float("nan"),
            "min": float(valid.min()) if len(valid) else float("nan"),
            "max": float(valid.max()) if len(valid) else float("nan"),
            "exclusion_reasons": ";".join(f"{k}:{v}" for k, v in sorted(reasons.items())),
        })
    return pd.DataFrame(out)


def kv_decision_quality_analysis(rows: pd.DataFrame) -> dict[str, object]:
    """Compute matched-budget score bins, coverage accuracy, MC calibration, first-crossing, and runtime on 8 envs."""
    pos = rows[(rows["arm_pull"] > 0) & (rows["uq_available"].astype(int) == 1)].copy()
    for col in ("bhattacharyya", "uq_v1", "uq_v2", "uq_mc", "jaccard", "bernoulli", "normalized_budget",
                "time_eq10_ms", "time_v1_ms", "time_v2_ms", "time_mc_ms"):
        pos[col] = pd.to_numeric(pos[col], errors="coerce")
    pos["exact_error"] = (1 - pos["bernoulli"]).astype(int)
    pos["jaccard_loss"] = 1.0 - pos["jaccard"]

    # 1. Fixed normalized budgets: 0.25, 0.50, 1.00 across all 800 runs
    budget_tables = {}
    bins = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 1.01)]
    for b_norm in (0.25, 0.50, 1.0):
        sub = pos[np.isclose(pos["normalized_budget"], b_norm, atol=1e-6)]
        if sub.empty:
            continue
        b_entry = {"n": len(sub), "overall_error_rate": float(sub["exact_error"].mean()), "metrics": {}}
        for m in ("bhattacharyya", "uq_v1", "uq_v2", "uq_mc"):
            vals = sub[m].to_numpy()
            errs = sub["exact_error"].to_numpy()
            jloss = sub["jaccard_loss"].to_numpy()
            m_bins = []
            for b_lo, b_hi in bins:
                mask = (vals >= 0) & (vals <= b_hi) if b_lo == 0 else (vals > b_lo) & (vals <= b_hi)
                bn = int(mask.sum())
                bw = int(errs[mask].sum())
                er, lo, hi = _wilson_ci(bw, bn)
                m_bins.append({
                    "range": f"[{b_lo:.2f},{min(b_hi, 1.0):.2f}]" if b_lo == 0 else f"({b_lo:.2f},{min(b_hi, 1.0):.2f}]",
                    "n": bn, "wrong": bw, "error_rate": er if bn else None,
                    "ci_low": lo if bn else None, "ci_high": hi if bn else None,
                    "mean_jaccard_loss": float(jloss[mask].mean()) if bn else None,
                })
            # Matched coverage
            order = np.argsort(vals, kind="mergesort")
            cov_dict = {}
            for cov in (0.10, 0.25, 0.50):
                top_k = max(1, int(round(cov * len(sub))))
                idx = order[:top_k]
                cw = int(errs[idx].sum())
                cer, clo, chi = _wilson_ci(cw, top_k)
                cov_dict[f"{cov:.2f}"] = {
                    "n": top_k, "wrong": cw, "accuracy": 1.0 - cer, "error_rate": cer,
                    "ci_low": clo, "ci_high": chi, "cutoff": float(vals[idx[-1]]),
                }
            b_entry["metrics"][m] = {"bins": m_bins, "coverage": cov_dict}
        budget_tables[f"{b_norm:.2f}"] = b_entry

    # 2. Stopping first-crossing along the 101-point normalized budget grid across 8 envs
    thresholds = {
        "bhattacharyya": [0.05, 0.01, 0.0001],
        "uq_v1": [0.10, 0.05, 0.01],
        "uq_v2": [0.10, 0.05, 0.01],
        "uq_mc": [0.10, 0.05, 0.01],
    }
    stopping_rows = []
    for env_name, env_df in pos.groupby("environment", sort=True):
        reps = sorted(env_df["repetition"].unique())
        max_budget = int(env_df["max_budget"].iloc[0])
        for m_name, thr_list in thresholds.items():
            for thr in thr_list:
                crossed_list, err_list, jac_list, pull_list, norm_list = [], [], [], [], []
                for rep in reps:
                    rdf = env_df[env_df["repetition"] == rep].sort_values("arm_pull")
                    if m_name == "bhattacharyya":
                        cond = (rdf["eq10_nonempty_second"].astype(int) == 1) & (rdf[m_name] <= thr)
                    else:
                        cond = rdf[m_name] <= thr
                    hits = rdf[cond]
                    if len(hits) > 0:
                        first_hit = hits.iloc[0]
                        crossed_list.append(1)
                        err_list.append(int(first_hit["exact_error"]))
                        jac_list.append(float(first_hit["jaccard"]))
                        pull_list.append(int(first_hit["arm_pull"]))
                        norm_list.append(float(first_hit["normalized_budget"]))
                    else:
                        last_row = rdf.iloc[-1]
                        crossed_list.append(0)
                        err_list.append(int(last_row["exact_error"]))
                        jac_list.append(float(last_row["jaccard"]))
                        pull_list.append(int(last_row["arm_pull"]))
                        norm_list.append(float(last_row["normalized_budget"]))
                c_arr = np.asarray(crossed_list, dtype=bool)
                e_arr = np.asarray(err_list, dtype=int)
                j_arr = np.asarray(jac_list, dtype=float)
                n_arr = np.asarray(norm_list, dtype=float)
                p_arr = np.asarray(pull_list, dtype=int)
                nc = int(c_arr.sum())
                wc = int(e_arr[c_arr].sum())
                er_c, lo_c, hi_c = _wilson_ci(wc, nc)
                stopping_rows.append({
                    "environment": env_name, "max_budget": max_budget,
                    "measure": m_name, "threshold": thr,
                    "n_total": len(reps), "n_crossed": nc, "frac_crossed": nc / len(reps),
                    "error_at_crossing": er_c if nc else None,
                    "ci_low": lo_c if nc else None, "ci_high": hi_c if nc else None,
                    "mean_jaccard_at_crossing": float(j_arr[c_arr].mean()) if nc else None,
                    "mean_norm_budget_crossed": float(n_arr[c_arr].mean()) if nc else None,
                    "mean_norm_budget_all": float(n_arr.mean()),
                    "mean_pulls_all": float(p_arr.mean()),
                    "overall_error": float(e_arr.mean()),
                })

    # 3. Runtime & posterior disagreement MAE vs MC
    runtime_rows = []
    for env_name, env_df in pos.groupby("environment", sort=True):
        runtime_rows.append({
            "environment": env_name,
            "num_arms": int(len(json.loads(env_df["true_pareto"].iloc[0]))) if False else None,
            "time_eq10_ms": float(env_df["time_eq10_ms"].mean()),
            "time_v1_ms": float(env_df["time_v1_ms"].mean()),
            "time_v2_ms": float(env_df["time_v2_ms"].mean()),
            "time_mc512_ms": float(env_df["time_mc_ms"].mean()),
            "mae_eq10_vs_mc": float(np.abs(env_df["bhattacharyya"] - env_df["uq_mc"]).mean()),
            "mae_v1_vs_mc": float(np.abs(env_df["uq_v1"] - env_df["uq_mc"]).mean()),
            "mae_v2_vs_mc": float(np.abs(env_df["uq_v2"] - env_df["uq_mc"]).mean()),
            "corr_eq10_vs_mc": float(env_df["bhattacharyya"].corr(env_df["uq_mc"])),
            "corr_v1_vs_mc": float(env_df["uq_v1"].corr(env_df["uq_mc"])),
            "corr_v2_vs_mc": float(env_df["uq_v2"].corr(env_df["uq_mc"])),
            "empty_second_frac": float(1.0 - env_df["eq10_nonempty_second"].mean()),
        })

    return {
        "budget_tables": budget_tables,
        "stopping_8envs": stopping_rows,
        "runtime_and_mc_agreement": runtime_rows,
    }


def write_all_analyses(results_dir: Path, random_csv: Path | None, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_kv_results(results_dir)
    for m in ("bhattacharyya", "uq_v1", "uq_v2", "uq_mc"):
        evolution_summary(rows, "environment", m).to_csv(output_dir / f"kv_{m}_evolution.csv", index=False)

    corr_8env = trajectory_correlations(rows, "environment")
    corr_8env.to_csv(output_dir / "kv_8env_trajectory_correlations.csv", index=False)
    correlation_summary_table(corr_8env, "environment").to_csv(output_dir / "kv_8env_correlation_summary.csv", index=False)

    decision_info = kv_decision_quality_analysis(rows)
    (output_dir / "kv_decision_quality.json").write_text(json.dumps(decision_info, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(decision_info["stopping_8envs"]).to_csv(output_dir / "kv_8env_stopping.csv", index=False)
    pd.DataFrame(decision_info["runtime_and_mc_agreement"]).to_csv(output_dir / "kv_runtime_mc_agreement.csv", index=False)

    if random_csv is not None and random_csv.is_file():
        r_rows = pd.read_csv(random_csv, keep_default_na=True)
        corr_rand = trajectory_correlations(r_rows, "family")
        corr_rand.to_csv(output_dir / "random_trajectory_correlations.csv", index=False)
        correlation_summary_table(corr_rand, "family").to_csv(output_dir / "random_correlation_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--random-trajectories", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_all_analyses(args.results_dir, args.random_trajectories, args.output_dir)


if __name__ == "__main__":
    main()
