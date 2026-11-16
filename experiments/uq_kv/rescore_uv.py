#!/usr/bin/env python3
"""Re-score saved TTPFTS_UNKNOWN_VAR TPosterior shards with V1 and V2 (plus sampler-scale audit)."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from bandits.Posteriors import TPosterior
from experiments.final_config_loader import load_final_config
from experiments.uq.metrics import (
    posterior_stds_from_tposterior_state,
    uq_directional_bc_from_moments,
    uq_dominance_risk_from_moments,
)
from experiments.uq.replay import shard_paths


def verify_uv_posterior_contract() -> dict[str, float]:
    """Verify TPosterior sampler scale and logged std against batch closed forms."""
    data = np.array([0.0, 2.0, 0.0, 2.0, 1.5, -0.5], dtype=np.float64)
    post = TPosterior(1, 1, alpha=0.0)
    for val in data:
        post.update(0, np.array([val]))
    n_val = float(len(data))
    mean_val = float(np.mean(data))
    s_val = float(np.sum((data - mean_val) ** 2))
    df = n_val - 1.0
    expected_scale = float(np.sqrt(s_val / (n_val * df)))
    expected_std = float(np.sqrt(s_val / (n_val * (df - 2.0))))
    got_scale = float(post.get_scale()[0, 0])
    got_std = float(post.get_stds()[0, 0])
    reconstructed_std = float(posterior_stds_from_tposterior_state(post.n, post.M2, alpha=0.0)[0, 0])
    if not (np.isclose(got_scale, expected_scale) and np.isclose(got_std, expected_std) and np.isclose(reconstructed_std, expected_std)):
        raise AssertionError("TPosterior scale/std contract mismatch")
    return {
        "n": n_val,
        "df": df,
        "batch_scale": expected_scale,
        "sampler_scale": got_scale,
        "logged_std": got_std,
        "reconstructed_std": reconstructed_std,
        "ratio_std_over_scale": expected_std / expected_scale,
    }


def _rescore_one_rep(
    task: tuple[str, int, Path, list[tuple[int, float, int, float, float, int, float]]]
) -> list[dict]:
    env_name, rep, uv_run_dir, rep_rows = task
    _cpath, spath = shard_paths(uv_run_dir, env_name, rep)
    with np.load(spath, allow_pickle=False) as state:
        n_arr, m_arr, m2_arr = state["n"], state["means"], state["m2"]

    rows: list[dict] = []
    for idx, (arm_pull, norm_b, bern, jacc, misid, uq_avail, bhatt) in enumerate(rep_rows):
        if arm_pull == 0:
            v1_val: float | str = ""
            v2_val: float | str = ""
            avail = 0
        else:
            try:
                stds = posterior_stds_from_tposterior_state(n_arr[idx], m2_arr[idx], alpha=0.0)
            except ValueError:
                stds = np.full_like(m_arr[idx], np.inf)
            if np.all(np.isfinite(stds)) and np.all(stds > 0):
                v1_val = uq_directional_bc_from_moments(m_arr[idx], stds)
                v2_val = uq_dominance_risk_from_moments(m_arr[idx], stds)
                avail = 1
            else:
                v1_val = ""
                v2_val = ""
                avail = 0
        rows.append({
            "environment": env_name,
            "repetition": rep,
            "arm_pull": arm_pull,
            "normalized_budget": norm_b,
            "bernoulli": bern,
            "jaccard": jacc,
            "misidentification": misid,
            "eq10_available": uq_avail,
            "bhattacharyya": 0.0 if np.isnan(bhatt) and arm_pull > 0 else ("" if np.isnan(bhatt) else bhatt),
            "bhattacharyya_raw": "" if np.isnan(bhatt) else bhatt,
            "uv_finite_avail": avail,
            "uq_v1": v1_val,
            "uq_v2": v2_val,
        })
    return rows


def rescore_uv(uv_results_dir: Path, output_dir: Path, n_jobs: int = 1) -> Path:
    contract = verify_uv_posterior_contract()
    uv_run_dir = uv_results_dir / "run"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "uv_posterior_contract.json").open("w") as handle:
        json.dump(contract, handle, indent=2)

    environments = [item["name"] for item in load_final_config()["environments"]]
    tasks: list[tuple[str, int, Path, list[tuple[int, float, int, float, float, int, float]]]] = []
    for env_name in environments:
        env_csv = uv_results_dir / f"{env_name}.csv"
        df = pd.read_csv(env_csv)
        reps = sorted(df["repetition"].unique())
        for rep in reps:
            rep_df = df[df["repetition"] == rep].sort_values("arm_pull")
            rep_rows = [
                (
                    int(r.arm_pull),
                    float(r.normalized_budget),
                    int(r.bernoulli),
                    float(r.jaccard),
                    float(r.misidentification),
                    int(r.uq_available),
                    float("nan") if pd.isna(r.bhattacharyya) else float(r.bhattacharyya),
                )
                for r in rep_df.itertuples(index=False)
            ]
            tasks.append((env_name, int(rep), uv_run_dir, rep_rows))

    out_rows: list[dict] = []
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            for rep_out in pool.map(_rescore_one_rep, tasks):
                out_rows.extend(rep_out)
    else:
        for task in tasks:
            out_rows.extend(_rescore_one_rep(task))

    out_path = output_dir / "uv_rescored_checkpoints.csv"
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Saved UV rescored table ({len(out_rows)} rows) -> {out_path}", flush=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 1)))
    args = parser.parse_args()
    rescore_uv(args.uv_results_dir, args.output_dir, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
