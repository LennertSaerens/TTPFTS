#!/usr/bin/env python3
"""Re-score saved TTPFTS_UNKNOWN_VAR TPosterior shards with V1 and V2 (plus sampler-scale audit)."""

from __future__ import annotations

import argparse
import csv
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


def rescore_uv(uv_results_dir: Path, output_dir: Path) -> Path:
    verify_uv_posterior_contract()
    uv_run_dir = uv_results_dir / "run"
    output_dir.mkdir(parents=True, exist_ok=True)
    environments = [item["name"] for item in load_final_config()["environments"]]

    out_rows = []
    for env_name in environments:
        env_csv = uv_results_dir / f"{env_name}.csv"
        df = pd.read_csv(env_csv)
        reps = sorted(df["repetition"].unique())
        for rep in reps:
            rep_df = df[df["repetition"] == rep].sort_values("arm_pull")
            _cpath, spath = shard_paths(uv_run_dir, env_name, int(rep))
            with np.load(spath, allow_pickle=False) as state:
                n_arr, m_arr, m2_arr = state["n"], state["means"], state["m2"]
            for idx, row in enumerate(rep_df.itertuples(index=False)):
                if int(row.arm_pull) == 0:
                    v1_val = v2_val = ""
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
                        v1_val = v2_val = ""
                        avail = 0
                out_rows.append({
                    "environment": env_name,
                    "repetition": int(rep),
                    "arm_pull": int(row.arm_pull),
                    "normalized_budget": float(row.normalized_budget),
                    "bernoulli": int(row.bernoulli),
                    "jaccard": float(row.jaccard),
                    "misidentification": float(row.misidentification),
                    "eq10_available": int(row.uq_available),
                    "bhattacharyya": 0.0 if pd.isna(row.bhattacharyya) and int(row.arm_pull) > 0 else row.bhattacharyya,
                    "bhattacharyya_raw": "" if pd.isna(row.bhattacharyya) else float(row.bhattacharyya),
                    "uv_finite_avail": avail,
                    "uq_v1": v1_val,
                    "uq_v2": v2_val,
                })
        print(f"[UV rescore] Completed {env_name}", flush=True)

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
    args = parser.parse_args()
    rescore_uv(args.uv_results_dir, args.output_dir)


if __name__ == "__main__":
    main()
