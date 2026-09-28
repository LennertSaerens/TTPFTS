#!/usr/bin/env python3
"""Freeze and report the candidate heteroscedastic MOMAB ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import config_hash, load_candidate_config
from .instances import ROOT, freeze_candidates


REPORTS_DIR = ROOT / "reports"


def candidate_report(candidates) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        if candidate.condition != "heterogeneous":
            continue
        rows.append(
            {
                "instance_id": candidate.instance_id,
                "num_arms": candidate.means.shape[0],
                "num_objectives": candidate.means.shape[1],
                "pareto_size": len(candidate.pareto_indices),
                "minimum_gap": candidate.hardness.gaps_direct.min(),
                "hardness": candidate.hardness.hardness,
                "log10_hardness": np.log10(candidate.hardness.hardness),
                "budget": candidate.budget,
                "heterogeneous_rms_std": candidate.rms_std,
                "heterogeneous_std": float(candidate.stds.std()),
                "heterogeneous_min_std": float(candidate.stds.min()),
                "heterogeneous_max_std": float(candidate.stds.max()),
                "sigma_0_1_count": int(np.count_nonzero(candidate.stds == 0.1)),
                "sigma_0_2_count": int(np.count_nonzero(candidate.stds == 0.2)),
                "sigma_0_3_count": int(np.count_nonzero(candidate.stds == 0.3)),
                "sigma_0_4_count": int(np.count_nonzero(candidate.stds == 0.4)),
                "sigma_0_5_count": int(np.count_nonzero(candidate.stds == 0.5)),
            }
        )
    return pd.DataFrame(rows).sort_values("instance_id").reset_index(drop=True)


def write_report(candidates, cfg: dict) -> pd.DataFrame:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = candidate_report(candidates)
    report.to_csv(REPORTS_DIR / "candidate_hardness_report.csv", index=False)
    quantiles = {
        str(q): float(report["hardness"].quantile(q))
        for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    }
    summary = {
        "candidate_config_hash": config_hash(cfg),
        "candidate_count": int(len(report)),
        "hardness_quantiles": quantiles,
        "counts_above_hardness": {
            str(threshold): int((report["hardness"] > threshold).sum())
            for threshold in (1e6, 1e7, 1e8, 1e9)
        },
        "hardness_threshold_decision": "pending_explicit_user_approval",
        "note": (
            "No candidate has been silently rejected, clipped, or run. "
            "This report is the evidence for the required hardness gate."
        ),
    }
    (REPORTS_DIR / "candidate_hardness_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true", help="Generate/report without replacing frozen artifacts.")
    args = parser.parse_args()
    cfg = load_candidate_config()
    candidates, _manifest = freeze_candidates(cfg, write=not args.no_write)
    report = write_report(candidates, cfg)
    print(f"Candidate config hash: {config_hash(cfg)}")
    print(f"Candidates: {len(report)}")
    print(report["hardness"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
    print(f"Wrote {REPORTS_DIR / 'candidate_hardness_report.csv'}")


if __name__ == "__main__":
    main()
