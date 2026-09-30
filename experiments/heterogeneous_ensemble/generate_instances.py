#!/usr/bin/env python3
"""Generate, freeze, and audit the 1,000-instance heterogeneous cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import config_hash, load_config
from .instances import ROOT, freeze_instances


REPORTS_DIR = ROOT / "reports"


def cohort_report(instances, ledger: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Return accepted-instance and proposal-rejection summaries."""
    accepted = pd.DataFrame(
        [
            {
                "instance_id": instance.instance_id,
                "proposal_id": instance.proposal_id,
                "num_arms": instance.num_arms,
                "num_objectives": instance.num_objectives,
                "pareto_size": len(instance.pareto_indices),
                "pareto_fraction": len(instance.pareto_indices) / instance.num_arms,
                "minimum_gap": float(np.min(instance.hardness.gaps_direct)),
                "hardness": instance.hardness.hardness,
                "budget": instance.budget,
                "noise_diversity": instance.noise_diversity,
                "sigma_0_1_count": int(np.count_nonzero(instance.stds == 0.1)),
                "sigma_0_2_count": int(np.count_nonzero(instance.stds == 0.2)),
                "sigma_0_3_count": int(np.count_nonzero(instance.stds == 0.3)),
                "sigma_0_4_count": int(np.count_nonzero(instance.stds == 0.4)),
                "sigma_0_5_count": int(np.count_nonzero(instance.stds == 0.5)),
                "instance_hash": instance.instance_hash,
            }
            for instance in instances
        ]
    )
    ledger_frame = pd.DataFrame(ledger)
    proposal_summary = {
        "proposal_count": len(ledger_frame),
        "accepted_count": len(accepted),
        "acceptance_rate": len(accepted) / len(ledger_frame),
        "rejection_reason_counts": {
            str(key): int(value)
            for key, value in ledger_frame.loc[ledger_frame["reason"] != "accepted", "reason"]
            .value_counts()
            .sort_index()
            .items()
        },
        "accepted_hardness_quantiles": {
            str(quantile): float(accepted["hardness"].quantile(quantile))
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
        },
        "accepted_budget_total": int(accepted["budget"].sum()),
        "note": (
            "Accepted instances are the first ordered proposals satisfying the EgeExp3 "
            "mean-gap hardness ceiling; no budgets are clipped and no accepted instance "
            "is removed post hoc."
        ),
    }
    return accepted, proposal_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly regenerate the frozen cohort.")
    args = parser.parse_args()
    config = load_config()
    target = ROOT / "instances" / "manifest.json"
    if target.exists() and not args.overwrite:
        raise SystemExit(f"{target} already exists; use --overwrite only to regenerate before results exist.")
    instances, manifest, ledger = freeze_instances(config)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    accepted, summary = cohort_report(instances, ledger)
    accepted.to_csv(REPORTS_DIR / "accepted_cohort_report.csv", index=False)
    pd.DataFrame(ledger).to_csv(REPORTS_DIR / "proposal_audit.csv", index=False)
    summary.update(
        {
            "ensemble_config_hash": config_hash(config),
            "manifest_sha256": manifest["artifacts"],
            "hardness_ceiling": config["hardness"]["ceiling_hardness"],
        }
    )
    (REPORTS_DIR / "cohort_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Frozen {len(instances)} accepted instances from {len(ledger)} proposals; "
        f"total budget={summary['accepted_budget_total']}"
    )


if __name__ == "__main__":
    main()
