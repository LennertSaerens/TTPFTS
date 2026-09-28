"""Valid-count-aware summaries for hardness-budget UQ replay results."""

from __future__ import annotations

from statistics import NormalDist
from pathlib import Path

import pandas as pd

from experiments.final_config_loader import load_final_config

from .config import config_hash, load_config


MEASURES = ("bhattacharyya", "symmetric_kl", "entropy")


def load_results(results_dir: Path, environments: list[str] | None = None) -> pd.DataFrame:
    """Load UQ results, retaining unavailable values as missing rather than zero."""
    config = load_config()
    final_config = load_final_config()
    if environments is None:
        environments = [
            item["name"] for item in final_config["environments"]
            if (results_dir / f"{item['name']}.csv").is_file()
        ]
    if not environments:
        raise ValueError(f"no UQ environment CSVs found in {results_dir}")
    frames = []
    for environment in environments:
        path = results_dir / f"{environment}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, keep_default_na=True)
        if set(frame["uq_config_hash"]) != {config_hash(config)}:
            raise ValueError(f"{environment}: config hash mismatch")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def evolution_summary(rows: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Summarize one UQ measure at each normalized checkpoint across trajectories."""
    if measure not in MEASURES:
        raise ValueError(f"unknown UQ measure {measure}")
    output = []
    z = NormalDist().inv_cdf(0.975)
    for (environment, pull, normalized), group in rows.groupby(
        ["environment", "arm_pull", "normalized_budget"], sort=True
    ):
        valid = group[group["uq_available"].astype(int) == 1][measure].dropna()
        n_total = len(group)
        n_valid = len(valid)
        reasons = group.loc[group["uq_available"].astype(int) == 0, "uq_reason"].value_counts().to_dict()
        if n_valid:
            mean = float(valid.mean())
            std = float(valid.std(ddof=1)) if n_valid > 1 else 0.0
            se = std / n_valid ** 0.5
            lower, upper = mean - z * se, mean + z * se
        else:
            mean = std = se = lower = upper = float("nan")
        output.append(
            {
                "environment": environment,
                "arm_pull": int(pull),
                "normalized_budget": float(normalized),
                "measure": measure,
                "mean": mean,
                "std": std,
                "se": se,
                "lower": lower,
                "upper": upper,
                "n_valid": n_valid,
                "n_total": n_total,
                "n_unavailable": n_total - n_valid,
                "unavailable_reasons": ";".join(f"{key}:{value}" for key, value in sorted(reasons.items())),
            }
        )
    return pd.DataFrame(output)


def trajectory_correlations(rows: pd.DataFrame, measure: str) -> pd.DataFrame:
    """One within-trajectory Pearson UQ/Jaccard correlation per environment/run."""
    if measure not in MEASURES:
        raise ValueError(f"unknown UQ measure {measure}")
    output = []
    for (environment, repetition), group in rows.groupby(["environment", "repetition"], sort=True):
        valid = group[
            (group["arm_pull"] > 0)
            & (group["uq_available"].astype(int) == 1)
            & group[measure].notna()
        ].sort_values("normalized_budget")
        n_total_positive = int((group["arm_pull"] > 0).sum())
        n_valid = len(valid)
        reason = "available"
        correlation = float("nan")
        if n_valid < 3:
            reason = "fewer_than_three_valid_checkpoints"
        elif valid[measure].nunique(dropna=True) < 2:
            reason = "constant_uncertainty"
        elif valid["jaccard"].nunique(dropna=True) < 2:
            reason = "constant_jaccard"
        else:
            correlation = float(valid[measure].corr(valid["jaccard"], method="pearson"))
            if pd.isna(correlation):
                reason = "undefined_pearson_correlation"
            else:
                reason = "available"
        output.append(
            {
                "environment": environment,
                "repetition": int(repetition),
                "measure": measure,
                "pearson_r": correlation,
                "status": reason,
                "n_valid_checkpoints": n_valid,
                "n_positive_checkpoints": n_total_positive,
                "first_valid_normalized_budget": (
                    float(valid["normalized_budget"].min()) if n_valid else float("nan")
                ),
                "last_valid_normalized_budget": (
                    float(valid["normalized_budget"].max()) if n_valid else float("nan")
                ),
            }
        )
    return pd.DataFrame(output)


def correlation_summary(correlations: pd.DataFrame) -> pd.DataFrame:
    """Descriptive boxplot statistics and exclusion counts per environment."""
    output = []
    for environment, group in correlations.groupby("environment", sort=True):
        valid = group["pearson_r"].dropna()
        output.append(
            {
                "environment": environment,
                "n_total": len(group),
                "n_valid": len(valid),
                "n_excluded": len(group) - len(valid),
                "mean": valid.mean(),
                "std": valid.std(ddof=1) if len(valid) > 1 else 0.0,
                "min": valid.min(),
                "q1": valid.quantile(0.25),
                "median": valid.median(),
                "q3": valid.quantile(0.75),
                "max": valid.max(),
                "exclusion_reasons": ";".join(
                    f"{key}:{value}" for key, value in sorted(group.loc[group["pearson_r"].isna(), "status"].value_counts().to_dict().items())
                ),
            }
        )
    return pd.DataFrame(output)


def write_analysis(results_dir: Path, output_dir: Path, environments: list[str] | None = None) -> None:
    """Write all raw/summarized analysis tables for the six planned figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_results(results_dir, environments)
    rows.to_csv(output_dir / "trajectory_checkpoints.csv", index=False)
    for measure in MEASURES:
        evolution_summary(rows, measure).to_csv(output_dir / f"{measure}_evolution.csv", index=False)
        correlations = trajectory_correlations(rows, measure)
        correlations.to_csv(output_dir / f"{measure}_trajectory_correlations.csv", index=False)
        correlation_summary(correlations).to_csv(output_dir / f"{measure}_correlation_summary.csv", index=False)
