"""Instance-weighted summaries and paired contrasts for pilot outputs."""

from __future__ import annotations

from pathlib import Path
from statistics import NormalDist

import pandas as pd

from .pilot import environment_key, load_pilot_config, pilot_candidates


METRICS = ("bernoulli", "jaccard", "misidentification")


def load_pilot_rows(results_dir: Path) -> pd.DataFrame:
    config = load_pilot_config()
    rows = []
    for candidate in pilot_candidates(config):
        key = environment_key(candidate)
        data = pd.read_csv(results_dir / f"{key}.csv")
        data["instance_id"] = candidate.instance_id
        data["condition"] = candidate.condition
        data["normalized_budget"] = data["arm_pull"] / candidate.budget
        rows.append(data)
    return pd.concat(rows, ignore_index=True)


def _interval(values: pd.Series) -> tuple[float, float]:
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean
    se = float(values.std(ddof=1) / len(values) ** 0.5)
    z = NormalDist().inv_cdf(0.975)
    return mean - z * se, mean + z * se


def summarize_primary(rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Equal-instance primary summary; one pilot trajectory per instance."""
    output = []
    for keys, group in rows.groupby(["condition", "algorithm", "normalized_budget"], sort=True):
        lower, upper = _interval(group[metric])
        output.append(
            {
                "condition": keys[0],
                "algorithm": keys[1],
                "normalized_budget": keys[2],
                "metric": metric,
                "mean": group[metric].mean(),
                "lower": lower,
                "upper": upper,
                "n_instances": group["instance_id"].nunique(),
            }
        )
    return pd.DataFrame(output)


def paired_contrasts(rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Heterogeneous-minus-control metric contrast paired by frozen instance."""
    columns = ["instance_id", "algorithm", "normalized_budget", "condition", metric]
    pivot = rows[columns].pivot(
        index=["instance_id", "algorithm", "normalized_budget"],
        columns="condition",
        values=metric,
    )
    if set(pivot.columns) != {"heterogeneous", "homoscedastic_rms"}:
        raise ValueError("paired analysis requires exactly both noise conditions for every row")
    paired = (pivot["heterogeneous"] - pivot["homoscedastic_rms"]).rename("heterogeneous_minus_control")
    output = []
    for keys, values in paired.reset_index().groupby(["algorithm", "normalized_budget"], sort=True):
        lower, upper = _interval(values["heterogeneous_minus_control"])
        output.append(
            {
                "algorithm": keys[0],
                "normalized_budget": keys[1],
                "metric": metric,
                "heterogeneous_minus_control": values["heterogeneous_minus_control"].mean(),
                "lower": lower,
                "upper": upper,
                "n_instances": values["instance_id"].nunique(),
            }
        )
    return pd.DataFrame(output)
