#!/usr/bin/env python3
"""Early-curve artifact investigation (report only; figures are not changed).

Quantifies five mechanisms that shape the first evaluation points of the
publication curves and writes CSV tables, a diagnostic early-budget figure,
and a Markdown report to ``experiments/information_matched/reports``:

M1  zero-information recommendation at t = 0 (all-zero means: every arm is
    non-dominated, so every anytime algorithm recommends all K arms);
M2  native EGE classification of arms that have never been sampled at small
    budgets (EGE-SR at T <= K draws nothing; EGE-SH floors round-1 pulls to
    zero whenever T < K * ceil(log2 K));
M3  warm-up / round-robin phases that still cover the first grid point;
M4  zero-initialized means of unsampled arms in empirical recommendations
    (counterfactual: exclude unsampled arms);
M5  log-scale misidentification axes silently omit exact zeros.

The pull-order sensitivity of the empirical-mean D-EGE (round-robin versus
native block order within rounds) is reported as well.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bandits.APE import _pareto_optimal_arms_mask
from experiments.ege_doubling.schedule import doubling_schedule
from momab_main import EGE_ALGORITHMS, ENVIRONMENTS, _construct_environment
from runner.seeding import seed_rngs

from .build import plot_input_path
from .config import (
    EMPIRICAL_ALGORITHMS,
    REPO_ROOT,
    assert_safe_output,
    default_output_root,
    environment_specs,
    load_config,
    preregistered_dir,
)
from .run_empirical import SampleRecorder, results_path, round_robin, stage_plan

REPORT_DIR = REPO_ROOT / "experiments" / "information_matched" / "reports"
ANYTIME = ["Uniform", "APE", "APE_UV", "TTPFTS_KNOWN_VAR", "TTPFTS_UNKNOWN_VAR", "D_EGE_SH", "D_EGE_SR"]
LABELS = {"Uniform": "Uniform", "APE": "APE", "APE_UV": "APE-UV", "TTPFTS_KNOWN_VAR": "TTPFTS-KV",
          "TTPFTS_UNKNOWN_VAR": "TTPFTS-UV", "D_EGE_SH": "D-EGE-SH", "D_EGE_SR": "D-EGE-SR",
          "D_EGE_SH_N": "D-EGE-SH-N", "D_EGE_SR_N": "D-EGE-SR-N", "EGE_SH": "EGE-SH (fixed)",
          "EGE_SR": "EGE-SR (fixed)"}


# --------------------------------------------------------------------------- M2
def ege_sh_zero_sample(budget: int, num_arms: int) -> int:
    rounds = math.ceil(math.log2(num_arms))
    active, cumulative, zero = num_arms, 0, 0
    for _ in range(rounds):
        cumulative += math.floor(budget / (active * rounds))
        dismissed = active - math.ceil(active / 2)
        if cumulative == 0:
            zero += dismissed
        active = math.ceil(active / 2)
    return zero + int(cumulative == 0)


def ege_sr_zero_sample(budget: int, num_arms: int) -> int:
    log_k = 0.5 + float(np.sum(1.0 / np.arange(2, num_arms + 1)))
    n = np.ceil(np.concatenate(([0.0], (budget - num_arms) / (log_k * (num_arms + 1 - np.arange(1, num_arms)))))).astype(int)
    cumulative, zero = 0, 0
    for round_index in range(1, num_arms):
        cumulative += max(0, int(n[round_index] - n[round_index - 1]))
        zero += int(cumulative == 0)
    return zero + int(cumulative == 0)


ZERO_SAMPLE = {"EGE_SH": ege_sh_zero_sample, "EGE_SR": ege_sr_zero_sample}


def zero_sample_table(specs: list[dict]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        K, B = spec["num_arms"], spec["budget"]
        for item in stage_plan(K, B):
            for base, fn in ZERO_SAMPLE.items():
                rows.append({"environment": spec["name"], "context": "doubling stage", "algorithm": base,
                             "stage": item["stage"], "budget": item["budget"], "stage_end": item["end"],
                             "normalized_stage_end": item["end"] / B, "zero_sample_classified": fn(item["budget"], K),
                             "num_arms": K})
        for budget in spec["grid"][:6]:
            for base, fn in ZERO_SAMPLE.items():
                rows.append({"environment": spec["name"], "context": "fixed-budget grid", "algorithm": base,
                             "stage": "", "budget": budget, "stage_end": budget, "normalized_stage_end": budget / B,
                             "zero_sample_classified": fn(budget, K), "num_arms": K})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- M1 / M3
def early_points_table(output_root: Path, specs: list[dict], config: dict) -> pd.DataFrame:
    rows = []
    for spec in specs:
        name, K, B = spec["name"], spec["num_arms"], spec["budget"]
        environment = _construct_environment(ENVIRONMENTS[name][0], config["dist"])
        P = len(environment.get_top_arms())
        grid1 = spec["grid"][1]
        frame = pd.read_csv(plot_input_path(output_root, name), usecols=["algorithm", "arm_pull", "jaccard", "bernoulli", "misidentification"])
        fixed = pd.read_csv(preregistered_dir(config) / f"{name}.csv", usecols=["algorithm", "arm_pull", "jaccard", "bernoulli", "misidentification"])
        frame = pd.concat([frame, fixed[fixed["algorithm"].isin(["EGE_SH", "EGE_SR"])]], ignore_index=True)
        for algorithm, group in frame.groupby("algorithm"):
            first = int(group["arm_pull"].min())
            at0 = group[group["arm_pull"] == first]
            at1 = group[group["arm_pull"] == grid1]
            rows.append({
                "environment": name, "num_arms": K, "pareto_size": P, "zero_information_jaccard": P / K,
                "budget": B, "first_grid_point": grid1, "normalized_first_grid_point": grid1 / B,
                "ttpfts_ape_uv_warmup_end": 4 * K, "warmup_covers_first_grid_point": int(4 * K > grid1),
                "algorithm": algorithm, "first_point": first,
                "jaccard_first_point": at0["jaccard"].mean(), "bernoulli_first_point": at0["bernoulli"].mean(),
                "misidentification_first_point": at0["misidentification"].mean(),
                "jaccard_first_grid_point": at1["jaccard"].mean() if len(at1) else np.nan,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- M4
def unsampled_counterfactual(specs: list[dict], config: dict, repetitions: list[int], horizon: float = 0.05) -> pd.DataFrame:
    """Re-run the early doubling stages and score both unsampled-arm conventions.

    ``zero``: unsampled arms keep a zero mean (production convention);
    ``exclude``: unsampled arms are left out of the empirical Pareto set
    (all arms when nothing has been sampled).
    """
    rows = []
    for spec in specs:
        name, K, B = spec["name"], spec["num_arms"], spec["budget"]
        limit = horizon * B
        endpoints = {s.start for s in doubling_schedule(K, B)}
        points = sorted(t for t in set(spec["grid"]) | endpoints if t <= limit)
        plan = [item for item in stage_plan(K, B) if item["start"] < limit]
        for algorithm, base in EMPIRICAL_ALGORITHMS.items():
            for repetition in repetitions:
                environment = _construct_environment(ENVIRONMENTS[name][0], config["dist"])
                env_tag = f"{name}|{config['dist']}"
                seed_rngs(int(config["master_seed"]), env_tag, "env", repetition)
                environment.reset()
                pareto = set(environment.get_top_arms().tolist())
                times, arms, rewards = [], [], []
                for item in plan:
                    seed_rngs(int(config["master_seed"]), env_tag, base, repetition, "budget", item["budget"])
                    recorder = SampleRecorder(environment)
                    EGE_ALGORITHMS[base](T=item["budget"], K=K, D=environment.num_objectives, environment=recorder)
                    blocks = [round_robin(*block) for block in recorder.blocks]
                    if not blocks:
                        continue
                    a = np.concatenate([b[0] for b in blocks])
                    r = np.concatenate([b[1] for b in blocks])
                    used = min(a.size, item["end"] - item["start"])
                    times.append(item["start"] + 1 + np.arange(used))
                    arms.append(a[:used])
                    rewards.append(r[:used])
                D = environment.num_objectives
                times = np.concatenate(times) if times else np.empty(0, int)
                arms = np.concatenate(arms) if arms else np.empty(0, int)
                rewards = np.concatenate(rewards) if rewards else np.empty((0, D))
                sums, counts, position = np.zeros((K, D)), np.zeros(K, dtype=int), 0
                for t in points:
                    upto = int(np.searchsorted(times, t, side="right"))
                    np.add.at(sums, arms[position:upto], rewards[position:upto])
                    np.add.at(counts, arms[position:upto], 1)
                    position = upto
                    sampled = counts > 0
                    means = np.zeros((K, D))
                    means[sampled] = sums[sampled] / counts[sampled, None]
                    zero = set(np.flatnonzero(_pareto_optimal_arms_mask(means)).tolist())
                    if sampled.any():
                        idx = np.flatnonzero(sampled)
                        exclude = set(idx[_pareto_optimal_arms_mask(means[idx])].tolist())
                    else:
                        exclude = set(range(K))
                    jaccard = {c: len(r & pareto) / len(r | pareto) for c, r in (("zero", zero), ("exclude", exclude))}
                    rows.append({"environment": name, "algorithm": algorithm, "repetition": repetition,
                                 "arm_pull": t, "normalized_budget": t / B, "num_arms": K,
                                 "unsampled_arms": int((~sampled).sum()),
                                 "jaccard_zero": jaccard["zero"], "jaccard_exclude": jaccard["exclude"],
                                 "unsampled_recommended_zero": len(zero - set(np.flatnonzero(sampled).tolist()))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- order sensitivity
def order_sensitivity(output_root: Path, specs: list[dict]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        rr = pd.read_csv(results_path(output_root, spec["name"], "trajectory"))
        block = pd.read_csv(results_path(output_root, spec["name"], "block"))
        block["algorithm"] = block["algorithm"].str.removesuffix("_BLOCK")
        for algorithm in EMPIRICAL_ALGORITHMS:
            a = rr[rr["algorithm"] == algorithm].groupby("arm_pull")["jaccard"].mean()
            b = block[block["algorithm"] == algorithm].groupby("arm_pull")["jaccard"].mean()
            diff = (a - b).abs()
            rows.append({"environment": spec["name"], "algorithm": algorithm,
                         "max_abs_mean_jaccard_difference": diff.max(),
                         "normalized_budget_of_max": diff.idxmax() / spec["budget"],
                         "mean_abs_mean_jaccard_difference": diff.mean(),
                         "final_difference": float(a.iloc[-1] - b.iloc[-1])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- M5
def log_zero_table(config: dict) -> pd.DataFrame:
    """Rendered misidentification points a log axis cannot show: exact-zero means
    (omitted) and positive means whose clipped lower bound is zero (band reaches the axis floor)."""
    figures_dir = REPO_ROOT / config["figures_dir"]
    frames = []
    for set_name in config["figure_sets"]:
        table = pd.read_csv(figures_dir / f"publication_misidentification_{set_name}_plot_table.csv")
        table["figure_set"] = set_name
        frames.append(table)
    table = pd.concat(frames, ignore_index=True)
    table["issue"] = np.where(table["mean"] == 0.0, "mean omitted (exact zero)",
                              np.where(table["lower"] <= 0.0, "band reaches axis floor (lower bound 0)", ""))
    table = table[table["issue"] != ""]
    return (table.groupby(["environment", "algorithm", "issue"])
            .agg(points=("arm_pull", "nunique"), first=("normalized_budget", "min"), last=("normalized_budget", "max"),
                 figure_sets=("figure_set", lambda v: ",".join(sorted(set(v)))))
            .reset_index())


# ------------------------------------------------------------------- diagnostic
def diagnostic_figure(output_root: Path, specs: list[dict], config: dict, path: Path) -> None:
    chosen = ["EgeExp1", "EgeExp4", "EgeExp6", "EgeExp7"]
    curves = ["Uniform", "TTPFTS_UNKNOWN_VAR", "APE_UV", "D_EGE_SH", "D_EGE_SR_N", "EGE_SR"]
    colors = {"Uniform": "C2", "TTPFTS_UNKNOWN_VAR": "C4", "APE_UV": "C3", "D_EGE_SH": "C0", "D_EGE_SR_N": "C1", "EGE_SR": "0.35"}
    styles = {"D_EGE_SR_N": "--", "EGE_SR": ":"}
    plt.rcParams.update({"font.size": 11, "font.family": "serif"})
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    for ax, name in zip(axes, chosen):
        spec = next(s for s in specs if s["name"] == name)
        K, B = spec["num_arms"], spec["budget"]
        frame = pd.read_csv(plot_input_path(output_root, name), usecols=["algorithm", "arm_pull", "jaccard"])
        fixed = pd.read_csv(preregistered_dir(config) / f"{name}.csv", usecols=["algorithm", "arm_pull", "jaccard"])
        frame = pd.concat([frame, fixed[fixed["algorithm"] == "EGE_SR"]], ignore_index=True)
        P = len(_construct_environment(ENVIRONMENTS[name][0], config["dist"]).get_top_arms())
        for algorithm in curves:
            mean = frame[frame["algorithm"] == algorithm].groupby("arm_pull")["jaccard"].mean()
            mean = mean[mean.index <= 0.1 * B + 1]
            ax.plot(mean.index / B, mean.values, marker="o", markersize=3, linewidth=1.3,
                    color=colors[algorithm], linestyle=styles.get(algorithm, "-"), label=LABELS[algorithm])
        ax.axhline(P / K, color="k", linewidth=0.8, linestyle=":")
        ax.axvline(K / B, color="0.6", linewidth=0.8, linestyle="--")
        ax.axvline(4 * K / B, color="0.6", linewidth=0.8, linestyle="-.")
        ax.set_xlim(0, 0.1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Environment {name.removeprefix('EgeExp')}  (|P|/K = {P / K:.2f})", fontsize=11)
        ax.grid(True, alpha=0.22, linewidth=0.55)
    axes[0].set_ylabel("Jaccard Metric")
    fig.supxlabel("Normalized Budget (first 10%)", fontsize=11, y=0.06)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=True, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------- report
def _md(frame: pd.DataFrame, floatfmt: str = ".3f") -> str:
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    lines = [header, rule]
    for _, row in frame.iterrows():
        cells = [format(v, floatfmt) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(output_root: Path | None = None, config: dict | None = None, repetitions: list[int] | None = None) -> Path:
    config = load_config() if config is None else config
    output_root = Path(default_output_root(config) if output_root is None else output_root)
    assert_safe_output(REPORT_DIR, config)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    specs = environment_specs(config)
    repetitions = list(range(int(config["num_runs"]))) if repetitions is None else repetitions

    early = early_points_table(output_root, specs, config)
    zero = zero_sample_table(specs)
    counter = unsampled_counterfactual(specs, config, repetitions)
    order = order_sensitivity(output_root, specs)
    logs = log_zero_table(config)
    early.to_csv(REPORT_DIR / "early_first_points.csv", index=False)
    zero.to_csv(REPORT_DIR / "ege_zero_sample_classification.csv", index=False)
    counter["partial"] = (counter["unsampled_arms"] > 0) & (counter["unsampled_arms"] < counter["num_arms"])
    counter["abs_difference"] = (counter["jaccard_zero"] - counter["jaccard_exclude"]).abs()
    counter_summary = (counter.groupby(["environment", "algorithm"])
                       .agg(early_states=("arm_pull", "size"), partially_sampled_states=("partial", "sum"),
                            states_recommending_unsampled_arm=("unsampled_recommended_zero",
                                                               lambda v: int(((v > 0) & counter.loc[v.index, "partial"]).sum())),
                            mean_jaccard_zero=("jaccard_zero", "mean"), mean_jaccard_exclude=("jaccard_exclude", "mean"),
                            max_state_difference=("abs_difference", "max"))
                       .reset_index())
    counter_summary.to_csv(REPORT_DIR / "unsampled_arm_counterfactual.csv", index=False)
    order.to_csv(REPORT_DIR / "pull_order_sensitivity.csv", index=False)
    logs.to_csv(REPORT_DIR / "log_scale_zero_omissions.csv", index=False)
    diagnostic_figure(output_root, specs, config, REPORT_DIR / "early_curve_artifacts.png")

    # ---- narrative tables
    uniform = early[early["algorithm"] == "Uniform"].set_index("environment")
    t0 = early[early["first_point"] == 0]
    m1 = pd.DataFrame({
        "Env": [n.removeprefix("EgeExp") for n in uniform.index],
        "K": uniform["num_arms"].values, "|P|": uniform["pareto_size"].values,
        "|P|/K": uniform["zero_information_jaccard"].values,
        "Jaccard(t=0)": uniform["jaccard_first_point"].values,
        "Bernoulli(t=0)": uniform["bernoulli_first_point"].values,
        "Uniform Jaccard(first grid)": uniform["jaccard_first_grid_point"].values,
        "first grid / B": uniform["normalized_first_grid_point"].values,
        "4K > first grid": ["yes" if v else "no" for v in uniform["warmup_covers_first_grid_point"].values],
    })
    all_equal = all(abs(r.jaccard_first_point - r.zero_information_jaccard) < 1e-12
                    for r in t0[t0["algorithm"].isin(ANYTIME + ["EGE_SH"])].itertuples())
    sr0 = early[(early["algorithm"] == "EGE_SR") & (early["first_point"] == 0)]
    m2 = (zero[(zero["context"] == "doubling stage") & (zero["zero_sample_classified"] > 0)]
          .assign(Env=lambda f: f["environment"].str.removeprefix("EgeExp"))
          .groupby(["Env", "algorithm"])
          .apply(lambda g: ", ".join(f"s{int(s)}: {int(z)}/{int(k)}" for s, z, k in zip(g["stage"], g["zero_sample_classified"], g["num_arms"])))
          .reset_index(name="stage: arms classified without a sample"))
    grid_zero = zero[(zero["context"] == "fixed-budget grid") & (zero["zero_sample_classified"] > 0) & (zero["budget"] > 0)]
    m4 = counter_summary.assign(Env=counter_summary["environment"].str.removeprefix("EgeExp")).rename(columns={
        "partially_sampled_states": "partially sampled states", "states_recommending_unsampled_arm": "of which recommend an unsampled arm",
        "mean_jaccard_zero": "mean J (zero)", "mean_jaccard_exclude": "mean J (exclude)", "max_state_difference": "max |ΔJ| in one state"})
    m4 = m4[["Env", "algorithm", "partially sampled states", "of which recommend an unsampled arm", "mean J (zero)",
             "mean J (exclude)", "max |ΔJ| in one state"]]
    last_unsampled = max(
        float(frame.loc[frame["unsampled_arms"] > 0, "arm_pull"].max() / spec["budget"])
        for spec in specs
        for frame in [pd.read_csv(results_path(output_root, spec["name"], "trajectory"), usecols=["arm_pull", "unsampled_arms"])]
        if (frame["unsampled_arms"] > 0).any()
    )
    m4_mean_gap = float((counter_summary["mean_jaccard_zero"] - counter_summary["mean_jaccard_exclude"]).abs().max())
    m3 = early[(early["environment"] == "EgeExp7") & early["algorithm"].isin(["Uniform", "TTPFTS_KNOWN_VAR", "TTPFTS_UNKNOWN_VAR"])]
    m3_text = ", ".join(f"{LABELS[r.algorithm]} {r.jaccard_first_grid_point:.3f}" for r in m3.itertuples())
    order_md = order.assign(Env=order["environment"].str.removeprefix("EgeExp"))[
        ["Env", "algorithm", "max_abs_mean_jaccard_difference", "normalized_budget_of_max", "final_difference"]]
    log_summary = logs.assign(Env=logs["environment"].str.removeprefix("EgeExp"))[
        ["Env", "algorithm", "issue", "points", "first", "last", "figure_sets"]]
    omitted = logs[logs["issue"].str.startswith("mean omitted")]
    floor = logs[logs["issue"].str.startswith("band")]

    text = f"""# Early-curve artifacts in the publication figures

This report investigates the behavior of the first evaluation points of the
Jaccard, Bernoulli, and misidentification curves. **No figure was changed**:
the information-matched figures use the same conventions as the existing
publication figures. All numbers below come from the validated data in
`{output_root.name}` and from the preregistered results.
Data tables are in this directory (`*.csv`), and the diagnostic plot is
`early_curve_artifacts.png`.

![Early-budget diagnostic](early_curve_artifacts.png)

The figure shows mean Jaccard over the first 10% of the budget. Markers are
evaluation points. The dotted horizontal line is |P|/K, the dashed vertical
line is K/B, and the dash-dotted vertical line is 4K/B, the end of the
TTPFTS and APE-UV warm-up.

## M1 – zero-information recommendation at t = 0 (affects every anytime curve)

Before any pull, all empirical (and posterior) means are zero. No arm then
strictly dominates another, so **every anytime algorithm recommends all K
arms** at t = 0. The t = 0 point therefore measures the zero-information
convention, not learning: Jaccard = |P|/K, Bernoulli = 1 exactly when every
arm is Pareto-optimal, and misidentification = (K − |P|)/K.
This holds exactly for every anytime curve (Uniform, APE, APE-UV, TTPFTS-KV,
TTPFTS-UV, the empirical-mean D-EGE) and for fixed-budget EGE-SH:
**{'verified' if all_equal else 'NOT verified'}**. Fixed-budget EGE-SR at T = 0
instead returns a single random arm, with t = 0 Jaccard between
{sr0['jaccard_first_point'].min():.3f} and {sr0['jaccard_first_point'].max():.3f}.

{_md(m1)}

Consequences:
- **Environment 6** has |P| = K, so every anytime curve starts at a
  *perfect* score (Jaccard = Bernoulli = 1, misidentification = 0). The score
  then drops to about {uniform.loc['EgeExp6', 'jaccard_first_grid_point']:.2f}
  at the first grid point.
- **Environment 7** (|P|/K = 0.36) starts at 0.36 and dips to about
  {uniform.loc['EgeExp7', 'jaccard_first_grid_point']:.2f}.

  Both shapes are artifacts of the t = 0 convention, not of learning. Where
  |P|/K is small, the t = 0 point instead lies below the first informed
  point, so no dip appears.

## M2 – native EGE classifies arms that were never sampled (D-EGE-N stages, fixed-budget EGE)

The native implementations initialize means to zero and classify arms even
when the budget leaves them unsampled:
- EGE-SR draws nothing at T ≤ K, because its allocation `n_k` is
  `ceil((T − K)/…) = 0`. It rejects every arm except a random survivor.
- EGE-SH gives zero pulls per arm in its first round whenever
  T < K·⌈log₂K⌉. It then dismisses half of the arms on randomly tie-broken
  zero means.

Arms classified without a single sample, listed per doubling stage s as
"zero-sample arms / K":

{_md(m2)}

The first D-EGE-SR-N stage (budget K) is therefore always a single random
arm. This is why the D-EGE-SR-N curves start near zero. The first two to
three D-EGE-SH-N stages partly dismiss arms at random.

The same mechanism affects the original fixed-budget EGE curves at their
first grid budgets: {', '.join(sorted({f"{r.environment.removeprefix('EgeExp')} ({r.algorithm}, T = {r.budget}: {r.zero_sample_classified}/{r.num_arms})" for r in grid_zero.itertuples()})) or 'none beyond T = 0'}.

## M3 – warm-up still covers the first grid point (no defect, but identical curves)

TTPFTS (both variants) and APE-UV spend their first 4K pulls on a
round-robin warm-up, and APE spends its first K pulls the same way. Where
4K exceeds the first grid point (the `4K > first grid` column above), these
algorithms are still sampling uniformly at that point. Their first informed
points are therefore Uniform in distribution. For example, in Environment 7
at t = 70, the mean Jaccard values are {m3_text}, although the individual
recommendations differ. This explains coinciding early curves; it is not an
implementation fault.

## M4 – zero-initialized unsampled arms (empirical-mean D-EGE)

Because of M2, the empirical-mean D-EGE has unsampled arms during its first
stages. Under the production convention, an unsampled arm keeps a zero mean,
which is the same rule as Uniform and APE. The counterfactual below re-runs
the stages with identical samples and excludes unsampled arms from the
empirical Pareto set. It covers every evaluation point up to 5% of B.

{_md(m4)}

A "partially sampled state" is a repetition and evaluation point at which
some, but not all, arms are unsampled. With means in [0, 1], a zero mean is
dominated by almost every sampled arm, so the two conventions rarely differ.
The largest difference in mean early Jaccard between the conventions is
{m4_mean_gap:.4f}. Across all 100 repetitions of every environment, the last
evaluation point with any unsampled arm lies at {100 * last_unsampled:.2f}% of B,
well inside the 5% horizon. Beyond it the two conventions are identical.

## Pull order within rounds (empirical-mean D-EGE sensitivity)

The empirical-mean curves interleave each round's pulls round-robin. Native
block order (arm by arm) would place every sample of the lower-index arms
first. The largest difference between the mean Jaccard curves of the two
orders is:

{_md(order_md)}

## M5 – log-scale misidentification omits exact zeros

On log axes an exact zero cannot be drawn.
- **Omitted means:** an exact-zero mean is silently omitted. This happens at
  {len(omitted)} environment/algorithm combinations
  ({', '.join(sorted({f"{r.environment.removeprefix('EgeExp')}/{LABELS.get(r.algorithm, r.algorithm)}" for r in omitted.itertuples()})) or 'none'}),
  all at the very start of the curves.
- **Bands reaching the axis floor:** a positive mean whose interval lower
  bound is clipped to 0 makes the shaded band run to the bottom of the axis.
  This affects {len(floor)} combinations. The mean line is still drawn.

{_md(log_summary) if len(log_summary) else 'None.'}

## Remedy options (not applied)

1. **Treat t = 0 as a convention.** Either start every curve at the first
   grid point, or draw t = 0 as a separate, explicitly labelled marker.
   This removes the Environment 6 spike and the Environment 7 dip, and it
   leaves every informed point unchanged.
2. **Compare after a common initialization budget.** This could be max(4K)
   across algorithms, or an early-budget inset with a logarithmic x-axis, so
   that warm-up phases (M3) are visible rather than hidden in the first 1–2%
   of the axis.
3. **Disclose native EGE behavior at small budgets.** This applies to
   fixed-budget EGE at T = 0 and the first one to three doubling stages
   (M2). Options are a caption note, or starting the D-EGE curves at the
   first stage that samples every arm. Changing EGE itself would no longer
   be the published algorithm.
4. **State the unsampled-arm rule explicitly** (M4). Its effect is
   quantified above.
5. **Mark exact zeros on log axes** (M5), for example with a clipped floor
   marker or a symlog axis, instead of omitting them.
"""
    path = REPORT_DIR / "early_curve_artifacts.md"
    path.write_text(text)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    print(write_report(output_root=Path(args.output_root) if args.output_root else None))


if __name__ == "__main__":
    main()
