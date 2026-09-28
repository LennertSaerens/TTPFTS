"""Publication plotting utilities for MOMAB Pareto-set-identification results.

New experiment outputs use the tidy, named-column schema from ``runner/schema.py``
plus a ``RunManifest`` sidecar.  Legacy archive CSV support is intentionally kept
in a separate reader so publication plotting never silently depends on positional
renaming for current results.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from statistics import NormalDist
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

try:
    from runner.schema import SCHEMA_COLUMNS
except Exception:  # pragma: no cover - plotting can still handle legacy archives.
    SCHEMA_COLUMNS = []

CANONICAL_METRICS = {
    "bernoulli": "bernoulli",
    "Bernoulli": "bernoulli",
    "jaccard": "jaccard",
    "Jaccard": "jaccard",
    "misidentification": "misidentification",
    "Misidentification": "misidentification",
    "Misclassification": "misidentification",
}

METRIC_LABELS = {
    "bernoulli": "Bernoulli Metric",
    "jaccard": "Jaccard Metric",
    "misidentification": "Misidentification Metric",
}

LEGACY_COLUMNS = ["algorithm", "e", "t", "Bernoulli", "Jaccard", "Misclassification"]

DEFAULT_ALGORITHM_ORDER = [
    "EGE_SH",
    "EGE_SR",
    "Uniform",
    "APE",
    "TTPFTS_KNOWN_VAR",
    "TTPFTS_UNKNOWN_VAR",
]

DEFAULT_EGE_BUDGETS = {
    "EgeExp1": 15218,
    "EgeExp2": 2544,
    "EgeExp3": 685036,
    "EgeExp4": 15977,
    "EgeExp5": 197373,
    "EgeExp6": 51966,
    "EgeExp7": 7000,
    "EgeExp8": 237958,
}

PUBLICATION_FONT_SIZE = 12


def _canonical_metric(metric: str) -> str:
    try:
        return CANONICAL_METRICS[metric]
    except KeyError as exc:
        raise RuntimeError(f"Metric type {metric} is not supported") from exc


def _infer_environment_from_path(path: str | os.PathLike) -> str | None:
    match = re.search(r"EgeExp\d+", str(path))
    return match.group(0) if match else None


def _find_manifest_path(files: Sequence[str | os.PathLike], manifest_path: str | os.PathLike | None = None) -> Path | None:
    if manifest_path is not None:
        return Path(manifest_path)
    candidates: list[Path] = []
    for file in files:
        p = Path(file)
        candidates.extend([
            p.parent / "run" / "manifest.json",
            p.parent / "manifest.json",
            p.parent.parent / "run" / "manifest.json",
            p.parent.parent / "manifest.json",
        ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_run_manifest(manifest_path: str | os.PathLike | None) -> dict:
    """Read a runner ``RunManifest`` JSON sidecar into a plain dictionary."""
    if manifest_path is None:
        return {}
    with open(manifest_path, "r") as f:
        return json.load(f)


def load_tidy_results(file: str | os.PathLike) -> pd.DataFrame:
    """Load current runner output using named schema columns only.

    This is the publication path for new experiments.  It refuses non-tidy CSVs
    rather than assigning positional column names, preventing accidental plotting
    of malformed current data.
    """
    df = pd.read_csv(file, header=0, low_memory=False)
    required = {
        "environment", "algorithm", "repetition", "arm_pull", "max_budget",
        "bernoulli", "jaccard", "misidentification",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{file} is not a tidy runner CSV; missing columns: {sorted(missing)}")

    out = df.loc[:, [
        "environment", "algorithm", "repetition", "arm_pull", "max_budget",
        "bernoulli", "jaccard", "misidentification",
    ]].copy()
    out["posterior_model"] = df["posterior_model"] if "posterior_model" in df.columns else ""
    for col in ["repetition", "arm_pull", "max_budget"]:
        out[col] = pd.to_numeric(out[col], errors="raise").astype(int)
    for col in ["bernoulli", "jaccard", "misidentification"]:
        out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
    return out


def load_legacy_results(
    file: str | os.PathLike,
    *,
    environment: str | None = None,
    max_budget: int = 5000,
) -> pd.DataFrame:
    """Load archived pre-runner CSVs with the old six-column layout.

    This reader is deliberately separate from ``load_tidy_results``.  It accepts
    the legacy header names when present and, only for archive compatibility,
    falls back to the historical positional layout.
    """
    raw = pd.read_csv(file, header=0, low_memory=False)
    if set(LEGACY_COLUMNS).issubset(raw.columns):
        legacy = raw.loc[:, LEGACY_COLUMNS].copy()
    elif raw.shape[1] == len(LEGACY_COLUMNS):
        legacy = raw.copy()
        legacy.columns = LEGACY_COLUMNS
    else:
        raise ValueError(f"{file} is neither tidy runner output nor known legacy archive layout")

    env = environment or _infer_environment_from_path(file) or "legacy"
    out = pd.DataFrame({
        "environment": env,
        "algorithm": legacy["algorithm"].astype(str),
        "posterior_model": "",
        "repetition": pd.to_numeric(legacy["e"], errors="raise").astype(int),
        "arm_pull": pd.to_numeric(legacy["t"], errors="raise").astype(int),
        "max_budget": int(max_budget),
        "bernoulli": pd.to_numeric(legacy["Bernoulli"], errors="raise").astype(float),
        "jaccard": pd.to_numeric(legacy["Jaccard"], errors="raise").astype(float),
        "misidentification": pd.to_numeric(legacy["Misclassification"], errors="raise").astype(float),
    })
    return out


def load_results(
    file: str | os.PathLike,
    *,
    legacy: bool = False,
    environment: str | None = None,
    legacy_max_budget: int = 5000,
) -> pd.DataFrame:
    """Load one result CSV, selecting tidy or explicitly legacy parsing."""
    if legacy:
        return load_legacy_results(file, environment=environment, max_budget=legacy_max_budget)
    return load_tidy_results(file)


def _environment_budgets(
    frames: Sequence[pd.DataFrame],
    manifest: Mapping | None = None,
    overrides: Mapping[str, int] | None = None,
) -> dict[str, int]:
    budgets: dict[str, int] = {}
    if manifest and manifest.get("budgets"):
        budgets.update({str(k): int(v) for k, v in manifest["budgets"].items()})
    for frame in frames:
        for env, env_df in frame.groupby("environment", sort=False):
            values = set(pd.to_numeric(env_df["max_budget"], errors="raise").astype(int).unique())
            if len(values) == 1:
                budgets.setdefault(str(env), values.pop())
    if overrides:
        budgets.update({str(k): int(v) for k, v in overrides.items()})
    missing = sorted({str(env) for frame in frames for env in frame["environment"].unique()} - set(budgets))
    if missing:
        raise ValueError(f"No max budget available for environment(s): {missing}")
    return budgets


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a Bernoulli/binomial mean.

    Used for the binary exact-identification metric.  The implementation is the
    closed-form score interval, deterministic and independent of scipy.
    """
    if n <= 0:
        return (np.nan, np.nan)
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half_width = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


def bounded_mean_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    """Normal repetition-level mean interval clipped to the metric's [0, 1] range.

    Jaccard and misidentification are bounded per-repetition metrics but not
    binary exact-identification indicators.  We therefore aggregate over the
    actually recorded repetitions at each budget and report mean +/- z*SE,
    clipped to [0, 1].  No bootstrap randomness is used.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n <= 0:
        return (np.nan, np.nan)
    if n == 1:
        mean = float(arr[0])
        return max(0.0, mean), min(1.0, mean)
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(n))
    return max(0.0, mean - z * se), min(1.0, mean + z * se)


def summarize_metric_for_plot(
    df: pd.DataFrame,
    metric: str,
    budgets: Mapping[str, int],
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Return plotted means/intervals at recorded budgets only.

    The function groups by environment, algorithm, and raw ``arm_pull`` values
    present in the CSV; it never interpolates or fills sparse evaluation grids.
    Normalized x-values are ``arm_pull / environment_max_budget``.  Each plotted
    environment/algorithm curve must contain its manifest final budget, making
    the endpoint exactly 1.0.
    """
    metric_key = _canonical_metric(metric)
    rows: list[dict] = []
    group_cols = ["environment", "algorithm", "arm_pull"]
    for (env, algorithm, arm_pull), group in df.groupby(group_cols, sort=True):
        max_budget = int(budgets[str(env)])
        if int(arm_pull) > max_budget:
            raise AssertionError(f"{env}/{algorithm} arm_pull={arm_pull} exceeds max_budget={max_budget}")
        values = group[metric_key].astype(float).to_numpy()
        n = int(values.size)
        mean = float(np.mean(values)) if n else np.nan
        if metric_key == "bernoulli":
            lower, upper = wilson_interval(int(np.sum(values)), n, confidence=confidence)
        else:
            lower, upper = bounded_mean_interval(values, confidence=confidence)
        rows.append({
            "environment": str(env),
            "algorithm": str(algorithm),
            "metric": metric_key,
            "arm_pull": int(arm_pull),
            "max_budget": max_budget,
            "normalized_budget": float(int(arm_pull) / max_budget) if max_budget else 0.0,
            "mean": mean,
            "lower": lower,
            "upper": upper,
            "n": n,
        })
    summary = pd.DataFrame(rows).sort_values(["environment", "algorithm", "arm_pull"]).reset_index(drop=True)
    for (env, algorithm), curve in summary.groupby(["environment", "algorithm"], sort=False):
        final = float(curve["normalized_budget"].max())
        if final != 1.0:
            raise AssertionError(
                f"Final normalized budget for {env}/{algorithm} is {final}, expected exactly 1.0"
            )
    return summary


def _algorithm_sort_key(name: str, order: Sequence[str]) -> tuple[int, str]:
    try:
        return (order.index(name), name)
    except ValueError:
        if "_p" in name:
            try:
                return (len(order), f"{float(name.split('_p')[-1]):.12g}")
            except ValueError:
                pass
        return (len(order), name)


def _algorithm_label(name: str, format_legend_p: bool = False) -> str:
    if format_legend_p and "_p" in name:
        try:
            return f"p = {float(name.split('_p')[-1])}"
        except ValueError:
            pass
    return name


def _environment_label(name: str) -> str:
    """Return the compact publication label for a named EgeExp environment."""
    match = re.fullmatch(r"EgeExp(\d+)", name)
    return match.group(1) if match else name


def plot_pfi_metric(file, num_runs=None, num_arm_pulls=None, metric="Jaccard", rolling_avg_window=1,
                    plot_std=True, save_pdf=False, step=1, *, manifest_path=None, legacy=False,
                    show=True):
    """Plot one metric from a single tidy CSV or explicit legacy archive CSV."""
    df = load_results(file, legacy=legacy)
    manifest = read_run_manifest(_find_manifest_path([file], manifest_path))
    budgets = _environment_budgets([df], manifest)
    metric_key = _canonical_metric(metric)
    summary = summarize_metric_for_plot(df, metric_key, budgets)
    fig, ax = plt.subplots(figsize=(8, 6))
    algorithms = sorted(summary["algorithm"].unique(), key=lambda x: _algorithm_sort_key(x, DEFAULT_ALGORITHM_ORDER))
    for algorithm in algorithms:
        curve = summary[summary["algorithm"] == algorithm]
        line, = ax.plot(curve["normalized_budget"], curve["mean"], label=_algorithm_label(algorithm))
        if plot_std:
            ax.fill_between(curve["normalized_budget"], curve["lower"], curve["upper"], alpha=0.18,
                            color=line.get_color())
    ax.set_xlabel("Normalized Budget")
    ax.set_ylabel(METRIC_LABELS[metric_key])
    ax.grid(True, alpha=0.22, linewidth=0.55)
    ax.legend(loc="best")
    if metric_key in {"bernoulli", "jaccard"}:
        ax.set_ylim(0, 1)
    if save_pdf:
        fig.savefig(f"{file}_{metric_key}.pdf", format="pdf")
    if show:
        plt.show()
    return fig, ax, summary


def plot_all_pfi_metrics(file, num_runs=None, num_arm_pulls=None, rolling_avg_window=1,
                         plot_std=True, save_pdf=False, step=1, **kwargs):
    return [
        plot_pfi_metric(file, num_runs, num_arm_pulls, metric, rolling_avg_window, plot_std, save_pdf, step, **kwargs)
        for metric in ["Bernoulli", "Jaccard", "Misidentification"]
    ]


def plot_pfi_metric_ax(
        ax,
        file,
        num_runs=None,
        num_arm_pulls=None,
        num_arms=None,
        num_objs=None,
        metric="Jaccard",
        plot_std=True,
        step=1,
        show_legend=False,
        ylabel=None,
        title=None,
        cmap=None,
        format_legend_p=False,
        exclude_algs=None,
        sort_algs=True,
        *,
        manifest_path=None,
        legacy=False,
        environment_budget=None,
        algorithm_order: Sequence[str] = DEFAULT_ALGORITHM_ORDER,
        return_table=False,
):
    """Plot the publication metric curve for one subplot.

    X-values are normalized by the environment's manifest max budget, intervals
    are pointwise Wilson (Bernoulli) or bounded repetition-level normal intervals
    (Jaccard/misidentification), and only budgets recorded in the CSV are drawn.
    """
    df = load_results(file, legacy=legacy)
    manifest = read_run_manifest(_find_manifest_path([file], manifest_path))
    overrides = None
    if environment_budget is not None:
        env = str(df["environment"].iloc[0])
        overrides = {env: int(environment_budget)}
    budgets = _environment_budgets([df], manifest, overrides)
    summary = summarize_metric_for_plot(df, metric, budgets)

    algorithms = list(summary["algorithm"].unique())
    if exclude_algs:
        algorithms = [alg for alg in algorithms if alg not in set(exclude_algs)]
    if sort_algs:
        algorithms = sorted(algorithms, key=lambda x: _algorithm_sort_key(x, algorithm_order))

    for i, algorithm in enumerate(algorithms):
        curve = summary[summary["algorithm"] == algorithm]
        if step and step > 1:
            # Downsample displayed rows only; never synthesize missing observations.
            raw_stride = np.maximum(1, np.median(np.diff(curve["arm_pull"])) if len(curve) > 1 else 1)
            row_stride = max(1, int(step // raw_stride))
            curve = curve.iloc[::row_stride]
            if curve["normalized_budget"].iloc[-1] != 1.0:
                curve = pd.concat([curve, summary[(summary["algorithm"] == algorithm) &
                                                  (summary["normalized_budget"] == 1.0)]], ignore_index=True)
                curve = curve.drop_duplicates(subset=["environment", "algorithm", "arm_pull"]).sort_values("arm_pull")
        line_color = cmap(i / max(1, len(algorithms) - 1)) if cmap is not None else None
        label = _algorithm_label(algorithm, format_legend_p=format_legend_p)
        line, = ax.plot(curve["normalized_budget"], curve["mean"], label=label, color=line_color)
        if plot_std:
            ax.fill_between(curve["normalized_budget"], curve["lower"], curve["upper"],
                            alpha=0.18, color=line.get_color(), linewidth=0)

    metric_key = _canonical_metric(metric)
    ax.set_xlim(0, 1)
    if metric_key in {"bernoulli", "jaccard"}:
        ax.set_ylim(0, 1)
    elif metric_key == "misidentification":
        ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.22, linewidth=0.55)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=14)
    if show_legend:
        ax.legend(loc="best", fontsize=8)
    if return_table:
        return summary
    return None


def _read_grid_frames(env_files, env_labels, *, legacy=False, legacy_max_budget=5000):
    frames = []
    for file, env in zip(env_files, env_labels):
        frames.append(load_results(file, legacy=legacy, environment=env, legacy_max_budget=legacy_max_budget))
    return frames


def plot_all_pfi_metrics_grid_top_legend(
        env_files,
        env_labels,
        num_runs=None,
        num_arm_pulls=None,
        num_arms=None,
        num_objs=None,
        plot_std=True,
        step=1,
        figsize=(12, 24),
        save_png=False,
        save_file=None,
        cmap=None,
        format_legend_p=False,
        *,
        manifest_path=None,
        environment_budgets=None,
        table_file=None,
        legacy=False,
        show=True,
):
    outputs = []
    for metric in ["Bernoulli", "Jaccard", "Misidentification"]:
        outputs.append(plot_metric_all_envs_grid_top_legend(
            env_files, env_labels, num_arms or [None] * len(env_files), num_objs or [None] * len(env_files),
            num_runs, num_arm_pulls, metric, plot_std=plot_std, step=step, figsize=figsize,
            save_png=save_png, save_file=(None if save_file is None else f"{Path(save_file).with_suffix('')}_{_canonical_metric(metric)}.png"),
            cmap=cmap, format_legend_p=format_legend_p, manifest_path=manifest_path,
            environment_budgets=environment_budgets, table_file=table_file, legacy=legacy, show=show,
        ))
    return outputs


def plot_metric_all_envs_grid_top_legend(
        env_files,
        env_labels,
        env_arms,
        env_objs,
        num_runs=None,
        num_arm_pulls=None,
        metric="Jaccard",
        plot_std=True,
        step=1,
        figsize=(13, 6),
        save_png=False,
        save_file=None,
        cmap=None,
        format_legend_p=False,
        exclude_algs=None,
        sort_algs=True,
        *,
        manifest_path=None,
        environment_budgets=None,
        table_file=None,
        legacy=False,
        legacy_max_budget=5000,
        algorithm_order: Sequence[str] = DEFAULT_ALGORITHM_ORDER,
        algorithm_display_labels: Mapping[str, str] | None = None,
        annotate_environments: bool = True,
        log_misidentification: bool = False,
        show=True,
):
    """Plot a 2x4 publication grid with shared labels and top framed legend.

    Returns ``(fig, axes, plotted_table)``.  When ``save_png`` is true the figure
    is saved at publication resolution (500 dpi).  ``plotted_table`` can also be
    written as CSV via ``table_file`` and contains means, interval bounds, and
    sample counts for every plotted point.
    """
    metric_key = _canonical_metric(metric)
    n_env = len(env_files)
    ncols = min(4, n_env)
    nrows = int(np.ceil(n_env / ncols))

    frames = _read_grid_frames(env_files, env_labels, legacy=legacy, legacy_max_budget=legacy_max_budget)
    manifest = read_run_manifest(_find_manifest_path(env_files, manifest_path))
    budgets = _environment_budgets(frames, manifest, environment_budgets)
    combined = pd.concat(frames, ignore_index=True)
    full_summary = summarize_metric_for_plot(combined, metric_key, budgets)

    use_log_y = metric_key == "misidentification" and log_misidentification
    sharey = False if metric_key == "misidentification" else True
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )

    plotted_tables: list[pd.DataFrame] = []
    for i, env_name in enumerate(env_labels):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]
        n_arms = env_arms[i] if env_arms is not None else None
        n_objs = env_objs[i] if env_objs is not None else None
        if annotate_environments:
            ax.text(
                0.97,
                0.05,
                _environment_label(env_name),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=PUBLICATION_FONT_SIZE,
                color="0.25",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            )

        env_summary = full_summary[full_summary["environment"] == env_name]
        algorithms = list(env_summary["algorithm"].unique())
        if exclude_algs:
            algorithms = [alg for alg in algorithms if alg not in set(exclude_algs)]
        if sort_algs:
            algorithms = sorted(algorithms, key=lambda x: _algorithm_sort_key(x, algorithm_order))
        for alg_idx, algorithm in enumerate(algorithms):
            curve = env_summary[env_summary["algorithm"] == algorithm].sort_values("arm_pull")
            if curve.empty:
                continue
            if step and step > 1:
                raw_stride = np.maximum(1, np.median(np.diff(curve["arm_pull"])) if len(curve) > 1 else 1)
                row_stride = max(1, int(step // raw_stride))
                shown = curve.iloc[::row_stride].copy()
                if shown["normalized_budget"].iloc[-1] != 1.0:
                    shown = pd.concat([shown, curve[curve["normalized_budget"] == 1.0]], ignore_index=True)
                    shown = shown.drop_duplicates(subset=["environment", "algorithm", "arm_pull"]).sort_values("arm_pull")
            else:
                shown = curve
            color = cmap(alg_idx / max(1, len(algorithms) - 1)) if cmap is not None else None
            means = shown["mean"]
            if use_log_y:
                # Exact zero misidentification has no location on a strict log
                # axis, so omit it rather than depicting it as a positive value.
                means = means.where(means > 0)
            label = (algorithm_display_labels or {}).get(
                algorithm, _algorithm_label(algorithm, format_legend_p)
            )
            line, = ax.plot(shown["normalized_budget"], means, label=label, color=color)
            if plot_std:
                lower = shown["lower"]
                upper = shown["upper"]
                if use_log_y:
                    lower = lower.where(means.notna())
                    upper = upper.where(means.notna())
                ax.fill_between(
                    shown["normalized_budget"],
                    lower,
                    upper,
                    alpha=0.18,
                    color=line.get_color(),
                    linewidth=0,
                )
            plotted_tables.append(shown)
        ax.set_xlim(0, 1)
        if metric_key in {"bernoulli", "jaccard"}:
            ax.set_ylim(0, 1)
        elif metric_key == "misidentification":
            if use_log_y:
                positive = env_summary.loc[env_summary["mean"] > 0, "mean"]
                if positive.empty:
                    raise AssertionError(f"{env_name} has no positive misidentification values for log plotting")
                ax.set_yscale("log")
                ax.set_ylim(bottom=float(positive.min()) / 2.0)
                ax.tick_params(axis="y", labelsize=PUBLICATION_FONT_SIZE)
            else:
                ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(FixedLocator(np.linspace(0.0, 1.0, 4)))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:.2f}".rstrip("0").rstrip("."))
        )
        ax.tick_params(axis="both", labelsize=PUBLICATION_FONT_SIZE)
        ax.grid(True, alpha=0.22, linewidth=0.55)

    for j in range(n_env, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=min(6, max(1, len(labels))),
        fontsize=PUBLICATION_FONT_SIZE,
        frameon=True,
    )
    x_label = fig.supxlabel(
        "Normalized Budget",
        y=0.15,
        fontsize=PUBLICATION_FONT_SIZE,
    )
    fig.supylabel(
        METRIC_LABELS[metric_key],
        x=0.115,
        fontsize=PUBLICATION_FONT_SIZE,
    )
    # Reserve a fixed band for the figure-level legend.  This must not be
    # inferred from the axes' tick-label extents: log-scaled axes have taller
    # tick labels and would otherwise push the legend outside the saved figure.
    fig.tight_layout(rect=[0.145, 0.15, 1, 0.855])
    visible_axes = [ax for ax in axes.flat if ax.get_visible()]
    grid_left = min(ax.get_position().x0 for ax in visible_axes)
    grid_right = max(ax.get_position().x1 for ax in visible_axes)
    grid_center = (grid_left + grid_right) / 2.0
    x_label.set_x(grid_center)
    legend.set_bbox_to_anchor((grid_center, 0.89), transform=fig.transFigure)

    plotted_table = pd.concat(plotted_tables, ignore_index=True) if plotted_tables else pd.DataFrame()
    if table_file is not None:
        Path(table_file).parent.mkdir(parents=True, exist_ok=True)
        plotted_table.to_csv(table_file, index=False)
    if save_png and save_file is not None:
        Path(save_file).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_file, format="png", dpi=500, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes, plotted_table

def plot_uncertainty_all_envs(
        env_uncertainties,
        timesteps,
        set_title=True,
        save_png=False,
        save_file=None,
        ylabel="Uncertainty (Bhattacharyya Avg)",
        title="Uncertainty Across Environments",
        ax=None,
):
    """Plot UQ measure evolution over time for multiple environments.

    If *ax* is provided the plot is drawn on that axes (useful for subplots),
    otherwise a new figure is created.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 5))

    handles = []
    labels = []

    for env, uncertainties in env_uncertainties.items():
        mean_uncertainties = np.mean(uncertainties, axis=0)
        std_uncertainties = np.std(uncertainties, axis=0)

        num_label = env.replace("EgeExp", "")

        line, = ax.plot(timesteps, mean_uncertainties, label=env)
        ci = 1.96 * std_uncertainties / np.sqrt(uncertainties.shape[0])
        ax.fill_between(
            timesteps,
            mean_uncertainties - ci,
            mean_uncertainties + ci,
            alpha=0.3,
            color=line.get_color(),
        )

        handles.append(line)
        labels.append(num_label)

    if set_title and not save_png:
        ax.set_title(title, fontsize=14)
    ax.set_xlim(0, max(timesteps))
    ax.set_xlabel("Arm Pulls")
    ax.set_ylabel(ylabel)
    ax.grid(True)

    ax.legend(handles, labels, title="EgeExp", loc="center left",
              bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)

    if standalone:
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        if save_png and save_file is not None:
            plt.savefig(save_file, format="png", dpi=500, bbox_inches="tight")
        plt.show()

    return handles, labels


def plot_uq_comparison(
        env_measures,
        timesteps,
        measure_names=("Bhattacharyya Coeff.", "Symmetric KL-Divergence", "Posterior Entropy"),
        save_png=False,
        save_file=None,
):
    """
    Side-by-side time-evolution plots for multiple UQ measures.

    Parameters
    ----------
    env_measures : dict[str, dict[str, np.ndarray]]
        Outer key = measure name, inner key = environment name,
        value = array of shape (n_runs, n_timesteps).
    timesteps : array-like
    measure_names : sequence of str
    """
    n_measures = len(measure_names)
    fig, axes = plt.subplots(1, n_measures, figsize=(6 * n_measures, 5), squeeze=False)
    axes = axes[0]

    for col, name in enumerate(measure_names):
        ax = axes[col]
        for env, vals in env_measures[name].items():
            mean_v = np.mean(vals, axis=0)
            std_v = np.std(vals, axis=0)
            num_label = env.replace("EgeExp", "")
            line, = ax.plot(timesteps, mean_v, label=num_label)
            ci = 1.96 * std_v / np.sqrt(vals.shape[0])
            ax.fill_between(timesteps, mean_v - ci, mean_v + ci, alpha=0.3, color=line.get_color())
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("Arm Pulls")
        ax.set_xlim(0, max(timesteps))
        ax.grid(True)
        if col == 0:
            ax.set_ylabel("Measure Value")

    # Shared legend from last axes
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title="EgeExp", loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=min(8, len(labels)), fontsize=10, frameon=True)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_png and save_file is not None:
        plt.savefig(save_file, format="png", dpi=500, bbox_inches="tight")
    plt.show()


def plot_uq_correlation_boxplots(
        all_envs_corr_dfs,
        measure_names=("Bhattacharyya Coeff.", "Symmetric KL-Divergence", "Posterior Entropy"),
        environments=None,
        save_png=False,
        save_file=None,
):
    """
    Side-by-side boxplots showing per-experiment correlation between each
    UQ measure and the Jaccard metric, grouped by environment.

    Parameters
    ----------
    all_envs_corr_dfs : dict[str, pd.DataFrame]
        Key = measure name, value = DataFrame with columns 'jaccard', 'environment'.
    """
    import seaborn as sns

    n_measures = len(measure_names)
    fig, axes = plt.subplots(1, n_measures, figsize=(6 * n_measures, 5), squeeze=False)
    axes = axes[0]

    for col, name in enumerate(measure_names):
        ax = axes[col]
        df = all_envs_corr_dfs[name]
        sns.boxplot(
            data=df, x="jaccard", y="environment", hue="environment",
            orient="h", fill=True, ax=ax,
        )
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("Correlation with Jaccard")
        if col > 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Environment")

    plt.tight_layout()
    if save_png and save_file is not None:
        plt.savefig(save_file, format="png", dpi=500, bbox_inches="tight")
    plt.show()


def plot_uq_combined(
        env_measures,
        all_envs_corr_dfs,
        timesteps,
        measure_names=("Bhattacharyya Coeff.", "Symmetric KL-Divergence", "Posterior Entropy"),
        figsize=(16, 8),
        save_png=False,
        save_file=None,
):
    """
    Combined 2×3 figure: top row = time evolution, bottom row = correlation boxplots.
    """
    import seaborn as sns

    n_measures = len(measure_names)
    fig, axes = plt.subplots(2, n_measures, figsize=figsize,
                             gridspec_kw={"height_ratios": [1, 1]})

    # ── Top row: time evolution ──
    for col, name in enumerate(measure_names):
        ax = axes[0, col]
        for env, vals in env_measures[name].items():
            mean_v = np.mean(vals, axis=0)
            std_v = np.std(vals, axis=0)
            num_label = env.replace("EgeExp", "")
            line, = ax.plot(timesteps, mean_v, label=num_label)
            ci = 1.96 * std_v / np.sqrt(vals.shape[0])
            ax.fill_between(timesteps, mean_v - ci, mean_v + ci,
                            alpha=0.3, color=line.get_color())
        ax.set_title(name, fontsize=13)
        ax.set_xlim(0, max(timesteps))
        ax.grid(True)
        if col == 0:
            ax.set_ylabel("Measure Value")
        ax.set_xlabel("")  # suppress x-labels on top row

    # ── Bottom row: correlation boxplots ──
    for col, name in enumerate(measure_names):
        ax = axes[1, col]
        df = all_envs_corr_dfs[name]
        sns.boxplot(
            data=df, x="jaccard", y="environment", hue="environment",
            orient="h", fill=True, ax=ax,
        )
        ax.set_xlabel("Correlation with Jaccard")
        if col > 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("Environment")

    # Shared legend from the top-right axes
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, title="EgeExp", loc="upper center",
               bbox_to_anchor=(0.5, 1.01), ncol=len(labels),
               fontsize=10, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_png and save_file is not None:
        plt.savefig(save_file, format="png", dpi=500, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_all_pfi_metrics("results/EGEvsTTPFTSvsPUCB1vsUniform_EgeExp1.csv", 100, 5001)
