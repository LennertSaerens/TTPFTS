"""Standalone regression tests for publication plotting.

Run with:

    PYTHONPATH=. ./.venv/bin/python test_plotting_regression.py
"""

import json
import math
import sys
import traceback
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from plotting import (
    DEFAULT_ALGORITHM_ORDER,
    PUBLICATION_FONT_SIZE,
    load_legacy_results,
    plot_metric_all_envs_grid_top_legend,
    summarize_metric_for_plot,
    wilson_interval,
)

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def run(name, fn):
    print(f"\n--- {name} ---")
    try:
        fn()
    except Exception:
        FAILURES.append(name)
        print(f"[FAIL] {name} raised an exception:")
        traceback.print_exc()


def _write_synthetic_dataset():
    base = Path(".copilot/plotting_regression")
    base.mkdir(parents=True, exist_ok=True)
    environments = [f"EgeExp{i}" for i in range(1, 9)]
    budgets = {
        "EgeExp1": 13,
        "EgeExp2": 17,
        "EgeExp3": 19,
        "EgeExp4": 23,
        "EgeExp5": 29,
        "EgeExp6": 31,
        "EgeExp7": 37,
        "EgeExp8": 41,
    }
    algs = list(DEFAULT_ALGORITHM_ORDER)
    files = []
    for env_idx, env in enumerate(environments):
        budget = budgets[env]
        rows = []
        for alg_idx, alg in enumerate(algs):
            if alg.startswith("EGE"):
                grid = [0, budget // 2, budget]
            else:
                grid = sorted({0, budget // 3, 2 * budget // 3, budget})
            for rep in range(3):
                for t in grid:
                    success = int((rep + alg_idx + env_idx + t) % 3 == 0)
                    rows.append({
                        "environment": env,
                        "instance_hash": f"hash-{env}",
                        "algorithm": alg,
                        "posterior_model": "known" if alg.endswith("KNOWN_VAR") else "",
                        "repetition": rep,
                        "arm_pull": t,
                        "max_budget": budget,
                        "evaluation_grid": json.dumps(grid),
                        "rho": 0.5,
                        "master_seed": 123,
                        "bernoulli": success,
                        "jaccard": min(1.0, (t / budget) * (0.6 + 0.05 * rep)),
                        "misidentification": max(0.0, 1.0 - t / budget + 0.01 * alg_idx),
                        "recommendation": "[]",
                        "true_pareto": "[]",
                    })
        path = base / f"{env}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        files.append(str(path))
    manifest = {
        "master_seed": 123,
        "dist": "gaussian",
        "environments": environments,
        "algorithms": algs,
        "num_runs": 3,
        "step": 1,
        "explicit_budgets": None,
        "eval_points": None,
        "budgets": budgets,
        "evaluation_grids": {env: [0, budgets[env] // 2, budgets[env]] for env in environments},
        "n_jobs": 1,
        "rho": 0.5,
        "results_dir": str(base),
        "run_dir": str(base / "run"),
    }
    run_dir = base / "run"
    run_dir.mkdir(exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return environments, files, budgets, str(manifest_path)


def _make_plot(metric="Jaccard"):
    envs, files, budgets, manifest_path = _write_synthetic_dataset()
    fig, axes, table = plot_metric_all_envs_grid_top_legend(
        env_files=files,
        env_labels=envs,
        env_arms=[1] * 8,
        env_objs=[2] * 8,
        metric=metric,
        manifest_path=manifest_path,
        figsize=(12, 6),
        exclude_algs={"TTPFTS_KNOWN_VAR"},
        algorithm_display_labels={
            "EGE_SH": "EGE-SH",
            "EGE_SR": "EGE-SR",
            "TTPFTS_UNKNOWN_VAR": "TTPFTS",
        },
        log_misidentification=True,
        show=False,
    )
    return fig, axes, table, envs, budgets


def test_normalization_endpoint_and_unequal_budgets():
    fig, axes, table, envs, budgets = _make_plot()
    endpoints = table.groupby(["environment", "algorithm"])["normalized_budget"].max()
    check("every curve endpoint is exactly normalized budget 1.0", bool((endpoints == 1.0).all()), str(endpoints))
    plotted_budgets = table.groupby("environment")["max_budget"].first().to_dict()
    check("subplots preserve unequal environment budgets", len(set(plotted_budgets.values())) == 8,
          str(plotted_budgets))
    plt.close(fig)


def test_shared_labels_and_legend_entries():
    fig, axes, table, envs, budgets = _make_plot("Jaccard")
    x_texts = [text for text in fig.texts if text.get_text() == "Normalized Budget"]
    y_texts = [text for text in fig.texts if text.get_text() == "Jaccard Metric"]
    check("exactly one shared x-label", len(x_texts) == 1, f"found {len(x_texts)}")
    check("exactly one shared y-label", len(y_texts) == 1, f"found {len(y_texts)}")
    check("no repeated per-axes x-labels", all(ax.get_xlabel() == "" for ax in axes.flat))
    check("no repeated per-axes y-labels", all(ax.get_ylabel() == "" for ax in axes.flat))
    legend = fig.legends[0] if fig.legends else None
    labels = [t.get_text() for t in legend.get_texts()] if legend else []
    check("legend has five algorithm entries", len(labels) == 5, str(labels))
    check("legend omits known-variance TTPFTS", "TTPFTS_KNOWN_VAR" not in labels, str(labels))
    check("legend relabels unknown-variance TTPFTS", "TTPFTS" in labels, str(labels))
    check("legend labels are distinct", len(set(labels)) == 5, str(labels))
    annotations = [text.get_text() for ax in axes.flat for text in ax.texts]
    check("subplots have compact environment labels", annotations == [str(i) for i in range(1, 9)],
          str(annotations))
    grid_center = (axes[0, 0].get_position().x0 + axes[0, -1].get_position().x1) / 2.0
    x_label = next(text for text in fig.texts if text.get_text() == "Normalized Budget")
    legend_center = fig.legends[0].get_bbox_to_anchor().transformed(fig.transFigure.inverted()).x0
    check("shared x-label centers on subplot grid", abs(x_label.get_position()[0] - grid_center) < 1e-9)
    check("legend centers on subplot grid", abs(legend_center - grid_center) < 1e-9)
    plt.close(fig)


def test_sparse_ege_grids_not_interpolated():
    fig, axes, table, envs, budgets = _make_plot("Bernoulli")
    ege_curve = table[(table["environment"] == "EgeExp1") & (table["algorithm"] == "EGE_SR")]
    uniform_curve = table[(table["environment"] == "EgeExp1") & (table["algorithm"] == "Uniform")]
    check("sparse EGE grid keeps only actually recorded budgets", len(ege_curve) == 3,
          ege_curve[["arm_pull", "normalized_budget"]].to_string(index=False))
    check("non-EGE synthetic grid can have different recorded count", len(uniform_curve) == 4,
          uniform_curve[["arm_pull", "normalized_budget"]].to_string(index=False))
    plt.close(fig)


def test_misidentification_log_scale_and_compact_x_ticks():
    fig, axes, _, _, _ = _make_plot("Misidentification")
    check("misidentification axes use log scale", all(ax.get_yscale() == "log" for ax in axes.flat))
    tick_labels = [label.get_text() for label in axes[-1, 0].get_xticklabels() if label.get_text()]
    check("x ticks do not have redundant trailing zeros", all(not text.endswith(".00") for text in tick_labels),
          str(tick_labels))
    check("each subplot has four x ticks", all(len(ax.get_xticks()) == 4 for ax in axes.flat))
    check("x ticks use compact one-third labels", tick_labels == ["0", "0.33", "0.67", "1"],
          str(tick_labels))
    fig.canvas.draw()
    y_tick_sizes = [
        label.get_fontsize()
        for ax in axes.flat
        for label in ax.get_yticklabels()
        if label.get_visible()
    ]
    check(
        "misidentification y tick labels use publication font size",
        all(size == PUBLICATION_FONT_SIZE for size in y_tick_sizes),
          str(y_tick_sizes))
    plt.close(fig)


def test_publication_font_sizes_match_legend():
    fig, axes, _, _, _ = _make_plot("Jaccard")
    fig.canvas.draw()
    legend_sizes = [text.get_fontsize() for text in fig.legends[0].get_texts()]
    axis_sizes = [
        label.get_fontsize()
        for ax in axes.flat
        for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]
        if label.get_visible()
    ]
    annotation_sizes = [text.get_fontsize() for ax in axes.flat for text in ax.texts]
    shared_sizes = [
        text.get_fontsize()
        for text in fig.texts
        if text.get_text() in {"Normalized Budget", "Jaccard Metric"}
    ]
    check("legend uses publication font size", all(size == PUBLICATION_FONT_SIZE for size in legend_sizes))
    check("axis ticks use publication font size", all(size == PUBLICATION_FONT_SIZE for size in axis_sizes))
    check("environment annotations use publication font size",
          all(size == PUBLICATION_FONT_SIZE for size in annotation_sizes))
    check("shared labels use publication font size", all(size == PUBLICATION_FONT_SIZE for size in shared_sizes))
    plt.close(fig)


def test_legacy_archive_loading_still_works():
    path = Path("archive_prepatch/results/EgeExp1_main_combined.csv")
    if not path.exists():
        check("legacy archive fixture exists", False, str(path))
        return
    legacy = load_legacy_results(path)
    required = {"environment", "algorithm", "repetition", "arm_pull", "max_budget", "bernoulli", "jaccard", "misidentification"}
    check("legacy archive loads to canonical plotting columns", required.issubset(legacy.columns), str(legacy.columns))
    check("legacy archive infers environment", legacy["environment"].iloc[0] == "EgeExp1")
    check("legacy archive default budget remains explicit legacy-only value", int(legacy["max_budget"].iloc[0]) == 5000)


def test_wilson_interval_correctness():
    successes, n = 7, 11
    got = wilson_interval(successes, n, confidence=0.95)
    z = 1.959963984540054
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    expected = (max(0.0, center - half), min(1.0, center + half))
    check("Wilson interval matches independent closed-form computation",
          abs(got[0] - expected[0]) < 1e-12 and abs(got[1] - expected[1]) < 1e-12,
          f"got={got}, expected={expected}")


def test_summary_direct_endpoint_assertion():
    _, files, budgets, _ = _write_synthetic_dataset()
    df = pd.read_csv(files[0])
    summary = summarize_metric_for_plot(df, "jaccard", {"EgeExp1": budgets["EgeExp1"]})
    check("direct summary endpoint assertion allows exact manifest endpoint",
          bool((summary.groupby("algorithm")["normalized_budget"].max() == 1.0).all()))


if __name__ == "__main__":
    run("x-normalization endpoint and unequal budgets", test_normalization_endpoint_and_unequal_budgets)
    run("shared labels and six-entry legend", test_shared_labels_and_legend_entries)
    run("sparse EGE grids", test_sparse_ege_grids_not_interpolated)
    run("misidentification log scale and x ticks", test_misidentification_log_scale_and_compact_x_ticks)
    run("publication font-size consistency", test_publication_font_sizes_match_legend)
    run("legacy archive loading", test_legacy_archive_loading_still_works)
    run("Wilson interval correctness", test_wilson_interval_correctness)
    run("direct summary endpoint assertion", test_summary_direct_endpoint_assertion)

    print("\n" + "=" * 80)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("ALL TESTS PASSED")
    sys.exit(0)
