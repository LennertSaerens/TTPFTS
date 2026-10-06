#!/usr/bin/env python3
"""Regression gate for the information-matched figures (APE-UV, empirical-mean D-EGE)."""

from __future__ import annotations

import csv
import shutil
import tempfile
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from experiments.information_matched import build as build_module
from experiments.information_matched import plot as plot_module
from experiments.information_matched import run_ape_uv, run_empirical, validate as validate_module
from experiments.information_matched.build import plot_input_path
from experiments.information_matched.config import (
    REPO_ROOT,
    assert_safe_output,
    default_output_root,
    load_config,
    verify_protected,
    verify_sources,
)
from experiments.information_matched.run_empirical import (
    empirical_trajectory,
    round_robin,
    stage_plan,
)
from plotting import plot_metric_all_envs_grid_top_legend

FAILURES: list[str] = []
ENVIRONMENTS = ["EgeExp2", "EgeExp7"]
REPETITIONS = [0, 1, 2]
SAMPLES = {"ape_uv_rerun": {"environments": ["EgeExp2"], "repetitions": [1]},
           "empirical_recompute": {"repetitions": [2]}}


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def execute(name: str, fn) -> None:
    print(f"\n=== {name} ===")
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        check(f"{name} raised no exception", False, f"{type(error).__name__}: {error}")


def raises(fn, exception) -> bool:
    try:
        fn()
    except exception:
        return True
    return False


def test_protection_and_units() -> None:
    config = load_config()
    check("output root is config-hash namespaced", default_output_root(config).name.startswith("results_information_matched_"))
    for protected in ("figures/publication/publication_jaccard.png",
                      "figures/publication_ege_doubling/publication_jaccard_ege_doubling.png",
                      "results_preregistered_final_ff29cc456351/EgeExp1.csv",
                      "results_ege_doubling_3840c14dc78c/doubling/EgeExp1.csv"):
        check(f"refuses to write {protected}",
              raises(lambda p=protected: assert_safe_output(REPO_ROOT / p, config), PermissionError))
    check("allows the new figure directory",
          assert_safe_output(REPO_ROOT / config["figures_dir"] / "x.png", config) is not None)
    verify_sources(config)
    verify_protected(config)
    check("reused sources and protected figures match their recorded hashes", True)

    arms = np.repeat(np.array([4, 1, 7]), 3)
    rewards = np.arange(9, dtype=float)[:, None] * np.ones((1, 2))
    rr_arms, rr_rewards = round_robin(arms, rewards)
    check("round-robin interleaves equal-allocation blocks",
          rr_arms.tolist() == [4, 1, 7] * 3 and rr_rewards[:, 0].tolist() == [0, 3, 6, 1, 4, 7, 2, 5, 8])
    check("round-robin rejects unequal blocks",
          raises(lambda: round_robin(np.array([1, 1, 2]), np.zeros((3, 2))), ValueError))
    plan = stage_plan(22, 7000)
    check("EgeExp7 stage clock: eight complete stages and an incomplete ninth filling (5610, 7000]",
          [p["end"] for p in plan[:-1]] == [22, 66, 154, 330, 682, 1386, 2794, 5610]
          and plan[-1] == {"stage": 9, "budget": 5632, "start": 5610, "end": 7000, "completed": False})
    times, arms = np.array([1, 2, 3, 5]), np.array([0, 1, 0, 2])
    rewards = np.array([[0.2, 0.9], [0.8, 0.1], [0.4, 0.7], [0.0, 0.0]])
    out = empirical_trajectory(times, arms, rewards, [0, 2, 4, 5], 3)
    check("empirical recommendation: all arms at t=0, zero-initialized unsampled arms, all samples so far",
          [o[1].tolist() for o in out] == [[0, 1, 2], [0, 1], [0, 1], [0, 1]]
          and [o[2] for o in out] == [0, 2, 3, 4] and [o[3] for o in out] == [3, 1, 1, 0])


def test_renderer_styles() -> None:
    config = load_config()
    with tempfile.TemporaryDirectory() as directory:
        import pandas as pd
        rows = []
        for algorithm in config["figure_sets"]["uv_both"]:
            for repetition in range(3):
                for arm_pull in (0, 50, 100):
                    rows.append({"environment": "EgeExp2", "algorithm": algorithm, "posterior_model": "",
                                 "repetition": repetition, "arm_pull": arm_pull, "max_budget": 100,
                                 "bernoulli": 0, "jaccard": 0.5, "misidentification": 0.1})
        path = Path(directory) / "EgeExp2.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        fig, axes, _ = plot_metric_all_envs_grid_top_legend(
            [str(path)], ["EgeExp2"], [10], [2], metric="Jaccard", environment_budgets={"EgeExp2": 100},
            algorithm_order=config["figure_sets"]["uv_both"], algorithm_display_labels=config["labels"],
            algorithm_styles=config["styles"], legend_ncol=7, legend_kwargs=config["plot"]["legend_kwargs"]["uv_both"],
            show=False)
        lines = {line.get_label(): line for line in axes[0, 0].get_lines()}
        inverse = {label: key for key, label in config["labels"].items()}
        ok = all(
            mcolors.to_hex(line.get_color()) == mcolors.to_hex(config["styles"][inverse[label]]["color"])
            and line.get_linestyle() == config["styles"][inverse[label]]["linestyle"]
            for label, line in lines.items()
        )
        check("renderer applies fixed colors and dashed naive curves", ok and len(lines) == 7)
        check("seven-entry legend uses a single row", fig.legends[0]._ncols == 7)
        plt.close(fig)


def test_stage_only_memory() -> None:
    times = np.array([1, 2, 3, 4])
    arms = np.array([0, 1, 0, 1])
    rewards = np.array([[10.0, 10.0], [1.0, 1.0], [0.0, 0.0], [2.0, 2.0]])
    points = [0, 2, 3, 4, 5]
    pooled = empirical_trajectory(times, arms, rewards, points, 3)
    local = empirical_trajectory(times, arms, rewards, points, 3, reset_after=[2, 4])
    check("no-pool endpoint retains its just-completed stage",
          local[1][1].tolist() == [0] and local[3][1].tolist() == [1])
    check("no-pool discards previous-stage means at the next pull",
          pooled[3][1].tolist() == [0] and local[3][1].tolist() == [1])
    check("no-pool resets even when the next stage has no observations yet",
          local[4][1].tolist() == [0, 1, 2] and local[4][3] == 3)
    check("no-pool preserves cumulative acquisition counts",
          [row[2] for row in pooled] == [row[2] for row in local])
    check("no-pool resets correctly when evaluation skips multiple stages",
          empirical_trajectory(times, arms, rewards, [4], 3, reset_after=[2, 4])[0][1].tolist() == [1])

    from experiments.information_matched import no_pool

    config = no_pool.load_no_pool_config()
    check("no-pool workflow protects the old information-matched results",
          raises(lambda: assert_safe_output(no_pool.source_root(config) / "empirical/EgeExp2.csv", config),
                 PermissionError))
    check("no-pool workflow protects the old information-matched figures",
          raises(lambda: assert_safe_output(REPO_ROOT / "figures/publication_information_matched/x.png", config),
                 PermissionError))
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        no_pool.run_empirical.run(
            root, config=config, environments=["EgeExp2"], repetitions=[0, 1], n_jobs=1, allow_dirty=True
        )
        no_pool.build_inputs(root, config, ["EgeExp2"])
        report = no_pool.validate(root, config, ["EgeExp2"], [0, 1], require_clean=False)
        check("no-pool runner matches independent stage-reset replay in both pull orders",
              report["summaries"]["EgeExp2"]["empirical"]["independently_recomputed_points"] > 0)


def _run(root: Path, n_jobs: int, environments=ENVIRONMENTS) -> None:
    config = load_config()
    run_ape_uv.run(root, config, environments=environments, repetitions=REPETITIONS, n_jobs=n_jobs, allow_dirty=True)
    run_empirical.run(root, config, environments=environments, repetitions=REPETITIONS, n_jobs=n_jobs, allow_dirty=True)


def _validate(root: Path, **kwargs) -> dict:
    return validate_module.validate(root, load_config(), environments=ENVIRONMENTS, repetitions=REPETITIONS,
                                    require_clean=False, samples=SAMPLES, **kwargs)


def _rewrite(path: Path, mutate) -> bytes:
    original = path.read_bytes()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header, rows = list(reader.fieldnames), list(reader)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(mutate(rows))
    return original


def test_pipeline() -> None:
    config = load_config()
    with tempfile.TemporaryDirectory() as directory:
        root, serial, figures = Path(directory) / "root", Path(directory) / "serial", Path(directory) / "figs"
        _run(root, n_jobs=2)
        shards = sorted((root).rglob("*.csv"))
        stamps = {p: p.stat().st_mtime_ns for p in shards if "shards" in p.parts}
        _run(root, n_jobs=2)
        check("resume recomputes no completed task", all(p.stat().st_mtime_ns == s for p, s in stamps.items()))
        _run(serial, n_jobs=1, environments=["EgeExp7"])
        identical = all(
            (serial / sub / "EgeExp7.csv").read_bytes() == (root / sub / "EgeExp7.csv").read_bytes()
            for sub in ("ape_uv", "empirical")
        )
        check("serial and parallel runs are byte-identical", identical)
        build_module.build(root, config, environments=ENVIRONMENTS)
        check("render is refused before validation",
              raises(lambda: plot_module.render(root, config, environments=ENVIRONMENTS, figures_dir=figures,
                                                allow_dirty=True, verify_style=False), RuntimeError))
        report = _validate(root)
        check("validator passes clean smoke data", report["status"] == "pass", "; ".join(report["problems"][:5]))
        summary = report["summaries"]["EgeExp7"]["empirical"]
        check("every completed empirical stage reproduces the stored naive stage run",
              summary["reproduced_naive_stages"] == 2 * len(REPETITIONS) * 8)
        check("sampled trajectories were independently recomputed in both pull orders",
              summary["independently_recomputed_points"] > 0
              and report["summaries"]["EgeExp2"]["ape_uv"]["reran_rows"] == 101)

        def semantic(fragment: str) -> bool:
            result = _validate(root, write_report=False)
            return result["status"] == "fail" and any(fragment in p for p in result["problems"])

        stages = root / "empirical" / "EgeExp7_stages.csv"
        original = _rewrite(stages, lambda rows: [dict(r, recommendation="[0]" if r["recommendation"] != "[0]" else "[1]") if i == 3 else r
                                     for i, r in enumerate(rows)])
        check("validator rejects a stage that does not reproduce the naive run", semantic("does not reproduce the naive"))
        stages.write_bytes(original)
        trajectory = root / "empirical" / "EgeExp7.csv"
        target = next(i for i, r in enumerate(csv.DictReader(trajectory.open())) if r["repetition"] == "2" and int(r["arm_pull"]) > 1000)
        original = _rewrite(trajectory, lambda rows: [dict(r, recommendation="[0,1]" if r["recommendation"] != "[0,1]" else "[0,2]", bernoulli="0", jaccard="0.25",
                                                           misidentification=str(6 / 22)) if i == target else r
                                                      for i, r in enumerate(rows)])
        check("validator rejects a trajectory point that differs from the pull-by-pull recomputation",
              semantic("independent recomputation differs"))
        trajectory.write_bytes(original)
        ape = root / "ape_uv" / "EgeExp2.csv"
        target = next(i for i, r in enumerate(csv.DictReader(ape.open())) if r["repetition"] == "1" and r["arm_pull"] == "2544")
        original = _rewrite(ape, lambda rows: [dict(r, recommendation="[0,1,2]" if r["recommendation"] != "[0,1,2]" else "[0,1,3]",
                                                    bernoulli="0", jaccard=str(2 / 3), misidentification="0.1")
                                               if i == target else r
                                               for i, r in enumerate(rows)])
        check("validator rejects an APE-UV row that a bitwise re-run does not reproduce", semantic("re-run differs"))
        ape.write_bytes(original)
        inputs = plot_input_path(root, "EgeExp2")
        original = _rewrite(inputs, lambda rows: [dict(r, jaccard="0.123") if i == 5 else r for i, r in enumerate(rows)])
        check("validator rejects plot inputs that are not verbatim source rows", semantic("not verbatim"))
        inputs.write_bytes(original)
        check("validator passes again after restoring every file", _validate(root)["status"] == "pass")

        plot_module.render(root, config, environments=ENVIRONMENTS, figures_dir=figures, allow_dirty=True, verify_style=False)
        report = _validate(root, figures_dir=figures, check_figures=True)
        check("figures: reused curves equal existing publication tables and new curves equal recomputed means",
              report["status"] == "pass", "; ".join(report["problems"][:5]))
        check("twelve figures and plot tables were rendered", len(list(figures.glob("*.png"))) == 12
              and len(list(figures.glob("*_plot_table.csv"))) == 12)
        check("existing figures are never overwritten implicitly",
              raises(lambda: plot_module.render(root, config, environments=ENVIRONMENTS, figures_dir=figures,
                                                allow_dirty=True, verify_style=False), FileExistsError))
        verify_sources(config)
        verify_protected(config)
        check("reused sources and protected figures are unchanged after the gate", True)
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    execute("Protection, round-robin ordering, stage clock, and empirical recommendation", test_protection_and_units)
    execute("Renderer styles and legend", test_renderer_styles)
    execute("Stage-only sample memory", test_stage_only_memory)
    execute("Run, resume, serial/parallel identity, validation, corruption, and plotting", test_pipeline)
    print("\n" + "=" * 80)
    if FAILURES:
        print("FAILED:", ", ".join(FAILURES))
        raise SystemExit(1)
    print("ALL TESTS PASSED")
