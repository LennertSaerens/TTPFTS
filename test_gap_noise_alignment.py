#!/usr/bin/env python3
"""Regression gate for the paired custom gap/noise-alignment experiment."""

from __future__ import annotations

import csv
import tempfile
import traceback
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.image import imread

matplotlib.use("Agg")

from bandits.ege_kone import EGE_SH, EGE_SR
from experiments.gap_noise_alignment.config import ALGORITHMS, CONDITIONS, load_config
from experiments.gap_noise_alignment.instances import PairedGaussianEnvironment, load_conditions
from experiments.gap_noise_alignment.merge import merge, write_manifest
from experiments.gap_noise_alignment.plot import render_environment, render_performance
from experiments.gap_noise_alignment.run import (
    anytime_paths,
    run,
    run_anytime_task,
    run_ege_budget_task,
)
from experiments.gap_noise_alignment.validate import validate
from runner.seeding import seed_rngs


FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def execute(name: str, fn) -> None:
    print(f"\n--- {name} ---")
    try:
        fn()
    except Exception:
        FAILURES.append(name)
        print(f"[FAIL] {name} raised:")
        traceback.print_exc()


def test_frozen_contract_and_paired_latents() -> None:
    config, conditions = load_config(), load_conditions()
    large, small = conditions
    check("approved five-algorithm roster is frozen", tuple(config["algorithms"]) == ALGORITHMS)
    check("exactly two conditions are frozen", tuple(item.condition for item in conditions) == CONDITIONS)
    check("condition means are byte-identical", np.array_equal(large.means, small.means))
    check("Pareto set is exactly five anchors", large.pareto_indices.tolist() == [0, 1, 2, 3, 4])
    check("both conditions retain H=14290.552976858806", np.isclose(large.hardness.hardness, 14290.552976858806) and large.hardness.hardness == small.hardness.hardness)
    check("both conditions retain B=14291 and 21 checkpoints", large.budget == small.budget == 14291 and len(large.grid) == 21)
    check("variance multisets match", np.array_equal(np.sort(large.arm_stds), np.sort(small.arm_stds)))
    rank = np.argsort(large.noise_ranks)
    check("standard deviations reverse exactly by true-gap rank", np.array_equal(large.arm_stds[rank], small.arm_stds[rank][::-1]))
    environment_large = PairedGaussianEnvironment(large.means, large.stds, reward_seed=123)
    environment_small = PairedGaussianEnvironment(small.means, small.stds, reward_seed=123)
    arms = [0, 12, 0, 49, 12, 8]
    rewards_large = np.vstack([environment_large.pull_arm(arm) for arm in arms])
    rewards_small = np.vstack([environment_small.pull_arm(arm) for arm in arms])
    latent_large = (rewards_large - large.means[arms]) / large.stds[arms]
    latent_small = (rewards_small - small.means[arms]) / small.stds[arms]
    check("condition pairing shares arm/objective latent streams", np.allclose(latent_large, latent_small, rtol=0.0, atol=1e-14))


def test_ege_trace_is_passive() -> None:
    item, budget = load_conditions()[0], load_conditions()[0].budget
    for name, implementation in (("EGE-SH", EGE_SH), ("EGE-SR", EGE_SR)):
        first = PairedGaussianEnvironment(item.means, item.stds, reward_seed=777)
        seed_rngs(777, name)
        baseline = implementation(budget, item.num_arms, item.num_objectives, first)
        second = PairedGaussianEnvironment(item.means, item.stds, reward_seed=777)
        seed_rngs(777, name)
        trace = []
        traced = implementation(budget, item.num_arms, item.num_objectives, second, trace=trace)
        check(f"{name} trace preserves recommendation and draws", baseline == traced and first.reward_draws == second.reward_draws)
        check(f"{name} trace classifies every arm once", len(trace) == item.num_arms and {event["arm"] for event in trace} == set(range(item.num_arms)))


def _run_pair_smoke(root: Path) -> tuple[Path, Path]:
    config, conditions = load_config(), load_conditions()
    run_dir = root / "run"
    for item in conditions:
        for algorithm in ("Uniform", "APE", "TTPFTS_UNKNOWN_VAR"):
            run_anytime_task(item, 0, algorithm, run_dir, config)
        for algorithm in ("EGE_SH", "EGE_SR"):
            for budget in item.grid:
                run_ege_budget_task(item, 0, algorithm, budget, run_dir, config)
    result, trace = merge(run_dir, root / "results", conditions, [0], config)
    write_manifest(run_dir, root / "results", result, trace, conditions, [0], config, require_clean_source=False)
    return run_dir, root / "results"


def test_resume_merge_validation_and_corruption_rejection() -> None:
    config, item = load_config(), load_conditions()[0]
    with tempfile.TemporaryDirectory(prefix="mots-gap-noise-") as temporary:
        root = Path(temporary)
        baseline = run_anytime_task(item, 3, "Uniform", root / "baseline", config)
        try:
            run_anytime_task(item, 3, "Uniform", root / "resumed", config, stop_after_checkpoint=item.grid[1])
        except RuntimeError as exc:
            check("controlled checkpoint interruption is explicit", str(exc) == "controlled checkpoint interruption")
        resumed = run_anytime_task(item, 3, "Uniform", root / "resumed", config)
        check("resumed trajectory is byte-identical", baseline.read_bytes() == resumed.read_bytes())
        run_dir, results_dir = _run_pair_smoke(root / "paired")
        problems = validate(run_dir, results_dir, conditions=load_conditions(), repetitions=[0], require_manifest=True)
        check("two-condition smoke output passes independent validation", not problems, "\n".join(problems[:3]))
        result_path = results_dir / "gap_noise_results.csv"
        with result_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["jaccard"] = "1.0"
        with result_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        problems = validate(run_dir, results_dir, conditions=load_conditions(), repetitions=[0], require_manifest=False)
        check("metric-corrupted merged row is rejected", any("jaccard" in problem for problem in problems), "\n".join(problems[:3]))


def test_serial_parallel_task_invariance() -> None:
    config, item = load_config(), load_conditions()[0]
    with tempfile.TemporaryDirectory(prefix="mots-gap-noise-serial-") as serial, tempfile.TemporaryDirectory(
        prefix="mots-gap-noise-parallel-"
    ) as parallel:
        serial_root, parallel_root = Path(serial), Path(parallel)
        run([item], serial_root, config, n_jobs=1, repetitions=[0])
        run([item], parallel_root, config, n_jobs=2, repetitions=[0])
        serial_files = sorted(
            (path.relative_to(serial_root / "shards"), path.read_bytes())
            for path in (serial_root / "shards").rglob("*.csv")
        )
        parallel_files = sorted(
            (path.relative_to(parallel_root / "shards"), path.read_bytes())
            for path in (parallel_root / "shards").rglob("*.csv")
        )
        check("serial and parallel condition shards are byte-identical", serial_files == parallel_files)


def test_environment_and_performance_plot_contract() -> None:
    with tempfile.TemporaryDirectory(prefix="mots-gap-noise-plot-") as temporary:
        root = Path(temporary)
        for item in load_conditions():
            output, _ = render_environment(item, root)
            table = pd.read_csv(root / f"{item.condition}_environment_plotted_values.csv")
            check(f"{item.condition} environment figure renders", output.is_file() and output.stat().st_size > 0)
            check(f"{item.condition} environment plot retains all 50 arms", len(table) == 50)
        records = []
        for condition in CONDITIONS:
            for algorithm_index, algorithm in enumerate(ALGORITHMS):
                for grid_index in range(21):
                    normalized = grid_index / 20
                    mean = min(1.0, 0.1 + normalized * (0.5 + algorithm_index / 20))
                    records.append(
                        {
                            "condition": condition, "algorithm": algorithm, "grid_index": grid_index,
                            "normalized_budget": normalized, "metric": "jaccard", "mean": mean,
                            "lower": max(0.0, mean - 0.03), "upper": min(1.0, mean + 0.03),
                            "n_repetitions": 100, "bootstrap_replicates": 2000,
                        }
                    )
        summary_path = root / "jaccard_condition_summary.csv"
        pd.DataFrame(records).to_csv(summary_path, index=False)
        output, _ = render_performance("large_gap_noisy", "jaccard", summary_path, root)
        plotted = pd.read_csv(root / "large_gap_noisy_jaccard_plotted_values.csv")
        image = imread(output)
        check("performance figure renders", output.is_file() and output.stat().st_size > 0)
        check("performance figure retains exact 105 summary points", len(plotted) == 105)
        check("performance figure uses double-column aspect", image.shape[1] / image.shape[0] > 1.25)


if __name__ == "__main__":
    execute("Frozen custom conditions and paired rewards", test_frozen_contract_and_paired_latents)
    execute("Passive EGE classification traces", test_ege_trace_is_passive)
    execute("Resume, merge, validation, and corruption rejection", test_resume_merge_validation_and_corruption_rejection)
    execute("Serial/parallel task invariance", test_serial_parallel_task_invariance)
    execute("Environment and performance plotting contract", test_environment_and_performance_plot_contract)
    print("\n" + "=" * 80)
    if FAILURES:
        print("FAILED:", ", ".join(FAILURES))
        raise SystemExit(1)
    print("ALL TESTS PASSED")
