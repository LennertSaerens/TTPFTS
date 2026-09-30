"""Standalone regression tests for the hardness-normalized TTPFTS p sweep."""

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

from experiments.p_sweep.analyze import endpoint_auc_summary, metric_summary
from experiments.p_sweep.config import config_hash, environment_config, load_config
from experiments.p_sweep.merge import merge_environment
from experiments.p_sweep.plot import render
from experiments.p_sweep.run import run, run_repetition, shard_is_complete, shard_path
from experiments.p_sweep.validate import validate


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


def test_config_contract() -> None:
    config = load_config()
    details = environment_config("EgeExp3", config)
    check("sweep config fixes 11 p values", config["p_values"] == [round(index / 10, 1) for index in range(11)])
    check("sweep grid contains budget zero and exact endpoint", details["grid"][0] == 0 and details["grid"][-1] == details["budget"])
    check("sweep uses exactly 21 checkpoints", len(details["grid"]) == 21)


def test_p_half_identity_and_config_bound_resume() -> None:
    config = load_config()
    with tempfile.TemporaryDirectory(prefix="mots-p-sweep-test-") as temporary:
        root = Path(temporary)
        run_repetition("EgeExp2", 0.5, 0, root / "run", config)
        merged = merge_environment(root / "run", root / "results", "EgeExp2", [0.5], [0], config)
        problems = validate(
            root / "run",
            root / "results",
            Path("results_preregistered_final_ff29cc456351"),
            environments=["EgeExp2"],
            p_values=[0.5],
            repetitions=[0],
        )
        check("p=0.5 sparse trajectory matches validated final result", not problems, "\n".join(problems[:3]))
        check("p=0.5 merges exactly 21 rows", merged == 21, str(merged))
        path = shard_path(root / "run", "EgeExp2", 0.5, 0)
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[1]["jaccard"] = "1.0"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        merge_environment(root / "run", root / "results", "EgeExp2", [0.5], [0], config)
        corrupted_problems = validate(
            root / "run",
            root / "results",
            Path("results_preregistered_final_ff29cc456351"),
            environments=["EgeExp2"],
            p_values=[0.5],
            repetitions=[0],
        )
        check(
            "metric-corrupted shard is rejected",
            any("jaccard" in problem for problem in corrupted_problems),
            "\n".join(corrupted_problems[:3]),
        )
        rows[0]["sweep_config_hash"] = "corrupted"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        check(
            "config-corrupted shard is rejected for resume",
            not shard_is_complete(
                root / "run",
                "EgeExp2",
                0.5,
                0,
                environment_config("EgeExp2", config)["grid"],
                config_hash(config),
            ),
        )


def test_serial_parallel_invariance() -> None:
    config = load_config()
    with tempfile.TemporaryDirectory(prefix="mots-p-sweep-serial-") as serial_tmp, tempfile.TemporaryDirectory(
        prefix="mots-p-sweep-parallel-"
    ) as parallel_tmp:
        serial_root, parallel_root = Path(serial_tmp), Path(parallel_tmp)
        run(["EgeExp2"], [0.0], [0, 1], serial_root, config, n_jobs=1)
        run(["EgeExp2"], [0.0], [0, 1], parallel_root, config, n_jobs=2)
        serial = sorted(path.read_text() for path in (serial_root / "shards" / "EgeExp2" / "p0.0").glob("*.csv"))
        parallel = sorted(path.read_text() for path in (parallel_root / "shards" / "EgeExp2" / "p0.0").glob("*.csv"))
        check("serial and parallel sweep shards are byte-identical", serial == parallel)


def test_plot_uses_exact_summary_table() -> None:
    with tempfile.TemporaryDirectory(prefix="mots-p-sweep-plot-") as temporary:
        root = Path(temporary)
        records = []
        for environment_index in range(1, 9):
            for p in np.linspace(0.0, 1.0, 11):
                for normalized in np.linspace(0.0, 1.0, 21):
                    mean = min(1.0, 0.2 + normalized * (0.4 + p / 4))
                    records.append(
                        {
                            "environment": f"EgeExp{environment_index}",
                            "p": p,
                            "normalized_budget": normalized,
                            "mean": mean,
                            "lower": max(0.0, mean - 0.02),
                            "upper": min(1.0, mean + 0.02),
                            "n": 100,
                            "metric": "jaccard",
                        }
                    )
        input_path = root / "jaccard_summary.csv"
        table = pd.DataFrame(records)
        table.to_csv(input_path, index=False)
        output = render(input_path, root / "figures")
        plotted = pd.read_csv(root / "figures" / "jaccard_plotted_values.csv")
        check("p-sweep figure is rendered", output.is_file() and output.stat().st_size > 0)
        check("plotted values exactly retain summary rows", len(plotted) == len(table))
        image = imread(output)
        check(
            "p-sweep figure uses wide publication-grid aspect ratio",
            image.shape[1] / image.shape[0] > 1.8,
            f"aspect={image.shape[1] / image.shape[0]:.3f}",
        )


def test_analysis_endpoint_and_auc_contract() -> None:
    rows = pd.DataFrame(
        {
            "environment": ["EgeExp1"] * 4,
            "algorithm": ["TTPFTS_p0.5"] * 4,
            "rho": [0.5] * 4,
            "repetition": [0, 0, 1, 1],
            "arm_pull": [0, 10, 0, 10],
            "max_budget": [10] * 4,
            "normalized_budget": [0.0, 1.0, 0.0, 1.0],
            "bernoulli": [0.0, 1.0, 0.0, 0.0],
            "jaccard": [0.0, 0.5, 0.0, 1.0],
            "misidentification": [1.0, 0.5, 1.0, 0.0],
        }
    )
    summary = metric_summary(rows, "jaccard")
    trajectory, aggregate, comparison = endpoint_auc_summary(rows)
    check("checkpoint summary retains one row per normalized budget", len(summary) == 2)
    check(
        "trajectory AUC uses normalized-budget integration",
        np.allclose(sorted(trajectory[trajectory["metric"] == "jaccard"]["normalized_budget_auc"]), [0.25, 0.5]),
    )
    check(
        "endpoint and AUC summaries preserve all three metrics",
        len(aggregate) == 3 and len(comparison) == 3,
    )


if __name__ == "__main__":
    execute("Frozen p-sweep configuration", test_config_contract)
    execute("p=0.5 identity and config-bound resume", test_p_half_identity_and_config_bound_resume)
    execute("Serial/parallel invariance", test_serial_parallel_invariance)
    execute("Publication plotting input fidelity", test_plot_uses_exact_summary_table)
    execute("Endpoint and AUC analysis", test_analysis_endpoint_and_auc_contract)
    print("\n" + "=" * 80)
    if FAILURES:
        print("FAILED:", ", ".join(FAILURES))
        raise SystemExit(1)
    print("ALL TESTS PASSED")
