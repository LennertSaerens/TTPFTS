"""Additional UV/KV both-comparisons using current-stage-only empirical D-EGE.

Run with ``python -m experiments.information_matched.no_pool``. All existing
results and figures are read-only. Only EGE stages are replayed; naive curves,
Uniform, APE, APE-UV and both TTPFTS variants are reused verbatim.

At an endpoint the empirical recommendation uses the stage just completed.
At the next pull all recommendation statistics reset, including unsampled-arm
means (zero, as in the original workflow). Native EGE selection, seeds, stage
budgets, nominal idle slots and within-round ordering remain unchanged.
The first pull after every restart is explicitly recorded. Stage-only empirical
Pareto sets need not equal the native accepted-arm sets held by the -N curves;
sample pooling is removed here, not that estimator difference.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from . import plot, run_empirical
from .build import plot_input_path
from .config import (
    EMPIRICAL_ALGORITHMS,
    REPO_ROOT,
    assert_safe_output,
    config_hash,
    default_output_root,
    environment_specs,
    load_config,
    sha256_file,
    verify_protected,
    verify_sources,
)
from .fileio import atomic_csv, atomic_json
from .validate import (
    Problems,
    report_path,
    validate_empirical,
    validate_figures,
)

CONFIG_PATH = Path(__file__).with_name("no_pool_config.json")


def load_no_pool_config() -> dict:
    with CONFIG_PATH.open() as handle:
        binding = json.load(handle)
    base = load_config()
    if binding["base_config_hash"] != config_hash(base):
        raise ValueError("stage-only config does not match the original information-matched configuration")
    if binding["sample_memory"] != "current_stage":
        raise ValueError("stage-only comparison must reset recommendation statistics at each stage")
    config = copy.deepcopy(base)
    config["no_pool_binding"] = binding
    config["empirical_doubling"]["sample_memory"] = binding["sample_memory"]
    config["empirical_doubling"]["recommendation"] = (
        "Pareto set of empirical means from the current stage only; statistics reset strictly after endpoints"
    )
    config["empirical_doubling"]["evaluation_points"] = (
        "shared grid, completed stage endpoints, and first pull after each restart"
    )
    config["figures_dir"] = binding["figures_dir"]
    config["protected_files"].update(binding["additional_protected_files"])
    config["figure_sets"] = {
        f"{name}_no_pool": base["figure_sets"][name] for name in ("uv_both", "kv_both")
    }
    config["plot"]["legend_kwargs"] = {
        f"{name}_no_pool": base["plot"]["legend_kwargs"][name] for name in ("uv_both", "kv_both")
    }
    return config


def output_root(config: dict) -> Path:
    return REPO_ROOT / f"results_information_matched_no_pool_{config_hash(config)[:12]}"


def source_root(config: dict) -> Path:
    return REPO_ROOT / config["no_pool_binding"]["source_root"]


def build_inputs(root: Path, config: dict, names: list[str]) -> dict:
    """Keep all old plot-input rows except the two empirical D-EGE curves."""
    assert_safe_output(root, config)
    files = {}
    for name in names:
        original = pd.read_csv(plot_input_path(source_root(config), name), dtype=str, keep_default_na=False)
        fresh = pd.read_csv(run_empirical.results_path(root, name, "trajectory"), dtype=str, keep_default_na=False)
        replacement = fresh.loc[:, original.columns]
        combined = pd.concat(
            [original[~original.algorithm.isin(EMPIRICAL_ALGORITHMS)], replacement], ignore_index=True
        )
        path = plot_input_path(root, name)
        atomic_csv(path, list(combined.columns), combined.to_dict("records"))
        files[name] = sha256_file(path)
    atomic_json(root / "plot_inputs" / "no_pool_manifest.json", {
        "config_hash": config_hash(config),
        "sample_memory": "current_stage",
        "source_root": str(source_root(config)),
        "files": files,
    })
    return files


def validate(root: Path, config: dict, names: list[str], repetitions: list[int],
             require_clean: bool = True, check_figures: bool = False,
             figures_dir: Path | None = None) -> dict:
    """Independent per-pull stage-reset replay plus native-stage/source identity checks."""
    problems = Problems()
    verify_sources(config)
    verify_protected(config)
    specs = [spec for spec in environment_specs(config) if spec["name"] in names]
    summaries = {}
    for spec in specs:
        name = spec["name"]
        summary = validate_empirical(
            root, spec, config, repetitions, require_clean,
            config["validation_samples"]["empirical_recompute"], problems,
        )
        # The classifier's return value and allocation must remain identical to
        # the pooled workflow, including its incomplete final stage.
        old_stages = pd.read_csv(
            run_empirical.results_path(source_root(config), name, "stages"),
            dtype=str, keep_default_na=False,
        )
        old_stages = old_stages[old_stages.repetition.isin([str(r) for r in repetitions])]
        new_stages = pd.read_csv(run_empirical.results_path(root, name, "stages"), dtype=str, keep_default_na=False)
        columns = [column for column in old_stages if column != "config_hash"]
        if not old_stages[columns].reset_index(drop=True).equals(new_stages[columns].reset_index(drop=True)):
            problems.add(f"{name}: stage selection/returns changed from the pooled experiment")
        source = pd.read_csv(plot_input_path(source_root(config), name), dtype=str, keep_default_na=False)
        current = pd.read_csv(plot_input_path(root, name), dtype=str, keep_default_na=False)
        original_other = source[~source.algorithm.isin(EMPIRICAL_ALGORITHMS)].reset_index(drop=True)
        new_other = current[~current.algorithm.isin(EMPIRICAL_ALGORITHMS)].reset_index(drop=True)
        if not original_other.equals(new_other):
            problems.add(f"{name}: reused naive/anytime plot rows changed")
        fresh = pd.read_csv(
            run_empirical.results_path(root, name, "trajectory"), dtype=str, keep_default_na=False
        )
        if not current[current.algorithm.isin(EMPIRICAL_ALGORITHMS)].reset_index(drop=True).equals(
                fresh.loc[:, source.columns].reset_index(drop=True)):
            problems.add(f"{name}: stage-only plot inputs differ from their trajectories")
        summaries[name] = {
            "empirical": summary,
            "plot_input_sha256": sha256_file(plot_input_path(root, name)),
        }
    figure_files = None
    if check_figures:
        figure_files = validate_figures(
            root, figures_dir or REPO_ROOT / config["figures_dir"], specs, config, len(repetitions), problems
        )
    report = {
        "status": "pass" if not problems else "fail",
        "config_hash": config_hash(config),
        "sample_memory": "current_stage",
        "environments": [spec["name"] for spec in specs],
        "repetitions": repetitions,
        "summaries": summaries,
        "figures_checked": check_figures,
        "figure_files": figure_files,
        "problems": list(problems),
    }
    atomic_json(report_path(root), report)
    if problems:
        raise ValueError("stage-only validation failed: " + "; ".join(problems[:20]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["run", "build", "validate", "plot", "all"], default="all")
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    config = load_no_pool_config()
    root = output_root(config)
    names = [spec["name"] for spec in environment_specs(config)]
    repetitions = list(range(config["num_runs"]))
    verify_sources(config)
    verify_protected(config)
    if args.phase in ("all", "run"):
        run_empirical.run(root, config=config, n_jobs=args.n_jobs)
    if args.phase in ("all", "build"):
        build_inputs(root, config, names)
    if args.phase in ("all", "validate"):
        report = validate(root, config, names, repetitions)
        print("Validation PASS", {
            name: summary["empirical"]["reproduced_naive_stages"]
            for name, summary in report["summaries"].items()
        })
    if args.phase in ("all", "plot"):
        plot.render(root, config=config)
        validate(root, config, names, repetitions, check_figures=True)
        print(f"Six figures created in {config['figures_dir']}")
    verify_sources(config)
    verify_protected(config)
    print(root)


if __name__ == "__main__":
    main()
