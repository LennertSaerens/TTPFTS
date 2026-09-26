"""Local entry point for publication MOMAB result plots.

Uses the tidy runner CSVs (``<results_dir>/EgeExp*.csv``) and their sidecar
manifest (``<results_dir>/run/manifest.json``).  Legacy archive helpers remain
in plotting.py but are not used on this publication path.

The results directory is resolved in this order: an explicit ``--results_dir``
argument, then the preregistered final output directory if it exists, then the
development ``results/`` directory.  Budgets and labels always come from the
sidecar manifest when one is present, never from hard-coded filenames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from plotting import DEFAULT_EGE_BUDGETS, plot_metric_all_envs_grid_top_legend

plt.rcParams.update({"font.size": 14, "font.family": "serif"})


def get_env_configs():
    return [60, 10, 200, 50, 20, 10, 22, 5], [2, 2, 2, 10, 2, 2, 2, 2]


def _manifest_or_defaults(results_dir: Path) -> tuple[Path | None, dict[str, int]]:
    manifest_path = results_dir / "run" / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        return manifest_path, {str(k): int(v) for k, v in manifest.get("budgets", {}).items()}
    return None, DEFAULT_EGE_BUDGETS


PREREGISTERED_RESULTS_GLOB = "results_preregistered_final*"
DEVELOPMENT_RESULTS_DIR = Path("results")


def resolve_results_dir(explicit: str | None = None) -> Path:
    """Pick the results directory for the publication plotting path.

    The preregistered run writes to ``results_preregistered_final_<hash>``,
    where the suffix is the configuration hash, so match by glob rather than
    an exact name. If several config hashes are present the choice is
    ambiguous and must be made explicitly rather than guessed.
    """
    if explicit is not None:
        return Path(explicit)
    candidates = sorted(p for p in Path(".").glob(PREREGISTERED_RESULTS_GLOB) if p.is_dir())
    if len(candidates) > 1:
        raise SystemExit(
            "Multiple preregistered result directories found: "
            + ", ".join(str(c) for c in candidates)
            + "\nPass --results_dir to choose one explicitly."
        )
    if candidates:
        return candidates[0]
    return DEVELOPMENT_RESULTS_DIR


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir",
        default=None,
        help="Directory holding the tidy EgeExp*.csv files and run/manifest.json.",
    )
    parser.add_argument(
        "--figures_dir",
        default="figures",
        help="Directory to write publication figures and plot tables into.",
    )
    args = parser.parse_args(argv)

    environments = [f"EgeExp{i}" for i in range(1, 9)]
    env_arms, env_objs = get_env_configs()
    results_dir = resolve_results_dir(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading results from {results_dir}")

    env_files = [results_dir / f"{env}.csv" for env in environments]
    missing = [str(path) for path in env_files if not path.exists()]
    if missing:
        print(f"Missing tidy result files ({len(missing)}/8): {missing}")
        return

    manifest_path, budgets = _manifest_or_defaults(results_dir)
    for metric in ["Jaccard", "Bernoulli", "Misidentification"]:
        metric_key = metric.lower()
        if metric_key == "misidentification":
            metric_key = "misidentification"
        print(f"Plotting {metric}...")
        plot_metric_all_envs_grid_top_legend(
            env_files=[str(p) for p in env_files],
            env_labels=environments,
            env_arms=env_arms,
            env_objs=env_objs,
            metric=metric,
            figsize=(13, 6.6),
            save_png=True,
            save_file=str(figures_dir / f"publication_{metric_key}.png"),
            table_file=str(figures_dir / f"publication_{metric_key}_plot_table.csv"),
            manifest_path=str(manifest_path) if manifest_path else None,
            environment_budgets=budgets,
            exclude_algs={"TTPFTS_KNOWN_VAR"},
            algorithm_display_labels={
                "EGE_SH": "EGE-SH",
                "EGE_SR": "EGE-SR",
                "TTPFTS_UNKNOWN_VAR": "TTPFTS",
            },
            log_misidentification=True,
            show=False,
        )
    plt.close("all")


if __name__ == "__main__":
    main()
