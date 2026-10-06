#!/usr/bin/env python3
"""Render the doubling-trick alternative of the three main publication figures.

The unchanged ``plotting.plot_metric_all_envs_grid_top_legend`` renderer is
called with exactly the settings of ``plot_results_local.py``; only the two EGE
curves are replaced by their doubling-trick (D-EGE) counterparts. Rendering is
refused unless the latest validation report passed for the exact current data.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from plotting import plot_metric_all_envs_grid_top_legend

from .build import plot_input_path
from .config import (
    DOUBLING_ALGORITHMS,
    METRICS,
    REPO_ROOT,
    assert_safe_output,
    config_hash,
    default_output_root,
    environment_specs,
    load_config,
    sha256_file,
    source_provenance,
    source_results_dir,
    verify_protected_hashes,
)
from .run import write_json_atomic
from .validate import figure_manifest_path, figure_paths, report_path


METRIC_TITLES = {"jaccard": "Jaccard", "bernoulli": "Bernoulli", "misidentification": "Misidentification"}


def verify_reference_rendering(config: dict) -> dict:
    """Prove this rendering environment reproduces the existing publication figures exactly.

    Tick density depends on backend text metrics (the originals were drawn with
    the default macOS backend, whose layout differs from headless Agg), so the
    unchanged ``plot_results_local`` path re-renders the originals into a
    temporary directory and every PNG/table must be byte-identical.
    """
    import plot_results_local

    with tempfile.TemporaryDirectory() as directory:
        plot_results_local.main(["--results_dir", str(source_results_dir(config)), "--figures_dir", directory])
        plt.close("all")
        for name, expected in sorted(config["protected_publication_files"].items()):
            if sha256_file(Path(directory) / name) != expected:
                raise RuntimeError(
                    f"backend {matplotlib.get_backend()!r} does not reproduce {name}; "
                    "render with the backend used for the original publication figures"
                )
    return {"backend": matplotlib.get_backend(), "reference_publication_figures_reproduced": True}


def _require_validated_data(output_root: Path, config: dict, names: list[str], algorithms: list[str]) -> dict:
    path = report_path(output_root)
    if not path.is_file():
        raise RuntimeError(f"no validation report at {path}; run experiments.ege_doubling.validate first")
    with path.open() as handle:
        report = json.load(handle)
    if report.get("status") != "pass" or report.get("doubling_config_hash") != config_hash(config):
        raise RuntimeError("the latest doubling validation did not pass for this configuration")
    if report.get("environments") != names or report.get("algorithms") != algorithms:
        raise RuntimeError("the latest validation covered different environments or algorithms")
    for name in names:
        if report["summaries"][name]["data_sha256"]["plot_input"] != sha256_file(plot_input_path(output_root, name)):
            raise RuntimeError(f"{name}: plot input changed after validation")
    return report


def render(
    output_root: Path | None = None,
    config: dict | None = None,
    environments: list[str] | None = None,
    algorithms: list[str] | None = None,
    figures_dir: Path | None = None,
    overwrite: bool = False,
    allow_dirty: bool = False,
    verify_style: bool = True,
) -> dict[str, str]:
    config = load_config() if config is None else config
    output_root = Path(default_output_root(config) if output_root is None else output_root)
    figures_dir = Path(REPO_ROOT / config["figures_dir"] if figures_dir is None else figures_dir)
    assert_safe_output(figures_dir, config)
    algorithms = algorithms or list(DOUBLING_ALGORITHMS)
    specs = [spec for spec in environment_specs(config) if environments is None or spec["name"] in environments]
    names = [spec["name"] for spec in specs]
    report = _require_validated_data(output_root, config, names, algorithms)
    provenance = source_provenance(allow_dirty=allow_dirty)

    targets = [path for metric in METRICS for path in figure_paths(figures_dir, metric)]
    existing = [str(path) for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing doubling figures: {existing}")
    verify_protected_hashes(config)
    style = verify_reference_rendering(config) if verify_style else {
        "backend": matplotlib.get_backend(), "reference_publication_figures_reproduced": False,
    }

    plot = config["plot"]
    plt.rcParams.update({"font.size": plot["font_size"], "font.family": plot["font_family"]})
    labels = {name: entry["label"] for name, entry in config["doubling_algorithms"].items()}
    labels["TTPFTS_UNKNOWN_VAR"] = config["anytime_algorithms"]["TTPFTS_UNKNOWN_VAR"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        png, table = figure_paths(figures_dir, metric)
        plot_metric_all_envs_grid_top_legend(
            env_files=[str(plot_input_path(output_root, name)) for name in names],
            env_labels=names,
            env_arms=[spec["num_arms"] for spec in specs],
            env_objs=[spec["num_objectives"] for spec in specs],
            metric=METRIC_TITLES[metric],
            figsize=tuple(plot["figsize"]),
            save_png=True,
            save_file=str(png),
            table_file=str(table),
            manifest_path=None,
            environment_budgets={spec["name"]: spec["budget"] for spec in specs},
            exclude_algs={"TTPFTS_KNOWN_VAR"},
            algorithm_order=plot["algorithm_order"],
            algorithm_display_labels=labels,
            log_misidentification=plot["log_misidentification"],
            show=False,
        )
        plt.close("all")
    verify_protected_hashes(config)

    files = {}
    for metric in METRICS:
        for path in figure_paths(figures_dir, metric):
            files[path.name] = sha256_file(path)
    write_json_atomic(figure_manifest_path(figures_dir), {
        "schema_version": "ege-doubling-figures-v1",
        "doubling_config_hash": config_hash(config),
        "environments": names,
        "algorithms": algorithms,
        "display_labels": labels,
        "plot_settings": plot,
        "rendering": style,
        "validation_report_sha256": sha256_file(report_path(output_root)),
        "plot_input_sha256": {name: report["summaries"][name]["data_sha256"]["plot_input"] for name in names},
        "files": files,
        **provenance,
    })
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing doubling figures only.")
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument("--skip-style-check", action="store_true",
                        help="Testing only: skip re-rendering the original figures byte-identically.")
    args = parser.parse_args()
    files = render(
        output_root=Path(args.output_root) if args.output_root else None,
        figures_dir=Path(args.figures_dir) if args.figures_dir else None,
        overwrite=args.overwrite,
        allow_dirty=args.allow_dirty_source,
        verify_style=not args.skip_style_check,
    )
    print(json.dumps(files, indent=2))


if __name__ == "__main__":
    main()
