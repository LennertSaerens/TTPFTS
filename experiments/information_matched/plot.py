#!/usr/bin/env python3
"""Render the four information-matched figure sets (12 figures).

The unchanged publication renderer is used with the settings of
``plot_results_local.py``; fixed colors keep every algorithm family
recognisable across sets and naive doubling curves (``-N``) are dashed.
Rendering is refused unless the latest validation passed for the exact
current plot inputs, and (by default) unless this machine first re-renders
both existing publication figure sets byte-identically.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from plotting import plot_metric_all_envs_grid_top_legend

from .build import ALL_ALGORITHMS, plot_input_path
from .config import (
    METRICS,
    REPO_ROOT,
    assert_safe_output,
    config_hash,
    default_output_root,
    environment_specs,
    load_config,
    sha256_file,
    source_provenance,
    verify_protected,
)
from .fileio import atomic_json
from .validate import figure_manifest_path, figure_paths, report_path

METRIC_TITLES = {"jaccard": "Jaccard", "bernoulli": "Bernoulli", "misidentification": "Misidentification"}


def verify_reference_rendering(config: dict) -> dict:
    """Re-render ``figures/publication`` and ``figures/publication_ege_doubling`` byte-identically.

    Tick density depends on backend text metrics, and the renderer gained
    optional style arguments; both existing figure sets must still be
    reproduced exactly before any new figure is drawn.
    """
    import plot_results_local
    from experiments.ege_doubling.plot import render as render_doubling

    protected = config["protected_files"]
    with tempfile.TemporaryDirectory() as directory:
        publication, doubling = Path(directory) / "publication", Path(directory) / "doubling"
        plot_results_local.main(["--results_dir", config["source_results_dir"], "--figures_dir", str(publication)])
        plt.close("all")
        render_doubling(figures_dir=doubling, verify_style=False)
        plt.close("all")
        for folder, files in (("figures/publication", publication), ("figures/publication_ege_doubling", doubling)):
            for name, expected in protected[folder].items():
                if name == "figure_manifest.json":
                    continue
                if sha256_file(files / name) != expected:
                    raise RuntimeError(f"backend {matplotlib.get_backend()!r} does not reproduce {folder}/{name}")
    return {"backend": matplotlib.get_backend(), "reference_figures_reproduced": True}


def _require_validated(output_root: Path, config: dict, names: list[str]) -> dict:
    path = report_path(output_root)
    if not path.is_file():
        raise RuntimeError(f"no validation report at {path}")
    with path.open() as handle:
        report = json.load(handle)
    if report.get("status") != "pass" or report.get("config_hash") != config_hash(config):
        raise RuntimeError("the latest validation did not pass for this configuration")
    if report.get("environments") != names:
        raise RuntimeError("the latest validation covered different environments")
    for name in names:
        if report["summaries"][name]["plot_input_sha256"] != sha256_file(plot_input_path(output_root, name)):
            raise RuntimeError(f"{name}: plot input changed after validation")
    return report


def render(output_root: Path | None = None, config: dict | None = None, environments: list[str] | None = None,
           figures_dir: Path | None = None, overwrite: bool = False, allow_dirty: bool = False,
           verify_style: bool = True) -> dict[str, str]:
    config = load_config() if config is None else config
    output_root = Path(default_output_root(config) if output_root is None else output_root)
    figures_dir = Path(REPO_ROOT / config["figures_dir"] if figures_dir is None else figures_dir)
    assert_safe_output(figures_dir, config)
    specs = [s for s in environment_specs(config) if environments is None or s["name"] in environments]
    names = [s["name"] for s in specs]
    report = _require_validated(output_root, config, names)
    provenance = source_provenance(allow_dirty=allow_dirty)
    targets = [p for set_name in config["figure_sets"] for metric in METRICS for p in figure_paths(figures_dir, set_name, metric)]
    existing = [str(p) for p in targets if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing figures: {existing}")
    verify_protected(config)
    style = verify_reference_rendering(config) if verify_style else {
        "backend": matplotlib.get_backend(), "reference_figures_reproduced": False,
    }

    plot = config["plot"]
    plt.rcParams.update({"font.size": plot["font_size"], "font.family": plot["font_family"]})
    figures_dir.mkdir(parents=True, exist_ok=True)
    for set_name, algorithms in config["figure_sets"].items():
        for metric in METRICS:
            png, table = figure_paths(figures_dir, set_name, metric)
            plot_metric_all_envs_grid_top_legend(
                env_files=[str(plot_input_path(output_root, name)) for name in names],
                env_labels=names,
                env_arms=[s["num_arms"] for s in specs],
                env_objs=[s["num_objectives"] for s in specs],
                metric=METRIC_TITLES[metric],
                figsize=tuple(plot["figsize"]),
                save_png=True,
                save_file=str(png),
                table_file=str(table),
                manifest_path=None,
                environment_budgets={s["name"]: s["budget"] for s in specs},
                exclude_algs=set(ALL_ALGORITHMS) - set(algorithms),
                algorithm_order=algorithms,
                algorithm_display_labels=config["labels"],
                algorithm_styles=config["styles"],
                legend_ncol=len(algorithms),
                legend_kwargs=plot["legend_kwargs"][set_name],
                log_misidentification=plot["log_misidentification"],
                show=False,
            )
            plt.close("all")
    verify_protected(config)
    files = {}
    for set_name in config["figure_sets"]:
        for metric in METRICS:
            for path in figure_paths(figures_dir, set_name, metric):
                files[path.name] = sha256_file(path)
    atomic_json(figure_manifest_path(figures_dir), {
        "schema_version": "information-matched-figures-v1",
        "config_hash": config_hash(config),
        "environments": names,
        "figure_sets": config["figure_sets"],
        "labels": config["labels"],
        "styles": config["styles"],
        "plot_settings": plot,
        "rendering": style,
        "validation_report_sha256": sha256_file(report_path(output_root)),
        "plot_input_sha256": {name: report["summaries"][name]["plot_input_sha256"] for name in names},
        "files": files,
        **provenance,
    })
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument("--skip-style-check", action="store_true")
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
