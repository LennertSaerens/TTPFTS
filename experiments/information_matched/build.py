#!/usr/bin/env python3
"""Assemble per-environment plot inputs from verbatim reused rows and the new trajectories."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from .config import (
    EMPIRICAL_ALGORITHMS,
    NAIVE_ALGORITHMS,
    NEW_ANYTIME_ALGORITHM,
    PREREGISTERED_ALGORITHMS,
    assert_safe_output,
    config_hash,
    default_output_root,
    doubling_dir,
    environment_specs,
    load_config,
    preregistered_dir,
    sha256_file,
    verify_protected,
    verify_sources,
)
from .fileio import atomic_json
from .run_ape_uv import manifest_path as ape_uv_manifest_path
from .run_ape_uv import results_path as ape_uv_results_path
from .run_empirical import manifest_path as empirical_manifest_path
from .run_empirical import results_path as empirical_results_path


PLOT_COLUMNS = [
    "environment", "algorithm", "posterior_model", "repetition", "arm_pull", "max_budget",
    "bernoulli", "jaccard", "misidentification",
]
ALL_ALGORITHMS = [*EMPIRICAL_ALGORITHMS, *NAIVE_ALGORITHMS, *PREREGISTERED_ALGORITHMS, NEW_ANYTIME_ALGORITHM]


def plot_input_path(output_root: Path, environment: str) -> Path:
    return Path(output_root) / "plot_inputs" / f"{environment}.csv"


def build_manifest_path(output_root: Path) -> Path:
    return Path(output_root) / "plot_inputs" / "manifest.json"


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def source_frames(output_root: Path, environment: str, config: dict) -> dict[str, pd.DataFrame]:
    """Every curve's rows as strings, exactly as stored in its source file."""
    preregistered = _read(preregistered_dir(config) / f"{environment}.csv")
    naive = _read(doubling_dir(config) / "doubling" / f"{environment}.csv")
    naive = naive[naive["algorithm"].isin(NAIVE_ALGORITHMS.values())].copy()
    naive["algorithm"] = naive["algorithm"].map({v: k for k, v in NAIVE_ALGORITHMS.items()})
    return {
        "preregistered": preregistered[preregistered["algorithm"].isin(PREREGISTERED_ALGORITHMS)].loc[:, PLOT_COLUMNS],
        "naive": naive.loc[:, PLOT_COLUMNS],
        "ape_uv": _read(ape_uv_results_path(output_root, environment)).loc[:, PLOT_COLUMNS],
        "empirical": _read(empirical_results_path(output_root, environment, "trajectory")).loc[:, PLOT_COLUMNS],
    }


def build(output_root: Path | None = None, config: dict | None = None,
          environments: list[str] | None = None) -> dict[str, dict[str, int]]:
    config = load_config() if config is None else config
    output_root = Path(default_output_root(config) if output_root is None else output_root)
    assert_safe_output(output_root, config)
    verify_sources(config)
    verify_protected(config)
    names = environments or [spec["name"] for spec in environment_specs(config)]
    summary, files = {}, {}
    for name in names:
        frames = source_frames(output_root, name, config)
        combined = pd.concat(frames.values(), ignore_index=True)
        if sorted(combined["algorithm"].unique()) != sorted(ALL_ALGORITHMS):
            raise ValueError(f"{name}: plot input algorithms {sorted(combined['algorithm'].unique())} are incomplete")
        path = plot_input_path(output_root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".csv")
        os.close(descriptor)
        try:
            combined.to_csv(temporary, index=False)
            os.replace(temporary, path)
        except BaseException:
            if os.path.exists(temporary):
                os.remove(temporary)
            raise
        summary[name] = combined.groupby("algorithm").size().to_dict()
        files[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "source_sha256": {
                "ape_uv": sha256_file(ape_uv_results_path(output_root, name)),
                "empirical": sha256_file(empirical_results_path(output_root, name, "trajectory")),
            },
        }
        for manifest in (ape_uv_manifest_path(output_root, name), empirical_manifest_path(output_root, name)):
            if not manifest.is_file():
                raise ValueError(f"{name}: missing run manifest {manifest}")
    atomic_json(build_manifest_path(output_root), {
        "schema_version": "information-matched-plot-inputs-v1",
        "config_hash": config_hash(config),
        "environments": names,
        "algorithms": ALL_ALGORITHMS,
        "rows": summary,
        "files": files,
    })
    verify_sources(config)
    verify_protected(config)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--environments", default=None)
    args = parser.parse_args()
    summary = build(
        output_root=Path(args.output_root) if args.output_root else None,
        environments=[e.strip() for e in args.environments.split(",")] if args.environments else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
