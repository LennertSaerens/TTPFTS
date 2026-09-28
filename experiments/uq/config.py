"""Immutable configuration for the hardness-budget UQ replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.final_config_loader import CONFIG_PATH as FINAL_CONFIG_PATH
from experiments.final_config_loader import config_hash as final_config_hash
from experiments.final_config_loader import load_final_config
from runner.grid import build_normalized_grid


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != "hardness-budget-uq-v1":
        raise ValueError("unexpected UQ config schema version")
    final_config = load_final_config()
    if config.get("final_config_sha256") != final_config_hash(final_config):
        raise ValueError("UQ config does not match the committed final experiment config")
    final_file_hash = sha256_file(FINAL_CONFIG_PATH)
    if config.get("final_config_file_sha256") != final_file_hash:
        raise ValueError("UQ config does not match final_config.json bytes")
    if config.get("algorithm") != "TTPFTS_UNKNOWN_VAR":
        raise ValueError("UQ replay is defined only for TTPFTS_UNKNOWN_VAR")
    if config.get("num_runs") != final_config["num_runs"]:
        raise ValueError("UQ repetition count differs from final experiment")
    if config.get("master_seed") != final_config["master_seed"]:
        raise ValueError("UQ master seed differs from final experiment")
    if config.get("eval_points") != final_config["eval_points"]:
        raise ValueError("UQ evaluation-grid size differs from final experiment")
    expected = []
    for environment in final_config["environments"]:
        budget = int(environment["budget"])
        grid = build_normalized_grid(budget, int(final_config["eval_points"]))
        expected.append(
            {
                "name": environment["name"],
                "budget": budget,
                "grid": grid,
                "instance_hash": environment["instance_hash"],
                "num_arms": environment["num_arms"],
                "num_objectives": environment["num_objectives"],
            }
        )
    expected_digest = hashlib.sha256(canonical_json(expected).encode("utf-8")).hexdigest()
    if config.get("environment_spec_sha256") != expected_digest:
        raise ValueError("UQ environments/budgets/grids do not match final experiment")
    return config


def config_hash(config: dict[str, Any] | None = None) -> str:
    config = load_config() if config is None else config
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def environment_config(name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_config() if config is None else config
    final_config = load_final_config()
    for environment in final_config["environments"]:
        if environment["name"] == name:
            budget = int(environment["budget"])
            return {
                "name": environment["name"],
                "budget": budget,
                "grid": build_normalized_grid(budget, int(final_config["eval_points"])),
                "instance_hash": environment["instance_hash"],
                "num_arms": environment["num_arms"],
                "num_objectives": environment["num_objectives"],
            }
    raise KeyError(f"unknown UQ environment: {name}")
