"""Frozen configuration for known-variance UQ replay and held-out random-instance evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.final_config_loader import (
    CONFIG_PATH as FINAL_CONFIG_PATH,
    config_hash as final_config_hash,
    load_final_config,
)
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


def default_config() -> dict[str, Any]:
    final_config = load_final_config()
    expected = []
    for env in final_config["environments"]:
        budget = int(env["budget"])
        expected.append({
            "name": env["name"],
            "budget": budget,
            "grid": build_normalized_grid(budget, int(final_config["eval_points"])),
            "instance_hash": env["instance_hash"],
            "num_arms": env["num_arms"],
            "num_objectives": env["num_objectives"],
        })
    env_digest = hashlib.sha256(canonical_json(expected).encode("utf-8")).hexdigest()
    return {
        "schema_version": "known-var-uq-v1",
        "final_config_sha256": final_config_hash(final_config),
        "final_config_file_sha256": sha256_file(FINAL_CONFIG_PATH),
        "environment_spec_sha256": env_digest,
        "algorithm": "TTPFTS_KNOWN_VAR",
        "posterior": {
            "class": "NormalPosterior",
            "known_std": float(final_config["reward_model"]["std"]),
            "warmup_pulls_per_arm": 4,
            "rho": 0.5,
            "unknown_variance": False,
        },
        "num_runs": int(final_config["num_runs"]),
        "master_seed": int(final_config["master_seed"]),
        "eval_points": int(final_config["eval_points"]),
        "mc_draws": 512,
        "mc_seed_base": 20261118,
        "measures": [
            "bhattacharyya",
            "bhattacharyya_raw",
            "uq_v1",
            "uq_v2",
            "uq_mc",
            "symmetric_kl",
            "entropy",
        ],
        "heldout_seeds": {
            "checkpoint_stopping_seed": 20261116,
            "trajectory_correlation_seed": 20261117,
        },
        "frozen_selection": {
            "risk_bins": [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0],
            "coverage_fractions": [0.10, 0.25, 0.50],
            "stopping_thresholds_v1_v2_mc": [0.10, 0.05, 0.01],
            "stopping_thresholds_eq10": [0.01, 0.0001, 0.000004],
        },
        "random_trajectory_families": [
            {"name": "Unif_K10_d2", "law": "uniform", "k": 10, "d": 2, "sigma": 0.25, "budget": 5000, "num_runs": 100},
            {"name": "Norm_K10_d2", "law": "normal", "k": 10, "d": 2, "sigma": 0.25, "budget": 5000, "num_runs": 100},
            {"name": "Unif_K20_d2", "law": "uniform", "k": 20, "d": 2, "sigma": 0.25, "budget": 15000, "num_runs": 100},
            {"name": "Norm_K20_d2", "law": "normal", "k": 20, "d": 2, "sigma": 0.25, "budget": 15000, "num_runs": 100},
            {"name": "Unif_K50_d2", "law": "uniform", "k": 50, "d": 2, "sigma": 0.25, "budget": 40000, "num_runs": 100},
            {"name": "Unif_K10_d4", "law": "uniform", "k": 10, "d": 4, "sigma": 0.25, "budget": 5000, "num_runs": 100},
            {"name": "Unif_K20_d4", "law": "uniform", "k": 20, "d": 4, "sigma": 0.25, "budget": 15000, "num_runs": 100},
            {"name": "Unif_K10_d2_s05", "law": "uniform", "k": 10, "d": 2, "sigma": 0.50, "budget": 15000, "num_runs": 100},
        ],
    }


def write_default_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    cfg = default_config()
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return cfg


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != "known-var-uq-v1":
        raise ValueError("unexpected known-var UQ config schema version")
    final_config = load_final_config()
    if config.get("final_config_sha256") != final_config_hash(final_config):
        raise ValueError("UQ KV config does not match final_config.json hash")
    if config.get("final_config_file_sha256") != sha256_file(FINAL_CONFIG_PATH):
        raise ValueError("UQ KV config does not match final_config.json bytes")
    if config.get("algorithm") != "TTPFTS_KNOWN_VAR":
        raise ValueError("UQ KV config requires TTPFTS_KNOWN_VAR")
    expected = []
    for env in final_config["environments"]:
        budget = int(env["budget"])
        expected.append({
            "name": env["name"],
            "budget": budget,
            "grid": build_normalized_grid(budget, int(final_config["eval_points"])),
            "instance_hash": env["instance_hash"],
            "num_arms": env["num_arms"],
            "num_objectives": env["num_objectives"],
        })
    if config.get("environment_spec_sha256") != hashlib.sha256(canonical_json(expected).encode("utf-8")).hexdigest():
        raise ValueError("environment spec sha256 mismatch")
    return config


def config_hash(config: dict[str, Any] | None = None) -> str:
    cfg = load_config() if config is None else config
    return hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()


def environment_config(name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    _cfg = load_config() if config is None else config
    final_config = load_final_config()
    for env in final_config["environments"]:
        if env["name"] == name:
            budget = int(env["budget"])
            return {
                "name": env["name"],
                "budget": budget,
                "grid": build_normalized_grid(budget, int(final_config["eval_points"])),
                "instance_hash": env["instance_hash"],
                "num_arms": env["num_arms"],
                "num_objectives": env["num_objectives"],
            }
    raise KeyError(f"unknown environment: {name}")


if __name__ == "__main__":
    cfg = write_default_config()
    print(f"Wrote {CONFIG_PATH} with hash {config_hash(cfg)}")
