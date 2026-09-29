"""Immutable configuration and provenance helpers for the TTPFTS p sweep."""

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
SCHEMA_VERSION = "hardness-normalized-ttpfts-p-sweep-v1"
P_VALUES = tuple(round(index / 10, 1) for index in range(11))


def canonical_json(data: Any) -> str:
    """Serialize data deterministically for configuration hashing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_environments(final_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Derive the sweep's frozen environment/grid specification from final config."""
    final_config = load_final_config() if final_config is None else final_config
    return [
        {
            "name": environment["name"],
            "budget": int(environment["budget"]),
            "grid": build_normalized_grid(int(environment["budget"]), 20),
            "instance_hash": environment["instance_hash"],
            "num_arms": int(environment["num_arms"]),
            "num_objectives": int(environment["num_objectives"]),
        }
        for environment in final_config["environments"]
    ]


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the p-sweep config and reject stale final-config bindings."""
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected p-sweep config schema version")
    final_config = load_final_config()
    if config.get("final_config_sha256") != final_config_hash(final_config):
        raise ValueError("p-sweep config does not match the committed final experiment config")
    if config.get("final_config_file_sha256") != sha256_file(FINAL_CONFIG_PATH):
        raise ValueError("p-sweep config does not match final_config.json bytes")
    if config.get("algorithm") != "TTPFTS_UNKNOWN_VAR":
        raise ValueError("p sweep is defined only for TTPFTS_UNKNOWN_VAR")
    if config.get("num_runs") != final_config["num_runs"]:
        raise ValueError("p-sweep repetition count differs from final experiment")
    if config.get("master_seed") != final_config["master_seed"]:
        raise ValueError("p-sweep master seed differs from final experiment")
    if config.get("eval_intervals") != 20:
        raise ValueError("p sweep must use exactly 20 normalized intervals")
    if tuple(float(value) for value in config.get("p_values", ())) != P_VALUES:
        raise ValueError("p sweep must contain p=0.0,0.1,...,1.0 exactly")
    expected_posterior = {
        "class": "TPosterior",
        "alpha": 0.0,
        "unknown_variance": True,
        "warmup_pulls_per_arm": 4,
    }
    if config.get("posterior") != expected_posterior:
        raise ValueError("p-sweep posterior semantics differ from the approved design")
    if config.get("seeding") != {
        "strategy": "matched-across-p",
        "bandit_namespace": "TTPFTS_UNKNOWN_VAR",
    }:
        raise ValueError("p-sweep seeding must preserve matched trajectories across p")
    environment_digest = hashlib.sha256(
        canonical_json(expected_environments(final_config)).encode("utf-8")
    ).hexdigest()
    if config.get("environment_spec_sha256") != environment_digest:
        raise ValueError("p-sweep environments, budgets, or grids differ from final experiment")
    return config


def config_hash(config: dict[str, Any] | None = None) -> str:
    """Return the canonical hash that namespaces every sweep result."""
    config = load_config() if config is None else config
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def environment_config(name: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the final-config-bound specification for one environment."""
    del config  # Configuration validation happens at the caller boundary.
    for environment in expected_environments():
        if environment["name"] == name:
            return environment
    raise KeyError(f"unknown p-sweep environment: {name}")


def p_tag(value: float) -> str:
    """Encode an approved p value as a stable path-safe tag."""
    value = round(float(value), 1)
    if value not in P_VALUES:
        raise ValueError(f"p must be one of {P_VALUES}, got {value}")
    return f"p{value:.1f}"


def algorithm_label(value: float) -> str:
    """Return the self-describing result label for a sweep setting."""
    return f"TTPFTS_{p_tag(value)}"


def default_output_root(config: dict[str, Any] | None = None) -> Path:
    """Return the config-hash-qualified local result root."""
    return Path(f"results_p_sweep_{config_hash(config)[:12]}")
