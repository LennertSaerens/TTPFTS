"""Strict configuration loading and canonical hashing for this experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CANDIDATE_CONFIG_PATH = ROOT / "candidate_config.json"
REQUIRED_ALGORITHMS = {"EGE_SH", "EGE_SR", "Uniform", "APE", "TTPFTS_UNKNOWN_VAR"}
REQUIRED_CONDITIONS = ["heterogeneous", "homoscedastic_rms"]


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def load_candidate_config(path: str | Path = CANDIDATE_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        config = json.load(f)
    validate_candidate_config(config)
    return config


def validate_candidate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != "heteroscedastic-candidate-v1":
        raise ValueError("unexpected heteroscedastic candidate schema_version")
    if int(config.get("candidate_instances", 0)) < 1:
        raise ValueError("candidate_instances must be positive")
    for name in ("arms", "objectives"):
        bounds = config.get(name, {})
        lower, upper = int(bounds.get("minimum", 0)), int(bounds.get("maximum", -1))
        if lower < 1 or upper < lower:
            raise ValueError(f"{name} bounds must be positive and ordered")
    means = config.get("means", {})
    if not 0 <= float(means.get("minimum", -1)) < float(means.get("maximum", -1)) <= 1:
        raise ValueError("means must specify an ordered range within [0, 1]")
    values = config.get("reward_model", {}).get("heterogeneous_std_values", [])
    if sorted(float(v) for v in values) != [0.1, 0.2, 0.3, 0.4, 0.5]:
        raise ValueError("heterogeneous_std_values must be exactly [0.1, ..., 0.5]")
    if config.get("conditions") != REQUIRED_CONDITIONS:
        raise ValueError(f"conditions must be {REQUIRED_CONDITIONS}")
    if set(config.get("algorithms", [])) != REQUIRED_ALGORITHMS:
        raise ValueError(f"algorithms must be exactly {sorted(REQUIRED_ALGORITHMS)}")
    if int(config.get("evaluation_grid", {}).get("num_intervals", 0)) != 10:
        raise ValueError("candidate experiment requires 10 normalized intervals")
