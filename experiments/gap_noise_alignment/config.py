"""Strict immutable configuration for the paired noise/gap-alignment study."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
SCHEMA_VERSION = "gap-noise-alignment-v1"
ALGORITHMS = ("EGE_SH", "EGE_SR", "Uniform", "APE", "TTPFTS_UNKNOWN_VAR")
CONDITIONS = ("large_gap_noisy", "small_gap_noisy")
STD_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5)


def canonical_json(data: Any) -> str:
    """Serialize source data byte-identically for a configuration identity."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and enforce every approved scientific and execution decision."""
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected gap-noise-alignment schema")
    if tuple(config.get("algorithms", ())) != ALGORITHMS:
        raise ValueError(f"algorithms must be exactly {ALGORITHMS}")
    if tuple(config.get("conditions", ())) != CONDITIONS:
        raise ValueError(f"conditions must be exactly {CONDITIONS}")
    if int(config.get("repetitions", 0)) != 100:
        raise ValueError("study requires exactly 100 paired repetitions")
    environment = config.get("environment", {})
    if (
        int(environment.get("num_arms", 0)) != 50
        or int(environment.get("num_objectives", 0)) != 2
        or int(environment.get("num_anchors", 0)) != 5
        or tuple(float(v) for v in environment.get("anchor_start", ())) != (0.9, 0.4)
        or tuple(float(v) for v in environment.get("anchor_step", ())) != (-0.125, 0.125)
        or tuple(float(v) for v in environment.get("child_nominal_gaps", ())) != (0.04, 0.08, 0.16)
        or tuple(float(v) for v in environment.get("child_epsilons", ())) != (0.001, 0.002, 0.003)
        or tuple(float(v) for v in environment.get("mean_domain", ())) != (0.0, 1.0)
    ):
        raise ValueError("custom mean geometry differs from the approved design")
    noise = config.get("noise_assignment", {})
    if (
        noise.get("assignment") != "reverse_standard_deviation_by_stable_true_gap_rank"
        or noise.get("rank_key") != "round(true_gap, 12), arm_index"
        or noise.get("per_arm_isotropic") is not True
        or tuple(float(v) for v in noise.get("standard_deviations", ())) != STD_VALUES
        or int(noise.get("arms_per_standard_deviation", 0)) != 10
    ):
        raise ValueError("noise/gap alignment differs from the approved design")
    hardness = config.get("hardness", {})
    if (
        hardness.get("definition") != "mean_only_sum_inverse_squared_gaps"
        or hardness.get("budget_rule") != "round(H(nu))"
        or hardness.get("zero_gap_policy") != "reject"
        or not math.isclose(float(hardness.get("expected_hardness", math.nan)), 14290.552976858806)
    ):
        raise ValueError("hardness semantics differ from the approved design")
    grid = config.get("evaluation_grid", {})
    if int(grid.get("num_intervals", 0)) != 20 or grid.get("type") != "normalized_endpoint_inclusive":
        raise ValueError("experiment requires the approved 21-point normalized grid")
    if config.get("reward_model") != {
        "distribution": "independent_gaussian",
        "paired_latent_streams": "per_arm_per_objective_streams_shared_across_conditions",
    }:
        raise ValueError("reward pairing semantics differ from the approved design")
    if config.get("ttpfts") != {
        "posterior": "TPosterior",
        "alpha": 0.0,
        "p": 0.5,
        "warmup_pulls_per_arm": 4,
    }:
        raise ValueError("TTPFTS semantics differ from the approved design")
    bootstrap = config.get("bootstrap", {})
    if int(bootstrap.get("replicates", 0)) != 2000 or float(bootstrap.get("confidence", 0.0)) != 0.95:
        raise ValueError("bootstrap specification differs from the approved design")
    if config.get("zero_budget_policy") != "native_algorithm":
        raise ValueError("native zero-budget behavior is required")
    return config


def config_hash(config: dict[str, Any] | None = None) -> str:
    config = load_config() if config is None else config
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def default_output_root(config: dict[str, Any] | None = None) -> Path:
    return Path(f"results_gap_noise_alignment_{config_hash(config)[:12]}")
