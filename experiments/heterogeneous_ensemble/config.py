"""Immutable configuration for the heterogeneous-noise ensemble experiment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments.final_config_loader import CONFIG_PATH as FINAL_CONFIG_PATH
from experiments.final_config_loader import config_hash as final_config_hash
from experiments.final_config_loader import load_final_config


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
SCHEMA_VERSION = "heterogeneous-ensemble-v1"
ALGORITHMS = ("EGE_SH", "EGE_SR", "Uniform", "APE", "TTPFTS_UNKNOWN_VAR")
STD_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5)


def canonical_json(data: Any) -> str:
    """Serialize configuration data identically across machines."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    """Stream a file digest without copying large frozen artifacts."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _final_egeexp3_hardness() -> float:
    final_config = load_final_config()
    for environment in final_config["environments"]:
        if environment["name"] == "EgeExp3":
            return float(environment["hardness_H"])
    raise ValueError("committed final configuration has no EgeExp3 hardness")


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load configuration and reject every source-of-truth mismatch."""
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected heterogeneous-ensemble schema version")
    final_config = load_final_config()
    if config.get("final_config_sha256") != final_config_hash(final_config):
        raise ValueError("ensemble config does not match the committed final config")
    if config.get("final_config_file_sha256") != sha256_file(FINAL_CONFIG_PATH):
        raise ValueError("ensemble config does not match final_config.json bytes")
    if tuple(config.get("algorithms", ())) != ALGORITHMS:
        raise ValueError(f"algorithms must be exactly {ALGORITHMS}")
    if int(config.get("trajectories_per_instance_algorithm", 0)) != 1:
        raise ValueError("ensemble requires exactly one trajectory per instance and algorithm")
    cohort = config.get("cohort", {})
    if int(cohort.get("accepted_instances", 0)) != 1000:
        raise ValueError("ensemble requires exactly 1000 accepted instances")
    if int(cohort.get("max_proposals", 0)) < int(cohort["accepted_instances"]):
        raise ValueError("max_proposals must cover every accepted instance")
    if cohort.get("proposal_rule") != "accept_first_eligible_in_increasing_proposal_id_order":
        raise ValueError("proposal acceptance must be an ordered prefix")
    grid = config.get("evaluation_grid", {})
    if int(grid.get("num_intervals", 0)) != 20:
        raise ValueError("ensemble requires exactly 20 normalized intervals")
    if grid.get("type") != "normalized_endpoint_inclusive":
        raise ValueError("ensemble requires an endpoint-inclusive normalized grid")
    generation = config.get("instance_generation", {})
    for name, minimum, maximum in (("arms", 10, 200), ("objectives", 2, 10)):
        item = generation.get(name, {})
        if (
            item.get("distribution") != "discrete_uniform_inclusive"
            or int(item.get("minimum", -1)) != minimum
            or int(item.get("maximum", -1)) != maximum
        ):
            raise ValueError(f"ensemble {name} distribution differs from approved design")
    means = generation.get("means", {})
    if (
        means.get("distribution") != "continuous_uniform"
        or float(means.get("minimum", math.nan)) != 0.0
        or float(means.get("maximum", math.nan)) != 1.0
    ):
        raise ValueError("ensemble mean distribution differs from approved design")
    reward = config.get("reward_model", {})
    if reward.get("distribution") != "gaussian_independent_objectives":
        raise ValueError("ensemble requires independent Gaussian rewards")
    if tuple(float(value) for value in reward.get("heterogeneous_std_values", ())) != STD_VALUES:
        raise ValueError("ensemble standard-deviation support differs from approved design")
    if reward.get("heterogeneous_assignment") != "iid_discrete_uniform_per_arm_objective":
        raise ValueError("ensemble standard-deviation assignment differs from approved design")
    if reward.get("require_within_instance_heterogeneity") is not True:
        raise ValueError("ensemble must reject noise-homogeneous instances")
    hardness = config.get("hardness", {})
    if (
        hardness.get("definition") != "mean_only_sum_inverse_squared_gaps"
        or hardness.get("budget_rule") != "round(H(nu))"
        or hardness.get("zero_gap_policy") != "reject_proposal"
        or hardness.get("ceiling_environment") != "EgeExp3"
    ):
        raise ValueError("ensemble hardness semantics differ from approved design")
    if not math.isclose(
        float(hardness.get("ceiling_hardness", math.nan)),
        _final_egeexp3_hardness(),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("ensemble hardness ceiling no longer equals EgeExp3 hardness")
    posterior = config.get("ttpfts", {})
    if posterior != {
        "posterior": "TPosterior",
        "alpha": 0.0,
        "p": 0.5,
        "warmup_pulls_per_arm": 4,
    }:
        raise ValueError("ensemble TTPFTS semantics differ from approved design")
    if config.get("zero_budget_policy") != "native_algorithm":
        raise ValueError("ensemble must preserve native zero-budget behavior")
    if config.get("seeding") != {
        "proposal_streams": "shape_means_stds_separate",
        "algorithm_streams": "independent_deterministic",
    }:
        raise ValueError("ensemble seeding policy differs from approved design")
    bootstrap = config.get("bootstrap", {})
    if int(bootstrap.get("replicates", 0)) < 1000 or not 0.0 < float(bootstrap.get("confidence", 0.0)) < 1.0:
        raise ValueError("ensemble bootstrap configuration is invalid")
    return config


def config_hash(config: dict[str, Any] | None = None) -> str:
    """Return the canonical experiment identity digest."""
    config = load_config() if config is None else config
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def default_output_root(config: dict[str, Any] | None = None) -> Path:
    """Return the config-hash-qualified output root for generated results."""
    return Path(f"results_heterogeneous_ensemble_{config_hash(config)[:12]}")
