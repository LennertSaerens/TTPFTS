"""Immutable configuration, provenance, and write protection for the information-matched figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.ege_doubling.config import (
    canonical_json,
    environment_specs as _doubling_environment_specs,
    sha256_file,
    source_provenance,
)
from experiments.final_config_loader import CONFIG_PATH as FINAL_CONFIG_PATH
from experiments.final_config_loader import config_hash as final_config_hash
from experiments.final_config_loader import load_final_config


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
CONFIG_PATH = ROOT / "config.json"
SCHEMA_VERSION = "information-matched-publication-v1"
OUTPUT_PREFIX = "results_information_matched_"
METRICS = ("jaccard", "bernoulli", "misidentification")
EMPIRICAL_ALGORITHMS = {"D_EGE_SH": "EGE_SH", "D_EGE_SR": "EGE_SR"}
NAIVE_ALGORITHMS = {"D_EGE_SH_N": "D_EGE_SH", "D_EGE_SR_N": "D_EGE_SR"}
PREREGISTERED_ALGORITHMS = ("Uniform", "APE", "TTPFTS_KNOWN_VAR", "TTPFTS_UNKNOWN_VAR")
NEW_ANYTIME_ALGORITHM = "APE_UV"
EXPECTED_SETS = {
    "uv_naive": ["D_EGE_SH_N", "D_EGE_SR_N", "Uniform", "APE_UV", "TTPFTS_UNKNOWN_VAR"],
    "kv_naive": ["D_EGE_SH_N", "D_EGE_SR_N", "Uniform", "APE", "TTPFTS_KNOWN_VAR"],
    "uv_both": ["D_EGE_SH", "D_EGE_SH_N", "D_EGE_SR", "D_EGE_SR_N", "Uniform", "APE_UV", "TTPFTS_UNKNOWN_VAR"],
    "kv_both": ["D_EGE_SH", "D_EGE_SH_N", "D_EGE_SR", "D_EGE_SR_N", "Uniform", "APE", "TTPFTS_KNOWN_VAR"],
}
EXPECTED_LABELS = {
    "D_EGE_SH": "D-EGE-SH", "D_EGE_SR": "D-EGE-SR",
    "D_EGE_SH_N": "D-EGE-SH-N", "D_EGE_SR_N": "D-EGE-SR-N",
    "Uniform": "Uniform", "APE": "APE", "APE_UV": "APE-UV",
    "TTPFTS_KNOWN_VAR": "TTPFTS-KV", "TTPFTS_UNKNOWN_VAR": "TTPFTS-UV",
}
APE_UV_SPEC = {
    "class": "bandits.APE_UV.APEUnknownVarianceBandit",
    "delta": 0.1,
    "warmup_pulls_per_arm": 4,
    "width": "per-objective r_{i,d}(n_i) + r_{j,d}(n_j)",
    "radius": "t_{n-1}^{-1}(1 - alpha_n/2) * s/sqrt(n), alpha_n = 6 delta / (pi^2 K D n^2)",
    "noise_information": "none",
    "seed_namespace": "APE_UV",
}
EMPIRICAL_SPEC = {
    "reference": "Jun & Nowak (2016), DSAR: recommend from all samples collected so far",
    "stages": "fresh native EGE runs with budget 2**(s-1) K for s = 1..S+1; stage S+1 is incomplete and truncated at B",
    "seed_contract": "(master_seed, env|dist, base_algorithm, repetition, 'budget', b_s), identical to the naive stages",
    "within_round_order": "round_robin",
    "time_mapping": "stage s draws at e_{s-1}+1 .. e_{s-1}+a_s and is idle until e_s",
    "recommendation": "Pareto set of empirical means over every sample drawn at time <= t, across stages",
    "unsampled_arm_mean": "zero, identical to the Uniform/APE empirical-mean convention",
    "evaluation_points": "shared 101-point grid together with every completed stage endpoint e_s",
    "sensitivity": "native block order within rounds, reported only",
}


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the configuration and reject any drift from the approved design."""
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected information-matched config schema version")
    final_config = load_final_config()
    if config.get("final_config_sha256") != final_config_hash(final_config):
        raise ValueError("config does not match the committed final experiment config")
    if config.get("final_config_file_sha256") != sha256_file(FINAL_CONFIG_PATH):
        raise ValueError("config does not match final_config.json bytes")
    if (config.get("master_seed"), config.get("num_runs"), config.get("dist")) != (
        final_config["master_seed"], final_config["num_runs"], final_config["reward_model"]["distribution"]
    ):
        raise ValueError("seed, repetition count, or reward model differs from the final experiment")
    if config.get("figure_sets") != EXPECTED_SETS:
        raise ValueError("figure sets differ from the approved design")
    if config.get("labels") != EXPECTED_LABELS:
        raise ValueError("legend labels differ from the approved design")
    if config.get("ape_uv") != APE_UV_SPEC:
        raise ValueError("APE-UV specification differs from the approved design")
    if config.get("empirical_doubling") != EMPIRICAL_SPEC:
        raise ValueError("empirical-mean doubling specification differs from the approved design")
    if not str(config.get("figures_dir", "")).startswith("figures/publication_information_matched"):
        raise ValueError("figures must be written to figures/publication_information_matched")
    return config


def config_hash(config: dict[str, Any] | None = None) -> str:
    config = load_config() if config is None else config
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def default_output_root(config: dict[str, Any] | None = None) -> Path:
    return REPO_ROOT / f"{OUTPUT_PREFIX}{config_hash(config)[:12]}"


def preregistered_dir(config: dict[str, Any]) -> Path:
    return REPO_ROOT / config["source_results_dir"]


def doubling_dir(config: dict[str, Any]) -> Path:
    return REPO_ROOT / config["doubling_results_dir"]


def environment_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Frozen K, D, B, shared 101-point grid, and instance hash for every environment."""
    return _doubling_environment_specs(config)


def environment_spec(name: str, config: dict[str, Any]) -> dict[str, Any]:
    for spec in environment_specs(config):
        if spec["name"] == name:
            return spec
    raise KeyError(f"unknown environment: {name}")


def _verify(root: Path, expected: dict[str, str], what: str) -> dict[str, str]:
    observed = {}
    for relative, digest in sorted(expected.items()):
        actual = sha256_file(root / relative)
        if actual != digest:
            raise ValueError(f"{what} changed: {root / relative}")
        observed[relative] = actual
    return observed


def verify_sources(config: dict[str, Any]) -> None:
    """Every reused source file must be byte-identical to its recorded hash."""
    _verify(preregistered_dir(config), config["source_files"], "preregistered source")
    _verify(doubling_dir(config), config["doubling_files"], "naive doubling source")


def verify_protected(config: dict[str, Any]) -> None:
    """Every existing publication artifact must be byte-identical to its recorded hash."""
    for directory, files in sorted(config["protected_files"].items()):
        _verify(REPO_ROOT / directory, files, "protected publication artifact")


def protected_roots(config: dict[str, Any]) -> list[Path]:
    roots = {(REPO_ROOT / directory).resolve() for directory in config["protected_files"]}
    roots.update({
        preregistered_dir(config).resolve(),
        doubling_dir(config).resolve(),
        (REPO_ROOT / "results").resolve(),
        (REPO_ROOT / "archive_prepatch").resolve(),
        (REPO_ROOT / "archive_provisional").resolve(),
    })
    for candidate in REPO_ROOT.glob("results*"):
        if candidate.is_dir() and not candidate.name.startswith(OUTPUT_PREFIX):
            roots.add(candidate.resolve())
    return sorted(roots)


def assert_safe_output(path: str | Path, config: dict[str, Any]) -> Path:
    """Refuse any output path inside existing results or publication artifacts."""
    resolved = Path(path).resolve()
    for root in protected_roots(config):
        if resolved == root or resolved.is_relative_to(root):
            raise PermissionError(f"refusing to write into protected location {root}: {resolved}")
    return resolved


__all__ = [
    "APE_UV_SPEC", "EMPIRICAL_ALGORITHMS", "EMPIRICAL_SPEC", "METRICS", "NAIVE_ALGORITHMS",
    "NEW_ANYTIME_ALGORITHM", "OUTPUT_PREFIX", "PREREGISTERED_ALGORITHMS", "REPO_ROOT",
    "assert_safe_output", "canonical_json", "config_hash", "default_output_root", "doubling_dir",
    "environment_spec", "environment_specs", "load_config", "preregistered_dir", "sha256_file",
    "source_provenance", "verify_protected", "verify_sources",
]
