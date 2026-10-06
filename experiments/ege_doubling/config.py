"""Immutable configuration, provenance, and overwrite protection for the doubling figures."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from experiments.final_config_loader import CONFIG_PATH as FINAL_CONFIG_PATH
from experiments.final_config_loader import config_hash as final_config_hash
from experiments.final_config_loader import load_final_config
from runner.grid import build_normalized_grid

from .schedule import doubling_schedule


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
CONFIG_PATH = ROOT / "config.json"
SCHEMA_VERSION = "ege-doubling-publication-v1"
OUTPUT_PREFIX = "results_ege_doubling_"
METRICS = ("jaccard", "bernoulli", "misidentification")
DOUBLING_ALGORITHMS = {"D_EGE_SH": "EGE_SH", "D_EGE_SR": "EGE_SR"}
ANYTIME_ALGORITHMS = ("Uniform", "APE", "TTPFTS_UNKNOWN_VAR")
DOUBLING_SEMANTICS = {
    "reference": "Jun & Nowak (2016), Anytime Exploration for Multi-armed Bandits using Confidence Information, ICML, Sec. 1",
    "stage_budget": "2**(s-1) * K",
    "stage_endpoint": "(2**s - 1) * K",
    "completed_stages": "every s with (2**s - 1) * K <= B; an incomplete final stage is never run",
    "recommendation": "output of the most recently completed stage, held until the next endpoint",
    "before_first_stage": "undefined; curves start at arm_pull = K",
    "stage_runs": "fresh independent calls to the unchanged native EGE implementation; no sample reuse",
    "budget_accounting": "nominal stage budget; unused native EGE allocation is not carried forward",
    "seed_contract": "existing EGE per-budget seed: (master_seed, env|dist, base_algorithm, repetition, 'budget', b_s)",
    "rendering": "exact step function evaluated at e_s, e_s - 1 (s >= 2), shared grid points >= K, and B",
}


def canonical_json(data: Any) -> str:
    """Serialize data deterministically for configuration hashing."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the doubling configuration and reject any drift from the approved design."""
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected doubling config schema version")
    final_config = load_final_config()
    if config.get("final_config_sha256") != final_config_hash(final_config):
        raise ValueError("doubling config does not match the committed final experiment config")
    if config.get("final_config_file_sha256") != sha256_file(FINAL_CONFIG_PATH):
        raise ValueError("doubling config does not match final_config.json bytes")
    if config.get("master_seed") != final_config["master_seed"]:
        raise ValueError("doubling master seed differs from the final experiment")
    if config.get("num_runs") != final_config["num_runs"]:
        raise ValueError("doubling repetition count differs from the final experiment")
    if config.get("dist") != final_config["reward_model"]["distribution"]:
        raise ValueError("doubling reward distribution differs from the final experiment")
    expected_doubling = {
        name: {"base_algorithm": base, "label": name.replace("_", "-")}
        for name, base in DOUBLING_ALGORITHMS.items()
    }
    if config.get("doubling_algorithms") != expected_doubling:
        raise ValueError("doubling algorithms must be D_EGE_SH/D_EGE_SR built from EGE_SH/EGE_SR")
    if config.get("anytime_algorithms") != {
        "Uniform": "Uniform", "APE": "APE", "TTPFTS_UNKNOWN_VAR": "TTPFTS",
    }:
        raise ValueError("anytime algorithms must be the unchanged publication Uniform/APE/TTPFTS curves")
    if config.get("doubling") != DOUBLING_SEMANTICS:
        raise ValueError("doubling semantics differ from the approved design")
    if not str(config.get("figures_dir", "")).startswith("figures/publication_ege_doubling"):
        raise ValueError("doubling figures must be written to figures/publication_ege_doubling")
    for environment, endpoints in config.get("reference_schedules", {}).items():
        spec = next(e for e in final_config["environments"] if e["name"] == environment)
        actual = [stage.start for stage in doubling_schedule(spec["num_arms"], spec["budget"])]
        if actual != list(endpoints):
            raise ValueError(f"{environment}: doubling endpoints {actual} differ from reference {endpoints}")
    return config


def config_hash(config: dict[str, Any] | None = None) -> str:
    """Return the canonical hash that namespaces every doubling result."""
    config = load_config() if config is None else config
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def default_output_root(config: dict[str, Any] | None = None) -> Path:
    """Return the config-hash-qualified local result root."""
    return REPO_ROOT / f"{OUTPUT_PREFIX}{config_hash(config)[:12]}"


def source_results_dir(config: dict[str, Any]) -> Path:
    return REPO_ROOT / config["source_results_dir"]


def verify_source_hashes(config: dict[str, Any]) -> dict[str, str]:
    """Require every read-only preregistered source file to be byte-identical."""
    root = source_results_dir(config)
    observed = {}
    for relative, expected in sorted(config["source_files"].items()):
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ValueError(f"preregistered source file changed: {root / relative}")
        observed[relative] = actual
    return observed


def verify_protected_hashes(config: dict[str, Any]) -> dict[str, str]:
    """Require the existing publication figures/tables to be byte-identical."""
    root = REPO_ROOT / config["protected_publication_dir"]
    observed = {}
    for name, expected in sorted(config["protected_publication_files"].items()):
        actual = sha256_file(root / name)
        if actual != expected:
            raise ValueError(f"protected publication artifact changed: {root / name}")
        observed[name] = actual
    return observed


def environment_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return each environment's frozen K, D, B, shared grid, and instance hash."""
    specs = []
    for environment in load_final_config()["environments"]:
        name = environment["name"]
        with (source_results_dir(config) / "run" / "manifests" / f"{name}.json").open() as handle:
            manifest = json.load(handle)
        budget = int(manifest["budgets"][name])
        grid = [int(value) for value in manifest["evaluation_grids"][name]]
        if budget != int(environment["budget"]):
            raise ValueError(f"{name}: source manifest budget differs from final config")
        if grid != build_normalized_grid(budget, int(load_final_config()["eval_points"])):
            raise ValueError(f"{name}: source manifest grid differs from the normalized publication grid")
        specs.append({
            "name": name,
            "num_arms": int(environment["num_arms"]),
            "num_objectives": int(environment["num_objectives"]),
            "budget": budget,
            "grid": grid,
            "instance_hash": environment["instance_hash"],
        })
    return specs


def environment_spec(name: str, config: dict[str, Any]) -> dict[str, Any]:
    for spec in environment_specs(config):
        if spec["name"] == name:
            return spec
    raise KeyError(f"unknown environment: {name}")


def protected_roots(config: dict[str, Any]) -> list[Path]:
    """Directories that this package must never write into."""
    roots = {
        (REPO_ROOT / config["protected_publication_dir"]).resolve(),
        source_results_dir(config).resolve(),
        (REPO_ROOT / "results").resolve(),
        (REPO_ROOT / "archive_prepatch").resolve(),
        (REPO_ROOT / "archive_provisional").resolve(),
    }
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


def source_provenance(allow_dirty: bool = False) -> dict[str, Any]:
    """Return the source commit and refuse unattributable uncommitted code."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, capture_output=True, check=True
    ).stdout.splitlines()
    source_changes, bytecode_changes = [], []
    for entry in status:
        path = entry[3:]
        (bytecode_changes if path.endswith(".pyc") and "__pycache__/" in path else source_changes).append(entry)
    if source_changes and not allow_dirty:
        raise RuntimeError("refusing to run from an uncommitted source tree: " + "; ".join(source_changes))
    return {
        "source_commit": commit,
        "source_dirty": bool(source_changes),
        "source_changes": source_changes,
        "ignored_generated_bytecode_changes": bytecode_changes,
    }
