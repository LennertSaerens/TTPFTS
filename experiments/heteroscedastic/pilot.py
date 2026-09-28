"""Immutable pilot configuration and generic sharded execution helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from momab_main import EGE_ALGORITHMS, _base_algorithm_name, run_algorithm_sharded
from runner.shards import merge_shards

from .config import canonical_json, load_candidate_config
from .instances import FrozenHeteroscedasticEnvironment, load_frozen_candidates


ROOT = Path(__file__).resolve().parent
PILOT_CONFIG_PATH = ROOT / "pilot_config.json"


def load_pilot_config(path: str | Path = PILOT_CONFIG_PATH) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        config = json.load(f)
    if config.get("schema_version") != "heteroscedastic-pilot-v1":
        raise ValueError("unexpected pilot schema version")
    candidate = load_candidate_config()
    candidate_hash = hashlib.sha256(canonical_json(candidate).encode()).hexdigest()
    if config.get("candidate_config_hash") != candidate_hash:
        raise ValueError("pilot config candidate_config_hash does not match committed candidate config")
    ids = config["hardness_admissibility"]["accepted_instance_ids"]
    if len(ids) != config["hardness_admissibility"]["accepted_count"] or len(ids) != len(set(ids)):
        raise ValueError("pilot accepted instance IDs are malformed")
    return config


def pilot_config_hash(config: dict | None = None) -> str:
    config = load_pilot_config() if config is None else config
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()


def pilot_candidates(config: dict | None = None):
    config = load_pilot_config() if config is None else config
    candidates = load_frozen_candidates(config["hardness_admissibility"]["accepted_instance_ids"])
    limit = float(config["hardness_admissibility"]["maximum_hardness"])
    if any(candidate.hardness.hardness > limit for candidate in candidates):
        raise ValueError("accepted pilot candidate exceeds approved hardness threshold")
    return candidates


def environment_key(candidate) -> str:
    return f"instance_{candidate.instance_id:03d}__{candidate.condition}"


def run_candidate_condition(candidate, config: dict, results_dir: Path, run_dir: Path, n_jobs: int = 1) -> int:
    """Execute one frozen condition through the existing exact-budget shard engine."""
    environment = FrozenHeteroscedasticEnvironment(candidate.means, candidate.stds)
    env_name = environment_key(candidate)
    algorithms = list(config["algorithms"])
    for algorithm in algorithms:
        if algorithm in EGE_ALGORITHMS:
            run_algorithm_sharded(
                str(run_dir), environment, env_name, "gaussian", algorithm, algorithm,
                candidate.budget, list(candidate.grid), int(config["num_runs"]),
                int(config["master_seed"]), n_jobs=n_jobs, is_ege=True,
                ege_func=EGE_ALGORITHMS[algorithm],
            )
        else:
            run_algorithm_sharded(
                str(run_dir), environment, env_name, "gaussian", algorithm,
                _base_algorithm_name(algorithm), candidate.budget, list(candidate.grid),
                int(config["num_runs"]), int(config["master_seed"]), n_jobs=n_jobs,
            )
    return merge_shards(
        str(run_dir),
        env_name,
        str(results_dir / f"{env_name}.csv"),
        expected_grids={algorithm: list(candidate.grid) for algorithm in algorithms},
        validate_metrics=True,
    )
