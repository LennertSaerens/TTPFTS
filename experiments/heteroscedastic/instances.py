"""Frozen paired random MOMAB instances for the heteroscedastic study.

The candidate ensemble is generated once from the committed configuration and
serialized with round-trip-safe float text.  Each mean matrix has two paired
reward conditions: i.i.d. discrete per-coordinate standard deviations and its
RMS-matched homogeneous counterfactual.  Means, Pareto set, gaps, hardness,
and budget are deliberately shared by the pair.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from environments.BaseEnvironment import BaseEnvironment
from environments.distributions import GaussianReward
from environments.hardness import HardnessResult, assert_non_degenerate, compute_pareto_mask
from runner.grid import build_normalized_grid
from runner.instance import compute_instance_hash
from runner.seeding import derive_seed

from .config import config_hash, load_candidate_config


ROOT = Path(__file__).resolve().parent
INSTANCES_DIR = ROOT / "instances"
CANDIDATES_CSV = INSTANCES_DIR / "candidates.csv"
MANIFEST_PATH = INSTANCES_DIR / "manifest.json"
CHECKSUMS_PATH = INSTANCES_DIR / "CHECKSUMS.sha256"


class HeteroscedasticInstanceError(RuntimeError):
    """Raised when immutable candidate data fails a scientific integrity check."""


@dataclass(frozen=True)
class FrozenCandidate:
    instance_id: int
    condition: str
    means: np.ndarray
    stds: np.ndarray
    pareto_indices: np.ndarray
    hardness: HardnessResult
    budget: int
    grid: tuple[int, ...]

    @property
    def rms_std(self) -> float:
        return float(np.sqrt(np.mean(self.stds**2)))

    @property
    def instance_name(self) -> str:
        return f"instance_{self.instance_id:03d}/{self.condition}"


class FrozenHeteroscedasticEnvironment(BaseEnvironment):
    """An immutable Gaussian MOMAB environment with per-coordinate stds."""

    def __init__(self, means: np.ndarray, stds: np.ndarray) -> None:
        means = np.asarray(means, dtype=np.float64)
        stds = np.asarray(stds, dtype=np.float64)
        if means.ndim != 2 or means.shape != stds.shape:
            raise ValueError(f"means/stds must have identical 2-D shape, got {means.shape}/{stds.shape}")
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(stds)) or np.any(stds <= 0):
            raise ValueError("means must be finite and standard deviations finite positive values")
        pareto_indices = np.flatnonzero(compute_pareto_mask(means))
        distributions = [GaussianReward(means[arm], stds[arm]) for arm in range(means.shape[0])]
        super().__init__(
            num_arms=means.shape[0],
            num_objectives=means.shape[1],
            pareto_indices=pareto_indices,
            distributions=distributions,
        )

    def reset(self) -> None:
        """Instances are frozen: a reset must not redraw means or variances."""


def _rng(master_seed: int, *parts: object) -> np.random.Generator:
    seed = derive_seed(master_seed, *parts)
    if seed is None:  # configuration always supplies a concrete master seed
        raise HeteroscedasticInstanceError("candidate generation requires a master seed")
    return np.random.default_rng(seed)


def generate_candidate_pair(instance_id: int, config: dict | None = None) -> tuple[FrozenCandidate, FrozenCandidate]:
    """Generate one deterministic paired mean/noise instance from config."""
    cfg = load_candidate_config() if config is None else config
    master_seed = int(cfg["master_seed"])
    arms_cfg, obj_cfg, means_cfg = cfg["arms"], cfg["objectives"], cfg["means"]
    shape_rng = _rng(master_seed, "heteroscedastic", "candidate", instance_id, "shape")
    num_arms = int(shape_rng.integers(int(arms_cfg["minimum"]), int(arms_cfg["maximum"]) + 1))
    num_objectives = int(shape_rng.integers(int(obj_cfg["minimum"]), int(obj_cfg["maximum"]) + 1))
    means_rng = _rng(master_seed, "heteroscedastic", "candidate", instance_id, "means")
    means = means_rng.uniform(
        float(means_cfg["minimum"]), float(means_cfg["maximum"]), size=(num_arms, num_objectives)
    ).astype(np.float64)
    std_values = np.asarray(cfg["reward_model"]["heterogeneous_std_values"], dtype=np.float64)
    std_rng = _rng(master_seed, "heteroscedastic", "candidate", instance_id, "heterogeneous-stds")
    heterogeneous_stds = std_rng.choice(std_values, size=means.shape, replace=True)
    control_rms = float(np.sqrt(np.mean(heterogeneous_stds**2)))
    homoscedastic_stds = np.full(means.shape, control_rms, dtype=np.float64)

    reference_pareto = np.flatnonzero(compute_pareto_mask(means))
    hardness = assert_non_degenerate(
        f"candidate_{instance_id:03d}", means, reference_pareto, require_tie_free=True
    )
    if not np.isfinite(hardness.hardness):
        raise HeteroscedasticInstanceError(f"candidate {instance_id} has non-finite hardness")
    budget = int(round(hardness.hardness))
    if budget < 1:
        raise HeteroscedasticInstanceError(f"candidate {instance_id} has invalid rounded budget {budget}")
    grid = tuple(build_normalized_grid(budget, int(cfg["evaluation_grid"]["num_intervals"])))
    if len(grid) != 11 or grid[0] != 0 or grid[-1] != budget:
        raise HeteroscedasticInstanceError(f"candidate {instance_id} has invalid normalized grid {grid}")

    common = dict(
        instance_id=instance_id,
        means=means,
        pareto_indices=reference_pareto,
        hardness=hardness,
        budget=budget,
        grid=grid,
    )
    return (
        FrozenCandidate(condition="heterogeneous", stds=heterogeneous_stds, **common),
        FrozenCandidate(condition="homoscedastic_rms", stds=homoscedastic_stds, **common),
    )


def generate_candidates(config: dict | None = None) -> list[FrozenCandidate]:
    cfg = load_candidate_config() if config is None else config
    candidates: list[FrozenCandidate] = []
    for instance_id in range(int(cfg["candidate_instances"])):
        candidates.extend(generate_candidate_pair(instance_id, cfg))
    return candidates


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_manifest_entry(candidate: FrozenCandidate) -> dict:
    env = FrozenHeteroscedasticEnvironment(candidate.means, candidate.stds)
    return {
        "instance_id": candidate.instance_id,
        "condition": candidate.condition,
        "num_arms": int(candidate.means.shape[0]),
        "num_objectives": int(candidate.means.shape[1]),
        "pareto_indices": candidate.pareto_indices.tolist(),
        "gaps": [float(value) for value in candidate.hardness.gaps_direct],
        "minimum_gap": float(candidate.hardness.gaps_direct.min()),
        "hardness": float(candidate.hardness.hardness),
        "budget": candidate.budget,
        "evaluation_grid": list(candidate.grid),
        "rms_std": candidate.rms_std,
        "std_values": sorted(float(value) for value in np.unique(candidate.stds)),
        "instance_hash": compute_instance_hash(env),
    }


def freeze_candidates(config: dict | None = None, *, write: bool = True) -> tuple[list[FrozenCandidate], dict]:
    """Generate all candidates and optionally write immutable data/checksums."""
    cfg = load_candidate_config() if config is None else config
    candidates = generate_candidates(cfg)
    manifest = {
        "schema_version": "heteroscedastic-frozen-instances-v1",
        "candidate_config_hash": config_hash(cfg),
        "candidate_config_file": "candidate_config.json",
        "conditions": cfg["conditions"],
        "candidates": [_candidate_manifest_entry(candidate) for candidate in candidates],
    }
    if not write:
        return candidates, manifest

    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)
    with CANDIDATES_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["instance_id", "condition", "arm", "objective", "mean", "std"]
        )
        writer.writeheader()
        for candidate in candidates:
            for arm, objective in np.ndindex(candidate.means.shape):
                writer.writerow(
                    {
                        "instance_id": candidate.instance_id,
                        "condition": candidate.condition,
                        "arm": arm,
                        "objective": objective,
                        "mean": format(candidate.means[arm, objective], ".17g"),
                        "std": format(candidate.stds[arm, objective], ".17g"),
                    }
                )
    manifest["artifacts"] = {"candidates_csv_sha256": _sha256(CANDIDATES_CSV)}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    CHECKSUMS_PATH.write_text(
        f"{_sha256(CANDIDATES_CSV)}  {CANDIDATES_CSV.name}\n"
        f"{_sha256(MANIFEST_PATH)}  {MANIFEST_PATH.name}\n",
        encoding="utf-8",
    )
    return candidates, manifest


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists() or not CANDIDATES_CSV.exists() or not CHECKSUMS_PATH.exists():
        raise HeteroscedasticInstanceError("frozen candidate artifacts are missing; run generate_candidates.py")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = manifest.get("artifacts", {}).get("candidates_csv_sha256")
    actual = _sha256(CANDIDATES_CSV)
    if expected != actual:
        raise HeteroscedasticInstanceError(
            f"candidate CSV checksum mismatch: expected {expected}, got {actual}"
        )
    return manifest


def load_frozen_candidates(instance_ids: Iterable[int] | None = None) -> list[FrozenCandidate]:
    """Load and checksum-verify frozen candidates without regenerating them."""
    manifest = _load_manifest()
    wanted = None if instance_ids is None else {int(value) for value in instance_ids}
    grouped: dict[tuple[int, str], list[dict[str, str]]] = {}
    with CANDIDATES_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (int(row["instance_id"]), row["condition"])
            if wanted is None or key[0] in wanted:
                grouped.setdefault(key, []).append(row)
    metadata = {
        (int(entry["instance_id"]), entry["condition"]): entry
        for entry in manifest["candidates"]
        if wanted is None or int(entry["instance_id"]) in wanted
    }
    loaded: list[FrozenCandidate] = []
    for key, rows in sorted(grouped.items()):
        entry = metadata.get(key)
        if entry is None:
            raise HeteroscedasticInstanceError(f"CSV candidate {key} is absent from manifest")
        num_arms, num_objectives = int(entry["num_arms"]), int(entry["num_objectives"])
        if len(rows) != num_arms * num_objectives:
            raise HeteroscedasticInstanceError(f"candidate {key} has {len(rows)} values, expected {num_arms * num_objectives}")
        means = np.empty((num_arms, num_objectives), dtype=np.float64)
        stds = np.empty_like(means)
        for row in rows:
            arm, objective = int(row["arm"]), int(row["objective"])
            means[arm, objective] = float(row["mean"])
            stds[arm, objective] = float(row["std"])
        pareto_indices = np.asarray(entry["pareto_indices"], dtype=int)
        hardness = assert_non_degenerate(
            f"candidate_{key[0]:03d}", means, pareto_indices, require_tie_free=True
        )
        if not np.isclose(hardness.hardness, float(entry["hardness"]), rtol=0.0, atol=0.0):
            raise HeteroscedasticInstanceError(f"candidate {key} hardness differs from manifest")
        candidate = FrozenCandidate(
            instance_id=key[0],
            condition=key[1],
            means=means,
            stds=stds,
            pareto_indices=pareto_indices,
            hardness=hardness,
            budget=int(entry["budget"]),
            grid=tuple(int(value) for value in entry["evaluation_grid"]),
        )
        actual_hash = compute_instance_hash(FrozenHeteroscedasticEnvironment(means, stds))
        if actual_hash != entry["instance_hash"]:
            raise HeteroscedasticInstanceError(f"candidate {key} instance hash differs from manifest")
        loaded.append(candidate)
    if set(grouped) != set(metadata):
        raise HeteroscedasticInstanceError("manifest and CSV candidate coverage differ")
    _validate_pairs(loaded)
    return loaded


def _validate_pairs(candidates: list[FrozenCandidate]) -> None:
    """Assert every loaded candidate has exactly the intended paired conditions."""
    by_id: dict[int, dict[str, FrozenCandidate]] = {}
    for candidate in candidates:
        by_id.setdefault(candidate.instance_id, {})[candidate.condition] = candidate
    for instance_id, pair in by_id.items():
        if set(pair) != {"heterogeneous", "homoscedastic_rms"}:
            raise HeteroscedasticInstanceError(
                f"candidate {instance_id} has conditions {sorted(pair)}, expected paired conditions"
            )
        heterogeneous = pair["heterogeneous"]
        control = pair["homoscedastic_rms"]
        if not np.array_equal(heterogeneous.means, control.means):
            raise HeteroscedasticInstanceError(f"candidate {instance_id} pair has different mean matrices")
        if not np.array_equal(heterogeneous.pareto_indices, control.pareto_indices):
            raise HeteroscedasticInstanceError(f"candidate {instance_id} pair has different Pareto sets")
        if heterogeneous.budget != control.budget or heterogeneous.grid != control.grid:
            raise HeteroscedasticInstanceError(f"candidate {instance_id} pair has different budget/grid")
        if heterogeneous.hardness.hardness != control.hardness.hardness:
            raise HeteroscedasticInstanceError(f"candidate {instance_id} pair has different hardness")
        if not np.all(np.isin(heterogeneous.stds, [0.1, 0.2, 0.3, 0.4, 0.5])):
            raise HeteroscedasticInstanceError(f"candidate {instance_id} heterogeneous std outside configured grid")
        if np.unique(control.stds).size != 1:
            raise HeteroscedasticInstanceError(f"candidate {instance_id} control is not homoscedastic")
        rms = np.sqrt(np.mean(heterogeneous.stds**2))
        if rms != control.stds[0, 0]:
            raise HeteroscedasticInstanceError(f"candidate {instance_id} control does not RMS-match heterogeneous stds")
