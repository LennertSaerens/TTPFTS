"""Freeze and verify the accepted heterogeneous-noise MOMAB cohort."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from environments.BaseEnvironment import BaseEnvironment
from environments.distributions import GaussianReward
from environments.hardness import HardnessResult, assert_non_degenerate, compute_pareto_mask
from runner.grid import build_normalized_grid
from runner.instance import compute_instance_hash
from runner.seeding import derive_seed

from .config import config_hash, load_config


ROOT = Path(__file__).resolve().parent
INSTANCES_DIR = ROOT / "instances"
INSTANCES_CSV = INSTANCES_DIR / "accepted_instances.csv"
LEDGER_CSV = INSTANCES_DIR / "proposal_ledger.csv"
MANIFEST_PATH = INSTANCES_DIR / "manifest.json"
CHECKSUMS_PATH = INSTANCES_DIR / "CHECKSUMS.sha256"


class EnsembleInstanceError(RuntimeError):
    """Raised when an accepted cohort artifact violates its frozen contract."""


@dataclass(frozen=True)
class FrozenInstance:
    """One accepted immutable heterogeneous-noise instance."""

    instance_id: int
    proposal_id: int
    means: np.ndarray
    stds: np.ndarray
    pareto_indices: np.ndarray
    hardness: HardnessResult
    budget: int
    grid: tuple[int, ...]
    instance_hash: str

    @property
    def name(self) -> str:
        return f"instance_{self.instance_id:04d}"

    @property
    def num_arms(self) -> int:
        return int(self.means.shape[0])

    @property
    def num_objectives(self) -> int:
        return int(self.means.shape[1])

    @property
    def noise_diversity(self) -> int:
        return int(np.unique(self.stds).size)


class FrozenHeterogeneousEnvironment(BaseEnvironment):
    """Immutable Gaussian environment with per-arm-objective standard deviations."""

    def __init__(self, means: np.ndarray, stds: np.ndarray) -> None:
        means = np.asarray(means, dtype=np.float64)
        stds = np.asarray(stds, dtype=np.float64)
        if means.ndim != 2 or means.shape != stds.shape:
            raise ValueError(f"means/stds must share a 2-D shape, got {means.shape}/{stds.shape}")
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(stds)) or np.any(stds <= 0.0):
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
        """Frozen instances never redraw means or standard deviations."""


def _rng(master_seed: int, *parts: object) -> np.random.Generator:
    seed = derive_seed(master_seed, *parts)
    if seed is None:
        raise EnsembleInstanceError("ensemble generation requires a concrete master seed")
    return np.random.default_rng(seed)


def _proposal_arrays(proposal_id: int, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Generate one proposal from streams that are independent by construction."""
    if proposal_id < 0:
        raise ValueError("proposal_id must be non-negative")
    master_seed = int(config["master_seed"])
    generation = config["instance_generation"]
    shape_rng = _rng(master_seed, "heterogeneous-ensemble", proposal_id, "shape")
    num_arms = int(
        shape_rng.integers(
            int(generation["arms"]["minimum"]), int(generation["arms"]["maximum"]) + 1
        )
    )
    num_objectives = int(
        shape_rng.integers(
            int(generation["objectives"]["minimum"]), int(generation["objectives"]["maximum"]) + 1
        )
    )
    means_rng = _rng(master_seed, "heterogeneous-ensemble", proposal_id, "means")
    means = means_rng.uniform(
        float(generation["means"]["minimum"]),
        float(generation["means"]["maximum"]),
        size=(num_arms, num_objectives),
    ).astype(np.float64)
    std_rng = _rng(master_seed, "heterogeneous-ensemble", proposal_id, "stds")
    stds = std_rng.choice(
        np.asarray(config["reward_model"]["heterogeneous_std_values"], dtype=np.float64),
        size=means.shape,
        replace=True,
    ).astype(np.float64)
    return means, stds


def evaluate_proposal(proposal_id: int, config: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """Evaluate one proposal without reference to earlier acceptance decisions."""
    config = load_config() if config is None else config
    means, stds = _proposal_arrays(proposal_id, config)
    record: dict[str, Any] = {
        "proposal_id": proposal_id,
        "num_arms": int(means.shape[0]),
        "num_objectives": int(means.shape[1]),
        "noise_diversity": int(np.unique(stds).size),
    }
    if record["noise_diversity"] < 2:
        record["reason"] = "insufficient_noise_heterogeneity"
        return "reject", record
    pareto_indices = np.flatnonzero(compute_pareto_mask(means))
    try:
        hardness = assert_non_degenerate(
            f"heterogeneous_ensemble_proposal_{proposal_id:06d}",
            means,
            pareto_indices,
            require_tie_free=True,
        )
    except ValueError as exc:
        record["reason"] = "nonfinite_or_degenerate_hardness"
        record["detail"] = str(exc)
        return "reject", record
    record.update(
        {
            "pareto_size": int(len(pareto_indices)),
            "minimum_gap": float(np.min(hardness.gaps_direct)),
            "hardness": float(hardness.hardness),
        }
    )
    if hardness.hardness > float(config["hardness"]["ceiling_hardness"]):
        record["reason"] = "hardness_above_egeexp3_ceiling"
        return "reject", record
    budget = int(round(hardness.hardness))
    grid = tuple(build_normalized_grid(budget, int(config["evaluation_grid"]["num_intervals"])))
    if len(grid) != 21 or grid[0] != 0 or grid[-1] != budget:
        raise EnsembleInstanceError(f"proposal {proposal_id} has invalid grid {grid}")
    environment = FrozenHeterogeneousEnvironment(means, stds)
    record.update(
        {
            "reason": "accepted",
            "budget": budget,
            "grid": grid,
            "means": means,
            "stds": stds,
            "pareto_indices": pareto_indices,
            "hardness_result": hardness,
            "instance_hash": compute_instance_hash(environment),
        }
    )
    return "accept", record


def generate_accepted_instances(config: dict[str, Any] | None = None) -> tuple[list[FrozenInstance], list[dict[str, Any]]]:
    """Return the deterministic ordered prefix of 1,000 eligible proposals."""
    config = load_config() if config is None else config
    target = int(config["cohort"]["accepted_instances"])
    instances: list[FrozenInstance] = []
    ledger: list[dict[str, Any]] = []
    for proposal_id in range(int(config["cohort"]["max_proposals"])):
        status, record = evaluate_proposal(proposal_id, config)
        if status == "accept":
            instance_id = len(instances)
            record["instance_id"] = instance_id
            instances.append(
                FrozenInstance(
                    instance_id=instance_id,
                    proposal_id=proposal_id,
                    means=record["means"],
                    stds=record["stds"],
                    pareto_indices=record["pareto_indices"],
                    hardness=record["hardness_result"],
                    budget=record["budget"],
                    grid=record["grid"],
                    instance_hash=record["instance_hash"],
                )
            )
        ledger.append(_ledger_row(record))
        if len(instances) == target:
            return instances, ledger
    raise EnsembleInstanceError(
        f"only {len(instances)} of {target} accepted instances found before max_proposals="
        f"{config['cohort']['max_proposals']}"
    )


def _ledger_row(record: dict[str, Any]) -> dict[str, Any]:
    """Remove arrays/state from a proposal record before durable ledger writing."""
    return {
        key: value
        for key, value in record.items()
        if key not in {"means", "stds", "pareto_indices", "hardness_result", "grid"}
    }


def _manifest_entry(instance: FrozenInstance) -> dict[str, Any]:
    return {
        "instance_id": instance.instance_id,
        "proposal_id": instance.proposal_id,
        "num_arms": instance.num_arms,
        "num_objectives": instance.num_objectives,
        "pareto_indices": instance.pareto_indices.tolist(),
        "gaps": [float(value) for value in instance.hardness.gaps_direct],
        "minimum_gap": float(np.min(instance.hardness.gaps_direct)),
        "hardness": float(instance.hardness.hardness),
        "budget": instance.budget,
        "evaluation_grid": list(instance.grid),
        "noise_diversity": instance.noise_diversity,
        "std_counts": {
            f"{value:.1f}": int(np.count_nonzero(instance.stds == value))
            for value in (0.1, 0.2, 0.3, 0.4, 0.5)
        },
        "instance_hash": instance.instance_hash,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def freeze_instances(config: dict[str, Any] | None = None) -> tuple[list[FrozenInstance], dict[str, Any], list[dict[str, Any]]]:
    """Generate and checksum-write the complete accepted cohort and proposal ledger."""
    config = load_config() if config is None else config
    instances, ledger = generate_accepted_instances(config)
    _write_csv(
        INSTANCES_CSV,
        ["instance_id", "proposal_id", "arm", "objective", "mean", "std"],
        (
            {
                "instance_id": instance.instance_id,
                "proposal_id": instance.proposal_id,
                "arm": arm,
                "objective": objective,
                "mean": format(instance.means[arm, objective], ".17g"),
                "std": format(instance.stds[arm, objective], ".17g"),
            }
            for instance in instances
            for arm, objective in np.ndindex(instance.means.shape)
        ),
    )
    ledger_fields = [
        "proposal_id", "instance_id", "reason", "num_arms", "num_objectives",
        "noise_diversity", "pareto_size", "minimum_gap", "hardness", "budget", "instance_hash", "detail",
    ]
    _write_csv(LEDGER_CSV, ledger_fields, ({field: row.get(field, "") for field in ledger_fields} for row in ledger))
    manifest = {
        "schema_version": "heterogeneous-ensemble-frozen-instances-v1",
        "ensemble_config_hash": config_hash(config),
        "ensemble_config_file": "config.json",
        "accepted_count": len(instances),
        "last_evaluated_proposal_id": ledger[-1]["proposal_id"],
        "proposal_count": len(ledger),
        "instances": [_manifest_entry(instance) for instance in instances],
    }
    manifest["artifacts"] = {
        "accepted_instances_csv_sha256": _sha256(INSTANCES_CSV),
        "proposal_ledger_csv_sha256": _sha256(LEDGER_CSV),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    CHECKSUMS_PATH.write_text(
        f"{_sha256(INSTANCES_CSV)}  {INSTANCES_CSV.name}\n"
        f"{_sha256(LEDGER_CSV)}  {LEDGER_CSV.name}\n"
        f"{_sha256(MANIFEST_PATH)}  {MANIFEST_PATH.name}\n",
        encoding="utf-8",
    )
    return instances, manifest, ledger


def _read_checksums() -> dict[str, str]:
    if not CHECKSUMS_PATH.is_file():
        raise EnsembleInstanceError(f"missing {CHECKSUMS_PATH}")
    checksums = {}
    for line in CHECKSUMS_PATH.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    return checksums


def _load_manifest() -> dict[str, Any]:
    if not all(path.is_file() for path in (INSTANCES_CSV, LEDGER_CSV, MANIFEST_PATH, CHECKSUMS_PATH)):
        raise EnsembleInstanceError("frozen ensemble artifacts are missing; run generate_instances.py")
    checksums = _read_checksums()
    for path in (INSTANCES_CSV, LEDGER_CSV, MANIFEST_PATH):
        if checksums.get(path.name) != _sha256(path):
            raise EnsembleInstanceError(f"checksum mismatch for {path.name}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    config = load_config()
    if manifest.get("ensemble_config_hash") != config_hash(config):
        raise EnsembleInstanceError("frozen instances do not match ensemble config")
    if int(manifest.get("accepted_count", 0)) != int(config["cohort"]["accepted_instances"]):
        raise EnsembleInstanceError("accepted instance count differs from config")
    return manifest


def load_frozen_instances(instance_ids: Iterable[int] | None = None) -> list[FrozenInstance]:
    """Load checksum-verified frozen instances without regenerating proposals."""
    manifest = _load_manifest()
    wanted = None if instance_ids is None else {int(value) for value in instance_ids}
    metadata = {
        int(item["instance_id"]): item
        for item in manifest["instances"]
        if wanted is None or int(item["instance_id"]) in wanted
    }
    grouped: dict[int, list[dict[str, str]]] = {}
    with INSTANCES_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            instance_id = int(row["instance_id"])
            if wanted is None or instance_id in wanted:
                grouped.setdefault(instance_id, []).append(row)
    if set(grouped) != set(metadata):
        raise EnsembleInstanceError("instance CSV and manifest coverage differ")
    instances: list[FrozenInstance] = []
    for instance_id, rows in sorted(grouped.items()):
        item = metadata[instance_id]
        num_arms, num_objectives = int(item["num_arms"]), int(item["num_objectives"])
        if len(rows) != num_arms * num_objectives:
            raise EnsembleInstanceError(f"instance {instance_id} has incomplete matrix data")
        means = np.empty((num_arms, num_objectives), dtype=np.float64)
        stds = np.empty_like(means)
        proposal_ids = set()
        for row in rows:
            arm, objective = int(row["arm"]), int(row["objective"])
            means[arm, objective] = float(row["mean"])
            stds[arm, objective] = float(row["std"])
            proposal_ids.add(int(row["proposal_id"]))
        if proposal_ids != {int(item["proposal_id"])}:
            raise EnsembleInstanceError(f"instance {instance_id} proposal ID mismatch")
        pareto_indices = np.asarray(item["pareto_indices"], dtype=int)
        try:
            hardness = assert_non_degenerate(
                f"heterogeneous_ensemble_instance_{instance_id:04d}",
                means,
                pareto_indices,
                require_tie_free=True,
            )
        except ValueError as exc:
            raise EnsembleInstanceError(str(exc)) from exc
        if hardness.hardness != float(item["hardness"]):
            raise EnsembleInstanceError(f"instance {instance_id} hardness differs from manifest")
        grid = tuple(int(value) for value in item["evaluation_grid"])
        budget = int(item["budget"])
        if tuple(build_normalized_grid(budget, 20)) != grid:
            raise EnsembleInstanceError(f"instance {instance_id} grid differs from frozen rule")
        instance_hash = compute_instance_hash(FrozenHeterogeneousEnvironment(means, stds))
        if instance_hash != item["instance_hash"]:
            raise EnsembleInstanceError(f"instance {instance_id} hash differs from manifest")
        instances.append(
            FrozenInstance(
                instance_id=instance_id,
                proposal_id=int(item["proposal_id"]),
                means=means,
                stds=stds,
                pareto_indices=pareto_indices,
                hardness=hardness,
                budget=budget,
                grid=grid,
                instance_hash=instance_hash,
            )
        )
    _validate_instances(instances, manifest, require_full=wanted is None)
    return instances


def _validate_instances(
    instances: list[FrozenInstance], manifest: dict[str, Any], *, require_full: bool
) -> None:
    config = load_config()
    if require_full and len(instances) != int(config["cohort"]["accepted_instances"]):
        raise EnsembleInstanceError("frozen cohort has an unexpected instance count")
    proposal_ids = [instance.proposal_id for instance in instances]
    if proposal_ids != sorted(proposal_ids) or len(set(proposal_ids)) != len(proposal_ids):
        raise EnsembleInstanceError("accepted proposal IDs must be unique and ordered")
    if any(instance.hardness.hardness > float(config["hardness"]["ceiling_hardness"]) for instance in instances):
        raise EnsembleInstanceError("frozen cohort includes an over-threshold instance")
    if any(instance.noise_diversity < 2 for instance in instances):
        raise EnsembleInstanceError("frozen cohort includes a noise-homogeneous instance")
    if require_full and [instance.instance_id for instance in instances] != list(range(len(instances))):
        raise EnsembleInstanceError("frozen instance IDs must be contiguous from zero")
    if require_full and int(manifest["proposal_count"]) != int(manifest["last_evaluated_proposal_id"]) + 1:
        raise EnsembleInstanceError("proposal ledger must cover every ID through its final proposal")
