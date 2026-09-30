"""Generate, freeze, reload, and audit the paired custom noise/gap environments."""

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

from .config import CONDITIONS, STD_VALUES, config_hash, load_config


ROOT = Path(__file__).resolve().parent
INSTANCES_DIR = ROOT / "instances"
ARMS_CSV = INSTANCES_DIR / "arms.csv"
MANIFEST_PATH = INSTANCES_DIR / "manifest.json"
CHECKSUMS_PATH = INSTANCES_DIR / "CHECKSUMS.sha256"
REPORTS_DIR = ROOT / "reports"


class GapNoiseInstanceError(RuntimeError):
    """Raised when a frozen custom environment violates its specification."""


@dataclass(frozen=True)
class FrozenCondition:
    """One immutable variance assignment over the shared custom mean matrix."""

    condition: str
    means: np.ndarray
    stds: np.ndarray
    gaps: np.ndarray
    noise_ranks: np.ndarray
    pareto_indices: np.ndarray
    hardness: HardnessResult
    budget: int
    grid: tuple[int, ...]
    instance_hash: str

    @property
    def name(self) -> str:
        return self.condition

    @property
    def num_arms(self) -> int:
        return int(self.means.shape[0])

    @property
    def num_objectives(self) -> int:
        return int(self.means.shape[1])

    @property
    def arm_stds(self) -> np.ndarray:
        return self.stds[:, 0]


class PairedGaussianEnvironment(BaseEnvironment):
    """Immutable Gaussian environment with condition-paired arm/objective streams."""

    def __init__(self, means: np.ndarray, stds: np.ndarray, reward_seed: int | None = None) -> None:
        means, stds = np.asarray(means, dtype=np.float64), np.asarray(stds, dtype=np.float64)
        if means.shape != (50, 2) or stds.shape != means.shape:
            raise ValueError(f"expected 50x2 means/stds, received {means.shape}/{stds.shape}")
        if not np.all(np.isfinite(means)) or np.any(means < 0.0) or np.any(means > 1.0):
            raise ValueError("means must be finite and contained in [0, 1]")
        if not np.all(np.isfinite(stds)) or np.any(stds <= 0.0):
            raise ValueError("standard deviations must be finite and positive")
        pareto = np.flatnonzero(compute_pareto_mask(means))
        super().__init__(
            num_arms=50,
            num_objectives=2,
            pareto_indices=pareto,
            distributions=[GaussianReward(means[arm], stds[arm]) for arm in range(50)],
        )
        self.reward_draws = 0
        self._reward_seed = reward_seed
        self._streams: list[np.random.Generator] | None = None
        if reward_seed is not None:
            self._streams = [
                np.random.default_rng(derive_seed(reward_seed, "gap-noise-reward", arm))
                for arm in range(self.num_arms)
            ]

    def reset(self) -> None:
        """Frozen means/noise never redraw; per-run streams are constructed explicitly."""

    def pull_arm(self, arm: int) -> np.ndarray:
        return self.sample([arm])[0]

    def sample(self, arms: Iterable[int]) -> np.ndarray:
        arm_array = np.asarray(list(arms), dtype=int)
        if arm_array.ndim != 1 or np.any(arm_array < 0) or np.any(arm_array >= self.num_arms):
            raise ValueError("sample arms must be valid one-dimensional arm indices")
        self.reward_draws += int(arm_array.size)
        if self._streams is None:
            return super().sample(arm_array)
        rewards = np.empty((len(arm_array), self.num_objectives), dtype=np.float64)
        for arm in np.unique(arm_array):
            positions = np.flatnonzero(arm_array == arm)
            latent = self._streams[int(arm)].normal(size=(len(positions), self.num_objectives))
            rewards[positions] = self.arms[int(arm)] + latent * self.stds[int(arm)]
        return rewards

    def export_stream_state(self) -> dict[str, Any]:
        """Return trusted local replay state for a checkpointed anytime task."""
        if self._streams is None:
            raise RuntimeError("cannot checkpoint an environment without paired reward streams")
        return {
            "reward_draws": self.reward_draws,
            "bit_generators": [stream.bit_generator.state for stream in self._streams],
        }

    def restore_stream_state(self, state: dict[str, Any]) -> None:
        """Restore a state emitted by :meth:`export_stream_state`."""
        if self._streams is None:
            raise RuntimeError("cannot restore an environment without paired reward streams")
        generator_states = state.get("bit_generators")
        if not isinstance(generator_states, list) or len(generator_states) != self.num_arms:
            raise ValueError("invalid paired reward-stream state")
        self.reward_draws = int(state["reward_draws"])
        if self.reward_draws < 0:
            raise ValueError("negative paired reward-draw count")
        for stream, generator_state in zip(self._streams, generator_states):
            stream.bit_generator.state = generator_state


def _mean_matrix(config: dict[str, Any]) -> np.ndarray:
    environment = config["environment"]
    start, step = np.asarray(environment["anchor_start"], dtype=float), np.asarray(environment["anchor_step"], dtype=float)
    anchors = np.asarray([start + index * step for index in range(environment["num_anchors"])], dtype=np.float64)
    children = []
    for anchor in anchors:
        for gap in environment["child_nominal_gaps"]:
            for epsilon in environment["child_epsilons"]:
                children.append(anchor - np.array([gap + epsilon, gap + 2.0 * epsilon]))
    means = np.vstack((anchors, np.asarray(children, dtype=np.float64)))
    if means.shape != (50, 2):
        raise GapNoiseInstanceError(f"custom construction produced {means.shape}, expected (50, 2)")
    return means


def _stable_gap_order(gaps: np.ndarray) -> np.ndarray:
    return np.asarray(sorted(range(len(gaps)), key=lambda arm: (round(float(gaps[arm]), 12), arm)), dtype=int)


def _standard_deviations(condition: str, gaps: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition}")
    noise = config["noise_assignment"]
    values = tuple(float(value) for value in noise["standard_deviations"])
    per_value = int(noise["arms_per_standard_deviation"])
    order = _stable_gap_order(gaps)
    ascending = np.repeat(np.asarray(values, dtype=float), per_value)
    if len(ascending) != len(gaps):
        raise GapNoiseInstanceError("standard-deviation assignment does not cover every arm")
    arm_stds = np.empty(len(gaps), dtype=float)
    arm_stds[order] = ascending if condition == "large_gap_noisy" else ascending[::-1]
    ranks = np.empty(len(gaps), dtype=int)
    ranks[order] = np.arange(len(gaps))
    if any(int(np.count_nonzero(arm_stds == value)) != per_value for value in values):
        raise GapNoiseInstanceError("condition does not preserve the approved variance multiset")
    return np.repeat(arm_stds[:, None], 2, axis=1), ranks


def build_conditions(config: dict[str, Any] | None = None) -> list[FrozenCondition]:
    """Build both matched conditions from the approved deterministic formula."""
    config = load_config() if config is None else config
    means = _mean_matrix(config)
    pareto = np.flatnonzero(compute_pareto_mask(means))
    hardness = assert_non_degenerate("gap_noise_shared_means", means, pareto, require_tie_free=True)
    expected_hardness = float(config["hardness"]["expected_hardness"])
    if not np.isclose(hardness.hardness, expected_hardness, rtol=0.0, atol=1e-10):
        raise GapNoiseInstanceError(f"hardness={hardness.hardness}, expected {expected_hardness}")
    budget = int(round(hardness.hardness))
    grid = tuple(build_normalized_grid(budget, int(config["evaluation_grid"]["num_intervals"])))
    if len(grid) != 21 or grid[0] != 0 or grid[-1] != budget:
        raise GapNoiseInstanceError(f"invalid normalized grid {grid}")
    conditions = []
    for condition in CONDITIONS:
        stds, ranks = _standard_deviations(condition, hardness.gaps_direct, config)
        environment = PairedGaussianEnvironment(means, stds)
        conditions.append(
            FrozenCondition(
                condition=condition,
                means=means.copy(),
                stds=stds,
                gaps=hardness.gaps_direct.copy(),
                noise_ranks=ranks,
                pareto_indices=pareto.copy(),
                hardness=hardness,
                budget=budget,
                grid=grid,
                instance_hash=compute_instance_hash(environment),
            )
        )
    if not np.array_equal(conditions[0].means, conditions[1].means):
        raise GapNoiseInstanceError("conditions must share an identical mean matrix")
    if not np.array_equal(np.sort(conditions[0].arm_stds), np.sort(conditions[1].arm_stds)):
        raise GapNoiseInstanceError("conditions must share the variance multiset")
    return conditions


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_conditions(config: dict[str, Any] | None = None) -> list[FrozenCondition]:
    """Write checksummed long-form conditions and an audit-ready manifest."""
    config = load_config() if config is None else config
    conditions = build_conditions(config)
    ARMS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with ARMS_CSV.open("w", newline="", encoding="utf-8") as handle:
        fields = ["condition", "arm", "objective", "mean", "std", "true_gap", "noise_rank", "is_pareto"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in conditions:
            pareto = set(item.pareto_indices.tolist())
            for arm, objective in np.ndindex(item.means.shape):
                writer.writerow(
                    {
                        "condition": item.condition,
                        "arm": arm,
                        "objective": objective,
                        "mean": format(item.means[arm, objective], ".17g"),
                        "std": format(item.stds[arm, objective], ".17g"),
                        "true_gap": format(item.gaps[arm], ".17g"),
                        "noise_rank": int(item.noise_ranks[arm]),
                        "is_pareto": int(arm in pareto),
                    }
                )
    manifest = {
        "schema_version": "gap-noise-alignment-frozen-conditions-v1",
        "config_hash": config_hash(config),
        "num_conditions": len(conditions),
        "means_shared": True,
        "hardness_shared": float(conditions[0].hardness.hardness),
        "budget_shared": conditions[0].budget,
        "evaluation_grid_shared": list(conditions[0].grid),
        "conditions": [
            {
                "condition": item.condition,
                "instance_hash": item.instance_hash,
                "pareto_indices": item.pareto_indices.tolist(),
                "gaps": [float(value) for value in item.gaps],
                "noise_ranks": item.noise_ranks.tolist(),
                "std_counts_per_arm": {str(value): int(np.count_nonzero(item.arm_stds == value)) for value in STD_VALUES},
            }
            for item in conditions
        ],
        "artifacts": {"arms_csv_sha256": _sha256(ARMS_CSV)},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    CHECKSUMS_PATH.write_text(
        f"{_sha256(ARMS_CSV)}  {ARMS_CSV.name}\n{_sha256(MANIFEST_PATH)}  {MANIFEST_PATH.name}\n",
        encoding="utf-8",
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORTS_DIR / "environment_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["condition", "num_arms", "num_objectives", "pareto_size", "hardness", "budget", "std", "arm_count"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in conditions:
            for std in STD_VALUES:
                writer.writerow(
                    {
                        "condition": item.condition,
                        "num_arms": item.num_arms,
                        "num_objectives": item.num_objectives,
                        "pareto_size": len(item.pareto_indices),
                        "hardness": format(item.hardness.hardness, ".17g"),
                        "budget": item.budget,
                        "std": std,
                        "arm_count": int(np.count_nonzero(item.arm_stds == std)),
                    }
                )
    return conditions


def _checksums() -> dict[str, str]:
    if not CHECKSUMS_PATH.is_file():
        raise GapNoiseInstanceError("missing frozen condition checksums")
    checksums = {}
    for line in CHECKSUMS_PATH.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    return checksums


def load_conditions() -> list[FrozenCondition]:
    """Reload and independently verify the frozen artifacts/formulas."""
    if not ARMS_CSV.is_file() or not MANIFEST_PATH.is_file():
        raise GapNoiseInstanceError("frozen environments are missing; run generate_instances.py")
    checksums = _checksums()
    for path in (ARMS_CSV, MANIFEST_PATH):
        if checksums.get(path.name) != _sha256(path):
            raise GapNoiseInstanceError(f"checksum mismatch for {path.name}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    config = load_config()
    if manifest.get("config_hash") != config_hash(config):
        raise GapNoiseInstanceError("frozen environments do not match config")
    generated = build_conditions(config)
    rows_by_condition: dict[str, list[dict[str, str]]] = {condition: [] for condition in CONDITIONS}
    with ARMS_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows_by_condition[row["condition"]].append(row)
    loaded = []
    by_condition = {item.condition: item for item in generated}
    for condition in CONDITIONS:
        item = by_condition[condition]
        rows = rows_by_condition[condition]
        if len(rows) != 100:
            raise GapNoiseInstanceError(f"{condition}: expected 100 long-form rows, found {len(rows)}")
        means, stds = np.empty((50, 2)), np.empty((50, 2))
        for row in rows:
            arm, objective = int(row["arm"]), int(row["objective"])
            means[arm, objective], stds[arm, objective] = float(row["mean"]), float(row["std"])
            if not np.isclose(float(row["true_gap"]), item.gaps[arm], atol=1e-12):
                raise GapNoiseInstanceError(f"{condition}: frozen gap mismatch for arm {arm}")
            if int(row["noise_rank"]) != int(item.noise_ranks[arm]):
                raise GapNoiseInstanceError(f"{condition}: frozen noise rank mismatch for arm {arm}")
        if not np.array_equal(means, item.means) or not np.array_equal(stds, item.stds):
            raise GapNoiseInstanceError(f"{condition}: frozen arrays differ from formula")
        if item.instance_hash != next(entry["instance_hash"] for entry in manifest["conditions"] if entry["condition"] == condition):
            raise GapNoiseInstanceError(f"{condition}: manifest instance hash mismatch")
        loaded.append(item)
    return loaded
