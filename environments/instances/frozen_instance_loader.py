"""
Loader for the frozen EgeExp1/EgeExp4/EgeExp5 arm-mean instances.

These environments used to redraw (part of) their arm means from
``np.random`` on every ``reset()``. For the paper's final experiments a
single preregistered instance is frozen instead (see freeze_instances.py).
This module loads that frozen data and verifies it against the committed
sha256 checksums, so any accidental edit to the CSVs is caught loudly
rather than silently changing the benchmark.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

INSTANCES_DIR = Path(__file__).resolve().parent


class FrozenInstanceError(RuntimeError):
    pass


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_frozen_arms(env_name: str) -> np.ndarray:
    """Load and checksum-verify the frozen arm means for `env_name`.

    :param env_name: one of "EgeExp1", "EgeExp4", "EgeExp5".
    :return: (num_arms, num_objectives) float64 array of frozen arm means.
    """
    csv_path = INSTANCES_DIR / f"{env_name}_arms.csv"
    manifest_path = INSTANCES_DIR / "manifest.json"

    if not csv_path.exists():
        raise FrozenInstanceError(
            f"Frozen instance data for {env_name} not found at {csv_path}. "
            f"Run `python -m environments.instances.freeze_instances` once to generate it."
        )
    if not manifest_path.exists():
        raise FrozenInstanceError(
            f"manifest.json not found at {manifest_path}; cannot verify checksum for {env_name}."
        )

    manifest = json.loads(manifest_path.read_text())
    expected_sha256 = manifest.get("environments", {}).get(env_name, {}).get("sha256")
    if expected_sha256 is None:
        raise FrozenInstanceError(f"manifest.json has no checksum entry for {env_name}.")

    actual_sha256 = _sha256_of_file(csv_path)
    if actual_sha256 != expected_sha256:
        raise FrozenInstanceError(
            f"Checksum mismatch for {csv_path.name}: expected {expected_sha256}, "
            f"got {actual_sha256}. The frozen instance data may have been altered; "
            f"regenerate it deliberately via freeze_instances.py if that was intended."
        )

    arms = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    if arms.ndim == 1:
        arms = arms.reshape(1, -1)
    return arms
