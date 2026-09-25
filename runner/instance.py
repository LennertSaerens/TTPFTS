"""Deterministic hashing of a frozen environment instance.

Some environments redraw their arm means on ``reset()`` (e.g. EgeExp1/4/5,
N50VS); others are fixed. Either way, once an environment instance has been
reset for a given repetition its means are frozen for the rest of that
repetition (every algorithm and every evaluation budget see the same arms).
``instance_hash`` records *which* frozen instance a row's metrics were
computed against, without requiring any change to the environment classes
themselves.
"""

from __future__ import annotations

import hashlib

import numpy as np


def compute_instance_hash(environment) -> str:
    """Hash an environment's frozen arm means and stds.

    Uses only the public ``arms``/``stds`` properties already exposed by
    ``BaseEnvironment`` subclasses, so it works unmodified for every
    environment in the repo.
    """
    arms = np.ascontiguousarray(np.asarray(environment.arms, dtype=np.float64))
    stds = np.ascontiguousarray(np.asarray(environment.stds, dtype=np.float64))
    hasher = hashlib.sha256()
    hasher.update(str(arms.shape).encode("utf-8"))
    hasher.update(arms.tobytes())
    hasher.update(str(stds.shape).encode("utf-8"))
    hasher.update(stds.tobytes())
    return hasher.hexdigest()[:16]
