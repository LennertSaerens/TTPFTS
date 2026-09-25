"""Deterministic RNG seeding from (master seed, ...tags).

Every seeded quantity in the runner is a pure function of a tuple of
string/int "parts": the master seed plus whatever identifies the unit of
work (environment/dist, algorithm, repetition, budget, ...). This is what
makes results invariant to:

  - dense vs. sparse evaluation grids (EGE reseeds per logged budget, not
    once per repetition, so evaluating budget 500 alone or as part of a
    denser grid gives byte-identical results for budget 500),
  - serial vs. parallel execution (each unit of work seeds itself; no
    shared/advancing global RNG state crosses unit boundaries),
  - running algorithms grouped together vs. in isolation (the algorithm name
    is part of the tag, and units for different algorithms never share a
    seeded RNG stream),
  - resuming a partially-completed run (a recomputed unit reseeds from the
    same tag it would have used the first time).
"""

from __future__ import annotations

import random
import zlib
from typing import Optional

import numpy as np


def derive_seed(master_seed: Optional[int], *parts) -> Optional[int]:
    """Derive a single deterministic uint32 seed from (master_seed, *parts).

    Returns ``None`` (meaning "do not seed anything") when ``master_seed`` is
    ``None``, preserving the historical no-seeding library default.
    """
    if master_seed is None:
        return None
    entropy = [int(master_seed) & 0xFFFFFFFF]
    for part in parts:
        entropy.append(zlib.crc32(str(part).encode("utf-8")) & 0xFFFFFFFF)
    seed_seq = np.random.SeedSequence(entropy)
    return int(seed_seq.generate_state(1, dtype=np.uint32)[0])


def seed_rngs(master_seed: Optional[int], *parts) -> Optional[int]:
    """Seed both ``numpy.random`` and stdlib ``random`` from (master_seed, *parts).

    Both generators must be seeded: Thompson samplers draw arms with
    ``random.choice`` while everything else uses ``np.random``. Returns the
    numpy seed used (or ``None`` if ``master_seed`` is ``None``), mostly for
    logging/debugging.
    """
    if master_seed is None:
        return None
    entropy = [int(master_seed) & 0xFFFFFFFF]
    for part in parts:
        entropy.append(zlib.crc32(str(part).encode("utf-8")) & 0xFFFFFFFF)
    seed_seq = np.random.SeedSequence(entropy)
    np_seed, py_seed = seed_seq.generate_state(2, dtype=np.uint32)
    np.random.seed(int(np_seed))
    random.seed(int(py_seed))
    return int(np_seed)
