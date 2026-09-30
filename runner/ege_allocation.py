"""Deterministic actual-draw counts for the repository's EGE implementations."""

from __future__ import annotations

import math

import numpy as np


def expected_ege_draws(algorithm: str, budget: int, num_arms: int) -> int:
    """Recompute EGE-SH/EGE-SR allocation counts without sampling rewards."""
    if num_arms < 2:
        raise ValueError("EGE requires at least two arms")
    if algorithm == "EGE_SH":
        rounds = math.ceil(math.log2(num_arms))
        active, total = num_arms, 0
        for _ in range(rounds):
            per_arm = math.floor(budget / (active * rounds))
            total += active * per_arm
            active = math.ceil(active / 2)
        return total
    if algorithm == "EGE_SR":
        log_k = 0.5 + float(np.sum(1.0 / np.arange(2, num_arms + 1)))
        n_ks = np.ceil(
            np.concatenate(
                ([0.0], (budget - num_arms) / (log_k * (num_arms + 1 - np.arange(1, num_arms))))
            )
        ).astype(int)
        total = 0
        for round_index in range(1, num_arms):
            per_arm = int(n_ks[round_index] - n_ks[round_index - 1])
            if per_arm > 0:
                total += (num_arms - round_index + 1) * per_arm
        return total
    raise ValueError(f"{algorithm} is not an EGE algorithm")
