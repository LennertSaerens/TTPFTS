"""Pure, validity-aware posterior uncertainty summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from paretoset import paretoset


@dataclass(frozen=True)
class UQResult:
    """UQ measures or an explicit scientific reason they are unavailable."""

    available: bool
    reason: str
    bhattacharyya: float | None
    symmetric_kl: float | None
    entropy: float | None
    first_front_size: int
    second_front_size: int


def _unavailable(reason: str, first_size: int = 0, second_size: int = 0) -> UQResult:
    return UQResult(False, reason, None, None, None, first_size, second_size)


def posterior_stds_from_tposterior_state(n: np.ndarray, m2: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    """Reconstruct actual posterior std(mu), preserving mathematically infinite values."""
    n = np.asarray(n, dtype=np.float64)
    m2 = np.asarray(m2, dtype=np.float64)
    if n.shape != m2.shape:
        raise ValueError("posterior n and M2 shapes differ")
    if np.any(n < 0) or np.any(m2 < 0) or not np.all(np.isfinite(m2)):
        raise ValueError("invalid TPosterior sufficient statistics")
    df = n + 2.0 * float(alpha) - 1.0
    if np.any(df <= 0):
        raise ValueError("posterior undefined: degrees of freedom are non-positive")
    scale = np.sqrt(m2 / (n * df))
    with np.errstate(divide="ignore", invalid="ignore"):
        variance_factor = np.where(df > 2.0, df / (df - 2.0), np.inf)
    return scale * np.sqrt(variance_factor)


def _fronts(means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = np.flatnonzero(paretoset(means, sense=["max"] * means.shape[1], distinct=False))
    remaining = np.setdiff1d(np.arange(means.shape[0]), first, assume_unique=True)
    if remaining.size == 0:
        return first, remaining
    second_mask = paretoset(means[remaining], sense=["max"] * means.shape[1], distinct=False)
    return first, remaining[np.flatnonzero(second_mask)]


def uncertainty_from_moments(means: np.ndarray, stds: np.ndarray) -> UQResult:
    """Compute the three existing moment-matched diagonal-Gaussian measures.

    The Student-t posterior is summarized by its posterior mean and actual
    marginal standard deviation. This preserves the former analysis's
    diagonal-Gaussian definitions while explicitly rejecting undefined states.
    """
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    if means.ndim != 2 or stds.shape != means.shape:
        raise ValueError("means and stds must be same-shaped two-dimensional arrays")
    if not np.all(np.isfinite(means)):
        return _unavailable("nonfinite_posterior_mean")
    if not np.all(np.isfinite(stds)) or np.any(stds <= 0):
        return _unavailable("nonfinite_or_nonpositive_posterior_std")
    first, second = _fronts(means)
    if second.size == 0:
        return _unavailable("no_second_estimated_front", int(first.size), 0)

    first_means, second_means = means[first], means[second]
    first_vars, second_vars = stds[first] ** 2, stds[second] ** 2
    total_bc = 0.0
    total_kl = 0.0
    for first_index in range(first.size):
        for second_index in range(second.size):
            var1 = first_vars[first_index]
            var2 = second_vars[second_index]
            var_avg = 0.5 * (var1 + var2)
            difference_squared = (first_means[first_index] - second_means[second_index]) ** 2
            db = 0.125 * np.sum(difference_squared / var_avg)
            db += 0.5 * (
                np.sum(np.log(var_avg))
                - 0.5 * (np.sum(np.log(var1)) + np.sum(np.log(var2)))
            )
            total_bc += float(np.exp(-db))
            kl12 = 0.5 * (
                np.sum(var1 / var2)
                + np.sum(difference_squared / var2)
                - means.shape[1]
                + np.sum(np.log(var2) - np.log(var1))
            )
            kl21 = 0.5 * (
                np.sum(var2 / var1)
                + np.sum(difference_squared / var1)
                - means.shape[1]
                + np.sum(np.log(var1) - np.log(var2))
            )
            total_kl += float(0.5 * (kl12 + kl21))
    pairs = first.size * second.size
    all_vars = np.vstack((first_vars, second_vars))
    entropy = 0.5 * means.shape[1] * (1.0 + np.log(2.0 * np.pi))
    entropy += 0.5 * np.mean(np.sum(np.log(all_vars), axis=1))
    return UQResult(
        True,
        "available",
        total_bc / pairs,
        total_kl / pairs,
        float(entropy),
        int(first.size),
        int(second.size),
    )


def uncertainty_from_tposterior_state(
    n: np.ndarray, means: np.ndarray, m2: np.ndarray, alpha: float = 0.0
) -> UQResult:
    """Compute UQ from serializable TPosterior Welford sufficient statistics."""
    try:
        stds = posterior_stds_from_tposterior_state(n, m2, alpha)
    except ValueError as exc:
        return _unavailable(str(exc))
    return uncertainty_from_moments(means, stds)
