"""
Reward distribution abstractions for multi-objective bandit environments.

Each concrete subclass encapsulates one arm's reward-generation process
across all objectives, exposing a common interface so that environments
are agnostic to the underlying distribution family.
"""

from abc import ABC, abstractmethod

import numpy as np


class RewardDistribution(ABC):
    """Per-arm multi-objective reward distribution."""

    @abstractmethod
    def sample(self) -> np.ndarray:
        """Draw one reward vector of shape (num_objectives,)."""

    @property
    @abstractmethod
    def mean(self) -> np.ndarray:
        """Expected reward vector."""

    @property
    @abstractmethod
    def std(self) -> np.ndarray:
        """Standard deviation of reward vector."""

    @property
    @abstractmethod
    def num_objectives(self) -> int:
        ...


class GaussianReward(RewardDistribution):
    """Gaussian (Normal) reward distribution per arm."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self._mean = np.asarray(mean, dtype=np.float64)
        self._std = np.asarray(std, dtype=np.float64)

    def sample(self) -> np.ndarray:
        return np.random.normal(self._mean, self._std)

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def std(self) -> np.ndarray:
        return self._std

    @property
    def num_objectives(self) -> int:
        return len(self._mean)


class BernoulliReward(RewardDistribution):
    """
    Bernoulli reward distribution per arm.

    Each objective is sampled independently as Bernoulli(p), yielding
    0 or 1 with E[X] = p.  Requires p in [0, 1].
    """

    def __init__(self, p: np.ndarray) -> None:
        self._p = np.asarray(p, dtype=np.float64)

    def sample(self) -> np.ndarray:
        return (np.random.random(self._p.shape) < self._p).astype(np.float64)

    @property
    def mean(self) -> np.ndarray:
        return self._p.copy()

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self._p * (1 - self._p))

    @property
    def num_objectives(self) -> int:
        return len(self._p)


class ExponentialReward(RewardDistribution):
    """
    Exponential reward distribution per arm.

    Each objective is sampled independently as Exponential(scale) where
    scale = mean.  The standard deviation equals the mean for this family.
    """

    def __init__(self, scale: np.ndarray) -> None:
        self._scale = np.asarray(scale, dtype=np.float64)

    def sample(self) -> np.ndarray:
        return np.random.exponential(self._scale)

    @property
    def mean(self) -> np.ndarray:
        return self._scale.copy()

    @property
    def std(self) -> np.ndarray:
        return self._scale.copy()

    @property
    def num_objectives(self) -> int:
        return len(self._scale)


class PoissonReward(RewardDistribution):
    """
    Poisson reward distribution per arm.

    Each objective is sampled independently as Poisson(lam) where
    lam is the rate parameter.  E[X] = lam, Var[X] = lam.
    """

    def __init__(self, lam: np.ndarray) -> None:
        self._lam = np.asarray(lam, dtype=np.float64)

    def sample(self) -> np.ndarray:
        return np.random.poisson(self._lam).astype(np.float64)

    @property
    def mean(self) -> np.ndarray:
        return self._lam.copy()

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self._lam)

    @property
    def num_objectives(self) -> int:
        return len(self._lam)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

DISTRIBUTION_REGISTRY = {
    "gaussian": GaussianReward,
    "bernoulli": BernoulliReward,
    "exponential": ExponentialReward,
    "poisson": PoissonReward,
}


def std_from_variance(variance: float | np.ndarray) -> float | np.ndarray:
    """Convert a variance (sigma^2) into the corresponding standard deviation (sigma).

    Use this helper at every Gaussian-reward call site instead of hand-computing
    ``sqrt(variance)`` inline, so the variance value being targeted is always
    visible next to the std it produces (e.g. ``std_from_variance(0.25)`` reads
    unambiguously as "sigma^2 = 0.25", producing sigma = 0.5).
    """
    return np.sqrt(np.asarray(variance, dtype=np.float64))


def make_distributions(
    arm_means: np.ndarray,
    dist: str = "gaussian",
    *,
    gaussian_std: float | np.ndarray = 0.5,
) -> list[RewardDistribution]:
    """
    Build a list of RewardDistribution objects from arm mean vectors.

    Parameters
    ----------
    arm_means : (num_arms, num_objectives) array
    dist : distribution family name
    gaussian_std : scalar, ``(num_objectives,)``, or
        ``(num_arms, num_objectives)`` array of Gaussian standard deviations
        sigma (NOT the variance sigma^2). E.g. to reproduce a manuscript spec
        of "diagonal covariance sigma^2 I_D with sigma^2=0.25", pass
        ``std_from_variance(0.25)`` (== 0.5), never 0.25 directly. A full
        matrix represents independent, heteroscedastic arm-objective rewards.
    """
    arm_means = np.asarray(arm_means, dtype=np.float64)
    std_arr = np.asarray(gaussian_std, dtype=np.float64)

    if dist == "gaussian":
        try:
            std_matrix = np.broadcast_to(std_arr, arm_means.shape)
        except ValueError as exc:
            raise ValueError(
                "gaussian_std must be scalar, shape (num_objectives,), or "
                f"shape {arm_means.shape}; got shape {std_arr.shape}."
            ) from exc
        if not np.all(np.isfinite(std_matrix)) or np.any(std_matrix <= 0.0):
            raise ValueError("gaussian_std must contain only finite positive values.")
        return [GaussianReward(m, std_matrix[i]) for i, m in enumerate(arm_means)]
    elif dist == "bernoulli":
        if np.any(arm_means < 0) or np.any(arm_means > 1):
            raise ValueError(
                f"Bernoulli distribution requires arm means in [0, 1], "
                f"got range [{arm_means.min():.4f}, {arm_means.max():.4f}]."
            )
        return [BernoulliReward(m) for m in arm_means]
    elif dist == "exponential":
        if np.any(arm_means <= 0):
            raise ValueError(
                f"Exponential distribution requires arm means > 0, "
                f"got min={arm_means.min():.4f}."
            )
        return [ExponentialReward(m) for m in arm_means]
    elif dist == "poisson":
        if np.any(arm_means <= 0):
            raise ValueError(
                f"Poisson distribution requires arm means > 0, "
                f"got min={arm_means.min():.4f}."
            )
        return [PoissonReward(m) for m in arm_means]
    else:
        raise ValueError(f"Unknown distribution: {dist!r}. "
                         f"Available: {list(DISTRIBUTION_REGISTRY)}")
