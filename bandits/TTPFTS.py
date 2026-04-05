import random
from abc import abstractmethod

import numpy as np
from paretoset import paretoset
from bandits.InterfaceMOMABPFI import BaseMOMABAlgorithm


def _non_dominated_sort(samples: np.ndarray, sense: list) -> list[np.ndarray]:
    """
    Perform non-dominated sorting on samples.
    Returns a list of fronts, where each front is an array of global arm indices.
    Front 0 is the first (best) Pareto front, front 1 is the second, etc.
    """
    fronts = []
    remaining_indices = np.arange(len(samples))

    while len(remaining_indices) > 0:
        remaining_samples = samples[remaining_indices]
        pareto_mask = paretoset(remaining_samples, sense=sense, distinct=False)
        front = remaining_indices[pareto_mask]
        fronts.append(front)
        remaining_indices = remaining_indices[~pareto_mask]

    return fronts


class BaseThompsonParetoBandit(BaseMOMABAlgorithm):
    """
    Base class for Thompson-sampling-based Pareto set identification bandits.

    Provides shared warm-up logic, posterior management, top-arm identification,
    learning, and reset. Subclasses implement only `_select_arm` to define
    how an arm is chosen from the posterior samples after warm-up.
    """

    def __init__(self, posterior, num_warmup_pulls: int = 2):
        super().__init__(posterior.num_arms, posterior.num_objectives)
        self.posterior = posterior
        self.num_warmup_pulls = num_warmup_pulls
        self.current_warmup_arm = 0
        self.warmup_pulls = np.zeros(posterior.num_arms, dtype=int)
        self._pareto_sense = ["max"] * posterior.num_objectives

    def choose_arm(self) -> int:
        # Warm-up: round-robin each arm num_warmup_pulls times
        if np.any(self.warmup_pulls < self.num_warmup_pulls):
            arm = self.current_warmup_arm
            self.warmup_pulls[arm] += 1
            self.current_warmup_arm = (self.current_warmup_arm + 1) % self.posterior.num_arms
            return arm

        samples = self.posterior.sample()
        return self._select_arm(samples)

    @abstractmethod
    def _select_arm(self, samples: np.ndarray) -> int:
        """Choose an arm given posterior samples (num_arms × num_objectives)."""

    def get_top_arms(self) -> np.ndarray:
        means = self.posterior.get_mean()
        pareto_mask = paretoset(means, sense=self._pareto_sense)
        return np.where(pareto_mask)[0]

    def learn(self, arm: int, reward: np.ndarray) -> None:
        self.posterior.update(arm, reward)

    def reset(self, env_stds) -> None:
        self.posterior.reset(env_stds)
        self.warmup_pulls = np.zeros(self.posterior.num_arms, dtype=int)
        self.current_warmup_arm = 0


class TTPFTSBandit(BaseThompsonParetoBandit):
    """
    Top-Two Pareto Fronts Thompson Sampling Bandit.

    Selects an arm from the first Pareto front with probability p,
    otherwise from the second Pareto front of the posterior samples.
    """

    def __init__(self, posterior, p: float = 0.5, num_warmup_pulls: int = 2):
        super().__init__(posterior, num_warmup_pulls)
        self.p = p

    def _select_arm(self, samples: np.ndarray) -> int:
        pareto_mask = paretoset(samples, sense=self._pareto_sense)
        pareto_indices = np.where(pareto_mask)[0]

        if np.random.random() < self.p:
            return random.choice(pareto_indices)

        non_pareto_indices = np.where(~pareto_mask)[0]
        if len(non_pareto_indices) == 0:
            return random.choice(pareto_indices)

        non_pareto_pareto_mask = paretoset(samples[non_pareto_indices], self._pareto_sense)
        return random.choice(non_pareto_indices[np.where(non_pareto_pareto_mask)[0]])


class CPFTSBandit(BaseThompsonParetoBandit):
    """
    Cascading Pareto Front Thompson Sampling (CPFTS) Bandit.

    Performs non-dominated sorting on posterior samples and selects an arm
    via a geometric cascade: front k is chosen with probability
    rho * (1-rho)^(k-1). The last front is the unconditional fallback.
    """

    def __init__(self, posterior, rho: float = 0.5, num_warmup_pulls: int = 2):
        super().__init__(posterior, num_warmup_pulls)
        self.rho = rho

    def _select_arm(self, samples: np.ndarray) -> int:
        fronts = _non_dominated_sort(samples, self._pareto_sense)

        for front in fronts[:-1]:
            if np.random.random() < self.rho:
                return random.choice(front)
        return random.choice(fronts[-1])


class RPFTSBandit(BaseThompsonParetoBandit):
    """
    Resampling Pareto Front Thompson Sampling (RPFTS) Bandit.

    With probability rho, selects an arm from the first Pareto front of
    the posterior samples. Otherwise, resamples all arms until at least
    one produces a sample that is non-dominated by the original first
    front, then selects one of those challenging arms.
    """

    def __init__(self, posterior, rho: float = 0.5, num_warmup_pulls: int = 2,
                 max_resamples: int = 100_000):
        super().__init__(posterior, num_warmup_pulls)
        self.rho = rho
        self.max_resamples = max_resamples

    def _select_arm(self, samples: np.ndarray) -> int:
        pareto_mask = paretoset(samples, sense=self._pareto_sense)
        pareto_indices = np.where(pareto_mask)[0]

        # With probability rho, pick from the first front
        if np.random.random() < self.rho:
            return random.choice(pareto_indices)

        # If all arms are on the first front, pick at random
        if len(pareto_indices) == len(samples):
            return random.choice(pareto_indices)

        front0_samples = samples[pareto_indices]

        # Resample all arms until a challenger is non-dominated by front 0
        for _ in range(self.max_resamples):
            new_samples = self.posterior.sample()
            combined = np.vstack([front0_samples, new_samples])
            combined_pareto = paretoset(combined, sense=self._pareto_sense, distinct=False)
            # Challenger indices are those in the new_samples portion of combined
            challenger_mask = combined_pareto[len(front0_samples):]
            if np.any(challenger_mask):
                return random.choice(np.where(challenger_mask)[0])

        # Fallback after max_resamples: pick any arm at random
        return random.randint(0, len(samples) - 1)
