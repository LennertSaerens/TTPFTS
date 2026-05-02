from abc import ABC
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from paretoset import paretoset

from environments.distributions import (
    RewardDistribution, GaussianReward, make_distributions,
)


class BaseEnvironment(ABC):
    def __init__(self, num_arms: int, num_objectives: int, pareto_indices: np.ndarray,
                 inverted_arms: Optional[np.ndarray] = None,
                 reference_point: Optional[np.ndarray] = None,
                 distributions: Optional[list[RewardDistribution]] = None) -> None:
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.pareto_indices = np.asarray(pareto_indices)
        self._pareto_set = set(self.pareto_indices.tolist())
        self.inverted_arms = np.asarray(inverted_arms) if inverted_arms is not None else None
        self.reference_point = reference_point
        self.distributions = distributions

    # ------------------------------------------------------------------
    # Derived properties (backward-compatible access to arms / stds)
    # ------------------------------------------------------------------

    @property
    def arms(self) -> np.ndarray:
        """Mean reward vectors, shape (num_arms, num_objectives)."""
        if self.distributions is not None:
            return np.array([d.mean for d in self.distributions])
        return self._arms

    @arms.setter
    def arms(self, value: np.ndarray) -> None:
        self._arms = np.asarray(value, dtype=np.float64)

    @property
    def stds(self) -> np.ndarray:
        """Standard-deviation vectors.

        Returns shape (num_arms, num_objectives) when per-arm stds differ,
        or (num_objectives,) when they are shared.
        """
        if self.distributions is not None:
            all_stds = np.array([d.std for d in self.distributions])
            # If all rows identical, collapse to 1-D for backward compat
            if np.all(all_stds == all_stds[0]):
                return all_stds[0]
            return all_stds
        return self._stds

    @stds.setter
    def stds(self, value: np.ndarray) -> None:
        self._stds = np.asarray(value, dtype=np.float64)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_pareto_indices(arms: np.ndarray) -> np.ndarray:
        """Returns indices of Pareto-optimal arms (no arm is strictly dominated)."""
        num_objectives = arms.shape[1]
        pareto_mask = paretoset(arms, sense=["max"] * num_objectives)
        return np.where(pareto_mask)[0]

    def _update_pareto_cache(self) -> None:
        """Update the cached pareto set after pareto_indices changes."""
        self._pareto_set = set(self.pareto_indices.tolist())

    def _init_standard_2obj(self, arms: np.ndarray,
                            pareto_indices: Optional[np.ndarray] = None,
                            dist: str = "gaussian") -> None:
        """Common init for standard 2-objective environments with stds=0.25."""
        arm_means = np.asarray(arms, dtype=np.float64)
        distributions = make_distributions(arm_means, dist=dist, gaussian_std=0.25)
        if pareto_indices is None:
            pareto_indices = BaseEnvironment._compute_pareto_indices(arm_means)
        reference_point = np.array([1.0, 1.0])
        inverted_arms = 1.0 - arm_means
        BaseEnvironment.__init__(self, len(arm_means), 2, pareto_indices,
                                 inverted_arms, reference_point,
                                 distributions=distributions)

    # ------------------------------------------------------------------
    # Reward sampling
    # ------------------------------------------------------------------

    def pull_arm(self, arm: int) -> np.ndarray:
        """Pulls the specified arm and returns a noisy reward vector."""
        if self.distributions is not None:
            return self.distributions[arm].sample()
        stds = self._stds[arm] if self._stds.ndim > 1 else self._stds
        return np.random.normal(self._arms[arm], stds)

    def get_top_arms(self) -> np.ndarray:
        """Returns the arms considered Pareto optimal."""
        return self.pareto_indices

    def learn(self, arm: int, reward: np.ndarray) -> None:
        """Updates the model with the observed reward from the chosen arm."""
        pass

    def reset(self) -> None:
        """Resets the environment."""
        pass

    def sample(self, arms: Union[list, np.ndarray]) -> np.ndarray:
        """
        Sample multiple arms at once.
        :param arms: A list/array of arm indices to sample.
        :return: A 2D array of rewards of shape (len(arms), num_objectives).
        """
        arms = np.asarray(arms)
        if self.distributions is not None:
            return np.array([self.distributions[a].sample() for a in arms])
        means = self._arms[arms]
        stds = self._stds[arms] if self._stds.ndim > 1 else self._stds
        return np.random.normal(means, stds)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def bernoulli_metric(self, recommendation: np.ndarray) -> int:
        """Returns 1 if the recommendation exactly matches the Pareto set, else 0."""
        return int(set(recommendation) == self._pareto_set)

    def jaccard_metric(self, recommendation: np.ndarray) -> float:
        """Jaccard similarity between the recommended arms and the Pareto optimal arms."""
        rec_set = set(recommendation)
        return len(rec_set & self._pareto_set) / len(rec_set | self._pareto_set)

    def mis_id_metric(self, recommendation: np.ndarray) -> float:
        """Average mis-identification rate over all arms (symmetric difference / total)."""
        rec_set = set(recommendation)
        symmetric_diff = rec_set.symmetric_difference(self._pareto_set)
        return len(symmetric_diff) / self.num_arms

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, save_png: bool = False, save_file: Optional[str] = None) -> None:
        """Plot the arms and the Pareto front (default: 2D scatter with uncertainty ellipses)."""
        plt.figure(figsize=(6, 3))
        arm_means = self.arms
        arm_stds = self.stds

        plt.scatter(*zip(*arm_means), label='Suboptimal Arms')
        plt.scatter(*zip(*[arm_means[i] for i in self.pareto_indices]), color='green', label='Optimal Arms')

        for i in self.pareto_indices:
            std_i = arm_stds[i] if arm_stds.ndim > 1 else arm_stds
            ellipse = Ellipse(xy=arm_means[i], width=2 * std_i[0], height=2 * std_i[1],
                              edgecolor='green', facecolor='none', alpha=0.5)
            plt.gca().add_patch(ellipse)

        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.legend()
        plt.grid()
        plt.subplots_adjust(bottom=0.2, left=0.15)
        if save_png and save_file is not None:
            plt.savefig(save_file, format='png', dpi=300)
        plt.show()
