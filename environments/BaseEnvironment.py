from abc import ABC
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from paretoset import paretoset


class BaseEnvironment(ABC):
    def __init__(self, num_arms: int, num_objectives: int, pareto_indices: np.ndarray,
                 inverted_arms: Optional[np.ndarray] = None,
                 reference_point: Optional[np.ndarray] = None) -> None:
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.pareto_indices = np.asarray(pareto_indices)
        self._pareto_set = set(self.pareto_indices.tolist())
        self.inverted_arms = np.asarray(inverted_arms) if inverted_arms is not None else None
        self.reference_point = reference_point

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
                            pareto_indices: Optional[np.ndarray] = None) -> None:
        """Common init for standard 2-objective environments with stds=0.25."""
        self.arms = np.asarray(arms, dtype=np.float64)
        self.stds = np.array([0.25, 0.25])
        if pareto_indices is None:
            pareto_indices = BaseEnvironment._compute_pareto_indices(self.arms)
        reference_point = np.array([1.0, 1.0])
        inverted_arms = 1.0 - self.arms
        BaseEnvironment.__init__(self, len(self.arms), 2, pareto_indices, inverted_arms, reference_point)

    def pull_arm(self, arm: int) -> np.ndarray:
        """Pulls the specified arm and returns a noisy reward vector."""
        stds = self.stds[arm] if self.stds.ndim > 1 else self.stds
        return np.random.normal(self.arms[arm], stds)

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
        means = self.arms[arms]
        # Handle per-arm stds (num_arms, num_objectives) vs global stds (num_objectives,)
        stds = self.stds[arms] if self.stds.ndim > 1 else self.stds
        return np.random.normal(means, stds)

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

    def plot(self, save_png: bool = False, save_file: Optional[str] = None) -> None:
        """Plot the arms and the Pareto front (default: 2D scatter with uncertainty ellipses)."""
        plt.figure(figsize=(6, 3))

        plt.scatter(*zip(*self.arms), label='Suboptimal Arms')
        plt.scatter(*zip(*[self.arms[i] for i in self.pareto_indices]), color='green', label='Optimal Arms')

        for i in self.pareto_indices:
            ellipse = Ellipse(xy=self.arms[i], width=2 * self.stds[0], height=2 * self.stds[1],
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
