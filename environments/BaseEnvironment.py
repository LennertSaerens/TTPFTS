from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from paretoset import paretoset
from pymoo.indicators.hv import HV


class BaseEnvironment(ABC):
    def __init__(self, num_arms: int, num_objectives: int, pareto_indices,
                 inverted_arms=None, reference_point=None) -> None:
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.pareto_indices = pareto_indices
        self.inverted_arms = inverted_arms
        self.reference_point = reference_point

    @staticmethod
    def _compute_pareto_indices(self, arms: np.ndarray) -> np.ndarray:
        """Returns indices of Pareto-optimal arms (no arm is strictly dominated)."""
        pareto_mask = paretoset(arms, sense=["max"] * self.num_objectives)
        pareto_indices = np.where(pareto_mask)[0]
        return pareto_indices

    def pull_arm(self, arm: int) -> list:
        """Pulls the specified arm and returns a noisy reward vector."""
        mu = self.arms[arm]
        return [np.random.normal(mu[i], self.stds[i]) for i in range(self.num_objectives)]

    def get_top_arms(self) -> np.ndarray:
        """Returns the arms considered Pareto optimal."""
        return self.pareto_indices

    def learn(self, arm: int, reward: float) -> None:
        """Updates the model with the observed reward from the chosen arm."""
        pass

    def reset(self) -> None:
        """Resets the environment."""
        pass

    def sample(self, arms):
        """
        Sample multiple arms at once.
        :param arms: A list of arm indices to sample.
        :return: A 2D array of rewards for the sampled arms.
        """
        rewards = [self.pull_arm(arm) for arm in arms]
        return np.array(rewards)

    def bernoulli_metric(self, recommendation):
        """
        Calculate the Bernoulli metric for the specified arm.
        :param recommendation: The recommended arms.
        :return: The Bernoulli metric for the arm.
        """
        return int(set(recommendation) == set(self.pareto_indices))

    def jaccard_metric(self, recommendation):
        """
        Calculate the Jaccard similarity between the recommended arms and the pareto optimal arms.
        :param recommendation: The recommended arms.
        :return: The Jaccard similarity.
        """
        return len(set(recommendation).intersection(set(self.pareto_indices))) / len(
            set(recommendation).union(set(self.pareto_indices)))

    def mis_id_metric(self, recommendation):
        """
        Calculate the average mis-identification rate over all arms.
        :param recommendation: The recommended arms.
        :return: The average mis-identification rate.
        """
        mis_identifications = 0
        for arm in np.arange(self.num_arms):
            if arm in recommendation:
                if arm not in self.pareto_indices:
                    mis_identifications += 1
            else:
                if arm in self.pareto_indices:
                    mis_identifications += 1
        return mis_identifications / self.num_arms

    def plot(self, save_png=False, save_file=None):
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
