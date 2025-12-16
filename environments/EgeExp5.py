import numpy as np
from environments.BaseEnvironment import BaseEnvironment
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def generate_arms():
    """
    Generate the arms for the EgeExp5 environment.
    :return: A list of arms.
    """
    arms = []
    # Generate the first 10 arms in the range [0.2, 0.4]^2
    for i in range(10):
        arm = np.random.uniform(0.2, 0.4, size=2)
        arms.append(tuple(arm))

    # Generate the next 10 arms in the range [0.5, 0.7]^2
    for i in range(10):
        arm = np.random.uniform(0.5, 0.7, size=2)
        arms.append(tuple(arm))

    return np.array(arms)


class EgeExp5(BaseEnvironment):
    """
    Experiment 5: 2 clusters of arms.
    There are 20 arms and 2 objectives.
    We choose mu_1, ..., mu_10 uniformly in [0.2, 0.4]^2 and mu_11, ..., mu_20 uniformly in [0.5, 0.7]^2.
    """

    def __init__(self):
        self.arms = generate_arms()
        self.stds = [0.25, 0.25]  # Standard deviation for the normal distribution
        is_strictly_worse = np.all(self.arms[:, None, :] < self.arms[None, :, :], axis=2)
        pareto_indices = np.where(~np.any(is_strictly_worse, axis=1))[0]
        reference_point = np.array([1.0, 1.0])
        inverted_arms = [(1 - arm[0], 1 - arm[1]) for arm in self.arms]
        super().__init__(len(self.arms), 2, pareto_indices, inverted_arms, reference_point)

    def pull_arm(self, arm):
        """
        Pull the specified arm and return the reward.
        :param arm: The index of the arm to pull.
        :return: The reward for the pulled arm.
        """
        mu = self.arms[arm]
        return [np.random.normal(mu[0], self.stds[0]), np.random.normal(mu[1], self.stds[1])]

    def plot(self, save_png=False, save_file=None):
        """
        Plot the arms and the Pareto front.
        """
        plt.figure(figsize=(6, 3))

        plt.scatter(*zip(*self.arms), label='Suboptimal Arms')
        plt.scatter(*zip(*[self.arms[i] for i in self.pareto_indices]), color='green', label='Optimal Arms')

        # Draw ellipses around Pareto optimal arms
        for i in self.pareto_indices:
            ellipse = Ellipse(xy=self.arms[i], width=2*self.stds[0], height=2*self.stds[1], edgecolor='green', facecolor='none', alpha=0.5)
            plt.gca().add_patch(ellipse)

        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.legend()
        plt.grid()
        plt.subplots_adjust(bottom=0.2, left=0.15)
        if save_png and save_file is not None:
            plt.savefig(save_file, format='png', dpi=300)
        plt.show()

    def reset(self):
        self.arms = generate_arms()
        is_strictly_worse = np.all(self.arms[:, None, :] < self.arms[None, :, :], axis=2)
        self.pareto_indices = np.where(~np.any(is_strictly_worse, axis=1))[0]
