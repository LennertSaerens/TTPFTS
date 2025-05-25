import numpy as np
from environments.BaseEnvironment import BaseEnvironment
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def generate_arms():
    """
    Generate the arms for the EgeExp8 environment.
    :return: A list of arms.
    """
    arms = []
    # Generate arms with the specified formula
    for i in range(1, 6):
        mu_i = (0.75 - 0.25 ** i)
        arms.append((mu_i, mu_i))
    return np.array(arms)


class EgeExp8(BaseEnvironment):
    """
    Experiment 8: Geometric progression with a single optimal arm.
    There are 5 arms and 2 objectives.
    For i = 1,...,5 we set mu_i = [0.75 - 0.25^i]^2
    """

    def __init__(self):
        self.arms = generate_arms()
        self.stds = 0.25  # Standard deviation for the normal distribution
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
        return [np.random.normal(mu[0], self.stds), np.random.normal(mu[1], self.stds)]

    def plot(self):
        """
        Plot the arms and the Pareto front.
        """
        plt.figure(figsize=(6, 6))
        plt.scatter(*zip(*self.arms), label='Arms')
        plt.scatter(*zip(*[self.arms[i] for i in self.pareto_indices]), color='green', label='Pareto Optimal Arms')

        # Draw ellipses around Pareto optimal arms
        for i in self.pareto_indices:
            ellipse = Ellipse(xy=self.arms[i], width=self.stds, height=self.stds, edgecolor='green', facecolor='none')
            plt.gca().add_patch(ellipse)

        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('EgeExp8 Environment Arms and Pareto Front')
        plt.legend()
        plt.grid()
        plt.show()