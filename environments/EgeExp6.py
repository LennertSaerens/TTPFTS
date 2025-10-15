import numpy as np
from environments.BaseEnvironment import BaseEnvironment
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def generate_arms():
    """
    Generate the arms for the EgeExp6 environment.
    :return: A list of arms.
    """
    arms = []
    # Generate arms with the specified formula
    for i in range(10):
        mu_i = (0.75 - 0.65 ** i, 0.25 + 0.65 ** i)
        arms.append(mu_i)
    return np.array(arms)


class EgeExp6(BaseEnvironment):
    """
    Experiment 6: All the arms are optimal
    There are 10 arms and 2 objectives.
    For any arm i, mu_i = (0.75 - 0.65^i, 0.25 + 0.65^i).
    """

    def __init__(self):
        self.arms = generate_arms()
        self.stds = 0.25  # Standard deviation for the normal distribution
        pareto_indices = np.arange(10)  # All arms are Pareto optimal
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
        plt.figure(figsize=(8, 6))
        plt.scatter(*zip(*self.arms), label='Arms')
        plt.scatter(*zip(*[self.arms[i] for i in self.pareto_indices]), color='green', label='Pareto Optimal Arms')

        # Draw ellipses around Pareto optimal arms
        for i in self.pareto_indices:
            ellipse = Ellipse(xy=self.arms[i], width=self.stds, height=self.stds, edgecolor='green', facecolor='none', alpha=0.5)
            plt.gca().add_patch(ellipse)

        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        # plt.title('EgeExp6 Environment Arms and Pareto Front')
        plt.legend()
        plt.grid()
        plt.savefig('environments/plots/EgeExp6.pdf', format='pdf')
        plt.show()
