import numpy as np
from environments.BaseEnvironment import BaseEnvironment
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def generate_arms():
    """
    Generate the arms for the EgeExp1 environment.
    :return: A list of arms.
    """
    arms1 = [(x ** 2, 1 / (4 * x ** 2)) for x in np.linspace(0.55, 0.95, 10)]
    arms2 = []
    while len(arms2) < 50:
        x = np.random.uniform(0.1, 0.8)
        y = np.random.uniform(0.1, 0.8)
        if x * y <= 0.2:
            arms2.append((x, y))
    return np.array(arms1 + arms2)


class EgeExp1(BaseEnvironment):
    """
    Experiment 1: Arms on a convex Pareto set.
    There are 60 arms and 2 objectives.
    We choose x_1, ..., x_10 equally spaced in [0.55, 0.95] and for i = 1, ..., 10 we set mu_i = (x_i^2, 1/(4x_i^2)).
    mu_11, ..., mu_60 are chosen from {(x, y) \in [0.1, 0.8]^2 | xy <= 0.2}.
    """

    def __init__(self):
        self.arms = generate_arms()
        self.stds = [0.25, 0.25]  # Standard deviation for the normal distribution
        pareto_indices = np.arange(10)  # The first 10 arms are Pareto optimal
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

    def plot(self):
        """
        Plot the arms and the Pareto front.
        """
        plt.figure(figsize=(8, 6))

        plt.scatter(*zip(*self.arms), label='Arms')
        plt.scatter(*zip(*[self.arms[i] for i in self.pareto_indices]), color='green', label='Pareto Optimal Arms')

        # Draw ellipses around Pareto optimal arms
        for i in self.pareto_indices:
            ellipse = Ellipse(xy=self.arms[i], width=self.stds[0], height=self.stds[1], edgecolor='green', facecolor='none', alpha=0.5)
            plt.gca().add_patch(ellipse)

        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        # plt.title('Kone et al Syhtetic Benchmark Experiment 1\nArms and Pareto Front')
        plt.legend()
        plt.grid()
        plt.savefig('environments/plots/EgeExp1.pdf', format='pdf')
        plt.show()

    def reset(self) -> None:
        """
        Reset the environment to its initial state.
        """
        self.arms = generate_arms()
