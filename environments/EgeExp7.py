import numpy as np
from environments.BaseEnvironment import BaseEnvironment
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def generate_arms():
    """
    Generate the arms for the EgeExp7 environment.
    :return: A list of arms.
    """
    arms = []
    c = 0.05  # Sub-optimality gap
    # Generate arms for i = 1,...,8
    for i in range(1, 9):
        c_i = (i - 1) * c
        arms.append((0.3 + c_i, 0.8 - c_i))

    # Generate arms for i = 9,...,15
    for i in range(9, 16):
        c_i = (i - 8) * c
        arms.append((0.25 + c_i, 0.7 - c_i))

    # Generate arms for i = 16,...,22
    for i in range(16, 23):
        mu_prev = arms[i - 8]  # mu_(i-7)
        arms.append((mu_prev[0], mu_prev[1] - 0.05))  # mu_i = mu_(i-7) - (0,-0.05)

    return np.array(arms)


class EgeExp7(BaseEnvironment):
    """
    Experiment 7: All the arms have the same sub-optimality gap.
    There are 22 arms and 2 objectives.
    For i = 1,...,8 we set mu_i = (0.3+c_i,0.8−c_i) where c_i = (i−1)*c
    For i = 9,...,15 we set mu_i = (0.25+c_(i−8),0.7−c_(i−8))
    For i = 16,...,22 we set mu_i = mu_(i-7) - (0,-0.05).
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
        plt.figure(figsize=(8, 6))
        plt.scatter(*zip(*self.arms), label='Arms')
        plt.scatter(*zip(*[self.arms[i] for i in self.pareto_indices]), color='green', label='Pareto Optimal Arms')

        # Draw ellipses around Pareto optimal arms
        for i in self.pareto_indices:
            ellipse = Ellipse(xy=self.arms[i], width=self.stds, height=self.stds, edgecolor='green', facecolor='none', alpha=0.5)
            plt.gca().add_patch(ellipse)

        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        # plt.title('EgeExp7 Environment Arms and Pareto Front')
        plt.legend()
        plt.grid()
        plt.savefig('environments/plots/EgeExp7.pdf', format='pdf')
        plt.show()
