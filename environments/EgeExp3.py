import numpy as np
from environments.BaseEnvironment import BaseEnvironment
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def generate_arms():
    """
    Generate the arms for the EgeExp3 environment.
    :return: A list of arms.
    """
    arms = []
    # Generate the first 20 arms in the range [π/12, 5π/12]
    for i in range(20):
        angle = np.pi / 12 + i * (np.pi / 3) / 19  # Evenly spaced angles
        arms.append((np.cos(angle), np.sin(angle)))

    # Generate the next 180 arms in the range [4π/6, 11π/6]
    for i in range(20, 200):
        angle = 4 * np.pi / 6 + i * (7 * np.pi / 6) / 179  # Evenly spaced angles
        arms.append((np.cos(angle), np.sin(angle)))

    return np.array(arms)


class EgeExp3(BaseEnvironment):
    """
    Experiment 3: Many arms on the unit circle.
    There are 200 arms and 2 objectives.
    We choose b_1, ..., b_20 evenly spaced in [π/12, π/2 - π/12] and b_21, ..., b_200 evenly spaced in [π/2 + π/6, 2π - π/6].
    For i = 1, ..., 200 we set mu_i = (cos(b_i), sin(b_i))
    """

    def __init__(self):
        self.arms = generate_arms()
        self.stds = [0.25, 0.25] # Standard deviation for the normal distribution
        pareto_indices = np.arange(20)  # The first 20 arms are Pareto optimal
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
        # plt.title('EgeExp3 Environment Arms and Pareto Front')
        plt.legend()
        plt.grid()
        plt.savefig('environments/plots/EgeExp3.pdf', format='pdf')
        plt.show()
