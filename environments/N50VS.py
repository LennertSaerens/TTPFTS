import numpy as np
from environments.BaseEnvironment import BaseEnvironment
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class N50VS(BaseEnvironment):
    def __init__(self, std_low=1, std_high=3):
        self.optimal_arms = [(1, 4.5), (1.5, 3.5), (3, 3), (3.5, 1.5), (4.5, 1), (2, 3.1), (3.1, 2)]
        self.suboptimal_arms = [(0.5, 4), (1, 3), (1, 2), (1, 1), (2, 1), (2.5, 2.5), (3, 1), (4, 0.5), (0.5, 0.5), (0.5, 1),
                                (0.5, 2), (0.5, 3), (1, 0.5), (2, 0.5), (3, 0.5), (2, 2), (1.5, 2.5), (2.5, 1.5), (1.5, 1.5),
                                (0.5, 3.5), (3.5, 0.5), (2, 2.5), (2.5, 2), (0.5, 1.5), (1.5, 0.5), (1.5, 2), (2, 1.5),
                                (0.5, 2.5), (2.5, 0.5), (1, 1.5), (1.5, 1), (1, 2.5), (2.5, 1)] + 10 * [(0.25, 0.25)]
        self.arms = self.optimal_arms + self.suboptimal_arms
        pareto_indices = [self.arms.index(arm) for arm in self.optimal_arms]
        reference_point = np.array([6, 6])
        # transform each arm by inverting all the means
        inverted_arms = [(5 - arm[0], 5 - arm[1]) for arm in self.arms]
        super().__init__(len(self.arms), 2, pareto_indices, inverted_arms, reference_point)
        # Generate uniformly distributed std's between 1 and 2 for each arm and objective
        self.stds = [(np.random.uniform(std_low, std_high), np.random.uniform(std_low, std_high)) for _ in range(self.num_arms)]

    def pull_arm(self, arm):
        """
        Pull the specified arm and return the reward.
        :param arm: The index of the arm to pull.
        :return: The reward for the pulled arm.
        """
        return [np.random.normal(self.arms[arm][0], self.stds[arm][0]),
                np.random.normal(self.arms[arm][1], self.stds[arm][1])]

    def plot(self):
        """
        Plot the arms and the Pareto front.
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(*zip(*self.arms), label='Arms')
        plt.scatter(*zip(*[self.arms[i] for i in self.pareto_indices]), color='green', label='Pareto Optimal Arms')

        # Draw ellipses around Pareto optimal arms
        for i in self.pareto_indices:
            ellipse = Ellipse(xy=self.arms[i], width=self.stds[i][0], height=self.stds[i][1], edgecolor='green', facecolor='none')
            plt.gca().add_patch(ellipse)

        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.title('N50VS Environment Arms and Pareto Front')
        plt.legend()
        plt.grid()
        plt.show()
