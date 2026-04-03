import numpy as np
from environments.BaseEnvironment import BaseEnvironment
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


class N50VS(BaseEnvironment):
    def __init__(self, std_low=1, std_high=3):
        self.optimal_arms = np.array([(1, 4.5), (1.5, 3.5), (3, 3), (3.5, 1.5), (4.5, 1), (2, 3.1), (3.1, 2)])
        self.suboptimal_arms = np.array([
            (0.5, 4), (1, 3), (1, 2), (1, 1), (2, 1), (2.5, 2.5), (3, 1), (4, 0.5), (0.5, 0.5), (0.5, 1),
            (0.5, 2), (0.5, 3), (1, 0.5), (2, 0.5), (3, 0.5), (2, 2), (1.5, 2.5), (2.5, 1.5), (1.5, 1.5),
            (0.5, 3.5), (3.5, 0.5), (2, 2.5), (2.5, 2), (0.5, 1.5), (1.5, 0.5), (1.5, 2), (2, 1.5),
            (0.5, 2.5), (2.5, 0.5), (1, 1.5), (1.5, 1), (1, 2.5), (2.5, 1),
        ] + 10 * [(0.25, 0.25)])
        self.arms = np.vstack([self.optimal_arms, self.suboptimal_arms])
        num_arms = len(self.arms)
        pareto_indices = np.arange(len(self.optimal_arms))
        reference_point = np.array([6, 6])
        inverted_arms = 5.0 - self.arms
        super().__init__(num_arms, 2, pareto_indices, inverted_arms, reference_point)
        self.std_low = std_low
        self.std_high = std_high
        self.stds = np.random.uniform(std_low, std_high, size=(self.num_arms, 2))

    def pull_arm(self, arm):
        """Pull the specified arm and return the reward with per-arm noise."""
        return np.random.normal(self.arms[arm], self.stds[arm])

    def plot(self, save_png=False, save_file=None):
        """Plot the arms and the Pareto front."""
        plt.figure(figsize=(6, 4))

        plt.scatter(*zip(*self.arms), label='Suboptimal Arms')
        plt.scatter(*zip(*[self.arms[i] for i in self.pareto_indices]), color='green', label='Optimal Arms')

        for i in self.pareto_indices:
            ellipse = Ellipse(xy=self.arms[i], width=2*self.stds[i][0], height=2*self.stds[i][1],
                              edgecolor='green', facecolor='none', alpha=0.5)
            plt.gca().add_patch(ellipse)

        plt.xlabel('Objective 1')
        plt.ylabel('Objective 2')
        plt.legend()
        plt.grid()
        if save_png and save_file is not None:
            plt.savefig(save_file, format='png', dpi=300)
        plt.show()

    def reset(self):
        self.arms = np.vstack([self.optimal_arms, self.suboptimal_arms])
        self.stds = np.random.uniform(self.std_low, self.std_high, size=(self.num_arms, 2))
