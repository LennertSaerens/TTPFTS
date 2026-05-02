import numpy as np
from environments.BaseEnvironment import BaseEnvironment
from environments.distributions import GaussianReward
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
        all_arms = np.vstack([self.optimal_arms, self.suboptimal_arms])
        num_arms = len(all_arms)
        pareto_indices = np.arange(len(self.optimal_arms))
        reference_point = np.array([6, 6])
        inverted_arms = 5.0 - all_arms
        self.std_low = std_low
        self.std_high = std_high
        per_arm_stds = np.random.uniform(std_low, std_high, size=(num_arms, 2))
        distributions = [GaussianReward(all_arms[i], per_arm_stds[i]) for i in range(num_arms)]
        super().__init__(num_arms, 2, pareto_indices, inverted_arms, reference_point,
                         distributions=distributions)

    def plot(self, save_png=False, save_file=None):
        """Plot the arms and the Pareto front."""
        plt.figure(figsize=(6, 4))
        arm_means = self.arms
        arm_stds = self.stds

        plt.scatter(*zip(*arm_means), label='Suboptimal Arms')
        plt.scatter(*zip(*[arm_means[i] for i in self.pareto_indices]), color='green', label='Optimal Arms')

        for i in self.pareto_indices:
            ellipse = Ellipse(xy=arm_means[i], width=2*arm_stds[i][0], height=2*arm_stds[i][1],
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
        all_arms = np.vstack([self.optimal_arms, self.suboptimal_arms])
        per_arm_stds = np.random.uniform(self.std_low, self.std_high, size=(self.num_arms, 2))
        self.distributions = [GaussianReward(all_arms[i], per_arm_stds[i]) for i in range(self.num_arms)]
