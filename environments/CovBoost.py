import matplotlib.pyplot as plt
import numpy as np

from environments.BaseEnvironment import BaseEnvironment


def generate_arms():
    """
    Generate the arms for the CovBoost environment.
    :return: A list of arms.
    """
    immune_responses = [
        (9.50, 6.86, 4.56),  # BNT/BNT → ChAd
        (9.29, 6.64, 4.04),  # BNT/BNT → NVX
        (9.05, 6.41, 3.56),  # BNT/BNT → NVX Half
        (10.21, 7.49, 4.43),  # BNT/BNT → BNT
        (10.05, 7.20, 4.36),  # BNT/BNT → BNT Half
        (8.34, 5.67, 3.51),  # BNT/BNT → VLA
        (8.22, 5.46, 3.64),  # BNT/BNT → VLA Half
        (9.75, 7.27, 4.71),  # BNT/BNT → Ad26
        (10.43, 7.61, 4.72),  # BNT/BNT → m1273
        (8.94, 6.19, 3.84),  # BNT/BNT → CVn
        (7.81, 5.26, 3.97),  # ChAd/ChAd → ChAd
        (8.85, 6.59, 4.73),  # ChAd/ChAd → NVX
        (8.44, 6.15, 4.59),  # ChAd/ChAd → NVX Half
        (9.93, 7.39, 4.75),  # ChAd/ChAd → BNT
        (8.71, 7.20, 4.91),  # ChAd/ChAd → BNT Half
        (7.51, 5.31, 3.96),  # ChAd/ChAd → VLA
        (7.27, 4.99, 4.02),  # ChAd/ChAd → VLA Half
        (8.62, 6.33, 4.66),  # ChAd/ChAd → Ad26
        (10.35, 7.77, 5.00),  # ChAd/ChAd → m1273
        (8.29, 5.92, 3.87)  # ChAd/ChAd → CVn
    ]
    return np.array(immune_responses)


class CovBoost(BaseEnvironment):
    """
    Bandit setting based on the CovBoost dataset.
    """

    def __init__(self):
        self.arms = generate_arms()
        self.stds = [0.7, 0.83, 1.54]  # Standard deviation for the normal distribution
        pareto_indices = BaseEnvironment._compute_pareto_indices(self.arms)
        reference_point = np.array([1.0, 1.0])
        inverted_arms = [(1 - arm[0], 1 - arm[1]) for arm in self.arms]
        super().__init__(len(self.arms), 3, pareto_indices, inverted_arms, reference_point)

    def plot(self):
        """
        Plot the arms and the Pareto front in 3D.
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Split optimal and suboptimal arms
        suboptimal_indices = np.setdiff1d(np.arange(len(self.arms)), self.pareto_indices)
        pareto_arms = self.arms[self.pareto_indices]
        suboptimal_arms = self.arms[suboptimal_indices]

        # Plot suboptimal arms
        ax.scatter(suboptimal_arms[:, 0], suboptimal_arms[:, 1], suboptimal_arms[:, 2],
                   label='Suboptimal Arms', c='b', s=50)

        # Plot Pareto optimal arms
        ax.scatter(pareto_arms[:, 0], pareto_arms[:, 1], pareto_arms[:, 2],
                   label='Pareto Optimal Arms', c='g', s=80)

        # Draw ellipsoids (stds) around Pareto optimal arms
        for arm in pareto_arms:
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            x = self.stds[0] * np.cos(u) * np.sin(v) + arm[0]
            y = self.stds[1] * np.sin(u) * np.sin(v) + arm[1]
            z = self.stds[2] * np.cos(v) + arm[2]
            ax.plot_wireframe(x, y, z, color='g', alpha=0.2)

        ax.set_xlabel('Anti-spike IgG')
        ax.set_ylabel(r'NT$_{50}$')
        ax.set_zlabel('Cellular Response')
        ax.legend()
        plt.tight_layout()
        plt.savefig('environments/plots/CovBoost.pdf', format='pdf')
        plt.show()
