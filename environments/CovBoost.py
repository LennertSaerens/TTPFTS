import numpy as np
from environments.BaseEnvironment import BaseEnvironment
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


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
        pareto_indices = np.arange(10)  # The first 10 arms are Pareto optimal
        reference_point = np.array([1.0, 1.0])
        inverted_arms = [(1 - arm[0], 1 - arm[1]) for arm in self.arms]
        super().__init__(len(self.arms), 3, pareto_indices, inverted_arms, reference_point)

    def pull_arm(self, arm):
        """
        Pull the specified arm and return the reward.
        :param arm: The index of the arm to pull.
        :return: The reward for the pulled arm.
        """
        mu = self.arms[arm]
        return [np.random.normal(mu[i], self.stds[i]) for i in range(3)]

    def plot(self):
        """
        Plot the arms and the Pareto front.
        """
        pass