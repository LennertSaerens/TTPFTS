import numpy as np
from environments.BaseEnvironment import BaseEnvironment


def generate_arms():
    """
    Generate the arms for the EgeExp2 environment.
    :return: A list of arms.
    """
    arms = [(0.4, 0.75), (0.75, 0.4)]
    for i in range(1, 5):
        arms.append((0.45 + 0.2 ** i, 0.35 - 0.2 ** i))
        arms.append((0.10 + 0.2 ** i, 0.70 - 0.2 ** i))
    return arms


class EgeExp2(BaseEnvironment):
    """
    Experiment 2: Group of suboptimal arms where each suboptimal arm i has a unique i⋆
    There are 10 arms and 2 objectives.
    mu_1 = (0.4, 0.75), mu_2 = (0.75, 0.4)
    for i = 1,...,4 we set mu_(2i+1) = (0.45 + 0.2^i, 0.35 − 0.2^i), mu_(2i+2) = (0.10 + 0.2^i, 0.70 − 0.2^i)
    """

    def __init__(self):
        self.arms = generate_arms()
        self.stds = [0.25, 0.25]  # Standard deviation for the normal distribution
        pareto_indices = np.arange(2)  # The first 2 arms are Pareto optimal
        reference_point = np.array([1.0, 1.0])
        inverted_arms = [(1 - arm[0], 1 - arm[1]) for arm in self.arms]
        super().__init__(len(self.arms), 2, pareto_indices, inverted_arms, reference_point)
