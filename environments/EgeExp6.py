import numpy as np
from environments.BaseEnvironment import BaseEnvironment


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
        self.stds = [0.25, 0.25]  # Standard deviation for the normal distribution
        pareto_indices = np.arange(10)  # All arms are Pareto optimal
        reference_point = np.array([1.0, 1.0])
        inverted_arms = [(1 - arm[0], 1 - arm[1]) for arm in self.arms]
        super().__init__(len(self.arms), 2, pareto_indices, inverted_arms, reference_point)
