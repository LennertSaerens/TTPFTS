import numpy as np
from environments.BaseEnvironment import BaseEnvironment


def generate_arms():
    """
    Generate the arms for the EgeExp8 environment.
    :return: A list of arms.
    """
    arms = []
    # Generate arms with the specified formula
    for i in range(1, 6):
        mu_i = (0.75 - 0.25 ** i)
        arms.append((mu_i, mu_i))
    return np.array(arms)


class EgeExp8(BaseEnvironment):
    """
    Experiment 8: Geometric progression with a single optimal arm.
    There are 5 arms and 2 objectives.
    For i = 1,...,5 we set mu_i = [0.75 - 0.25^i]^2
    """

    def __init__(self):
        self.arms = generate_arms()
        self.stds = [0.25, 0.25]  # Standard deviation for the normal distribution
        pareto_indices = BaseEnvironment._compute_pareto_indices(self.arms)
        reference_point = np.array([1.0, 1.0])
        inverted_arms = [(1 - arm[0], 1 - arm[1]) for arm in self.arms]
        super().__init__(len(self.arms), 2, pareto_indices, inverted_arms, reference_point)
