import numpy as np
from environments.BaseEnvironment import BaseEnvironment


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
        self.stds = [0.25, 0.25]  # Standard deviation for the normal distribution
        pareto_indices = BaseEnvironment._compute_pareto_indices(self.arms)
        reference_point = np.array([1.0, 1.0])
        inverted_arms = [(1 - arm[0], 1 - arm[1]) for arm in self.arms]
        super().__init__(len(self.arms), 2, pareto_indices, inverted_arms, reference_point)
