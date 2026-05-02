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
    return np.array(arms)


class EgeExp2(BaseEnvironment):
    """
    Experiment 2: Group of suboptimal arms where each suboptimal arm i has a unique i⋆
    There are 10 arms and 2 objectives.
    mu_1 = (0.4, 0.75), mu_2 = (0.75, 0.4)
    for i = 1,...,4 we set mu_(2i+1) = (0.45 + 0.2^i, 0.35 − 0.2^i), mu_(2i+2) = (0.10 + 0.2^i, 0.70 − 0.2^i)
    """

    def __init__(self, dist: str = "gaussian"):
        self.dist = dist
        pareto_indices = np.arange(2)  # The first 2 arms are Pareto optimal
        self._init_standard_2obj(generate_arms(), pareto_indices, dist=dist)
