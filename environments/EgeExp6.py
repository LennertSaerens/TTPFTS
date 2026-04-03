import numpy as np
from environments.BaseEnvironment import BaseEnvironment


def generate_arms():
    """
    Generate the arms for the EgeExp6 environment.
    :return: A numpy array of arms.
    """
    arms = []
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
        pareto_indices = np.arange(10)  # All arms are Pareto optimal
        self._init_standard_2obj(generate_arms(), pareto_indices)
