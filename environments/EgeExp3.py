import numpy as np
from environments.BaseEnvironment import BaseEnvironment


def generate_arms():
    """
    Generate the arms for the EgeExp3 environment.
    :return: A numpy array of arms.
    """
    arms = []
    for i in range(20):
        angle = np.pi / 12 + i * (np.pi / 3) / 19
        arms.append((np.cos(angle), np.sin(angle)))

    for i in range(20, 200):
        angle = 4 * np.pi / 6 + i * (7 * np.pi / 6) / 179
        arms.append((np.cos(angle), np.sin(angle)))

    return np.array(arms)


class EgeExp3(BaseEnvironment):
    """
    Experiment 3: Many arms on the unit circle.
    There are 200 arms and 2 objectives.
    We choose b_1, ..., b_20 evenly spaced in [π/12, π/2 - π/12] and b_21, ..., b_200 evenly spaced in [π/2 + π/6, 2π - π/6].
    For i = 1, ..., 200 we set mu_i = (cos(b_i), sin(b_i))
    """

    def __init__(self, dist: str = "gaussian"):
        self.dist = dist
        pareto_indices = np.arange(20)  # The first 20 arms are Pareto optimal
        self._init_standard_2obj(generate_arms(), pareto_indices, dist=dist)
