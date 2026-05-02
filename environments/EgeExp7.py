import numpy as np
from environments.BaseEnvironment import BaseEnvironment


def generate_arms():
    """
    Generate the arms for the EgeExp7 environment.
    :return: A numpy array of arms.
    """
    arms = []
    c = 0.05
    for i in range(1, 9):
        c_i = (i - 1) * c
        arms.append((0.3 + c_i, 0.8 - c_i))

    for i in range(9, 16):
        c_i = (i - 8) * c
        arms.append((0.25 + c_i, 0.7 - c_i))

    for i in range(16, 23):
        mu_prev = arms[i - 8]
        arms.append((mu_prev[0], mu_prev[1] - 0.05))

    return np.array(arms)


class EgeExp7(BaseEnvironment):
    """
    Experiment 7: All the arms have the same sub-optimality gap.
    There are 22 arms and 2 objectives.
    For i = 1,...,8 we set mu_i = (0.3+c_i,0.8−c_i) where c_i = (i−1)*c
    For i = 9,...,15 we set mu_i = (0.25+c_(i−8),0.7−c_(i−8))
    For i = 16,...,22 we set mu_i = mu_(i-7) - (0,-0.05).
    """

    def __init__(self, dist: str = "gaussian"):
        self.dist = dist
        self._init_standard_2obj(generate_arms(), dist=dist)
