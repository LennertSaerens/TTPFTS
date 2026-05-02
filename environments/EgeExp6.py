import numpy as np
from environments.BaseEnvironment import BaseEnvironment


def generate_arms():
    """
    Generate the arms for the EgeExp6 environment.

    Uses 0.65^(i+1) so that all means lie in (0, 1), making the
    environment compatible with bernoulli and exponential rewards
    while preserving the geometric trade-off structure.
    """
    arms = []
    for i in range(10):
        mu_i = (0.75 - 0.65 ** (i + 1), 0.25 + 0.65 ** (i + 1))
        arms.append(mu_i)
    return np.array(arms)


class EgeExp6(BaseEnvironment):
    """
    Experiment 6: All the arms are optimal.
    There are 10 arms and 2 objectives.
    For any arm i, mu_i = (0.75 - 0.65^(i+1), 0.25 + 0.65^(i+1)).
    All arms lie on the line obj1 + obj2 = 1 and form a Pareto front.
    """

    def __init__(self, dist: str = "gaussian"):
        self.dist = dist
        pareto_indices = np.arange(10)  # All arms are Pareto optimal
        self._init_standard_2obj(generate_arms(), pareto_indices, dist=dist)
