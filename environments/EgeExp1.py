import numpy as np
from environments.BaseEnvironment import BaseEnvironment


def generate_arms():
    """
    Generate the arms for the EgeExp1 environment.
    :return: A numpy array of arms.
    """
    arms1 = [(x ** 2, 1 / (4 * x ** 2)) for x in np.linspace(0.55, 0.95, 10)]
    arms2 = []
    while len(arms2) < 50:
        x = np.random.uniform(0.1, 0.8)
        y = np.random.uniform(0.1, 0.8)
        if x * y <= 0.2:
            arms2.append((x, y))
    return np.array(arms1 + arms2)


class EgeExp1(BaseEnvironment):
    """
    Experiment 1: Arms on a convex Pareto set.
    There are 60 arms and 2 objectives.
    We choose x_1, ..., x_10 equally spaced in [0.55, 0.95] and for i = 1, ..., 10 we set mu_i = (x_i^2, 1/(4x_i^2)).
    mu_11, ..., mu_60 are chosen from {(x, y) \in [0.1, 0.8]^2 | xy <= 0.2}.
    """

    def __init__(self):
        pareto_indices = np.arange(10)  # The first 10 arms are Pareto optimal
        self._init_standard_2obj(generate_arms(), pareto_indices)

    def reset(self) -> None:
        self.arms = generate_arms()
        self.inverted_arms = 1.0 - self.arms
