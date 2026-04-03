import numpy as np
from environments.BaseEnvironment import BaseEnvironment


def generate_arms():
    """
    Generate the arms for the EgeExp5 environment.
    :return: A numpy array of arms.
    """
    arms_low = np.random.uniform(0.2, 0.4, size=(10, 2))
    arms_high = np.random.uniform(0.5, 0.7, size=(10, 2))
    return np.vstack([arms_low, arms_high])


class EgeExp5(BaseEnvironment):
    """
    Experiment 5: 2 clusters of arms.
    There are 20 arms and 2 objectives.
    We choose mu_1, ..., mu_10 uniformly in [0.2, 0.4]^2 and mu_11, ..., mu_20 uniformly in [0.5, 0.7]^2.
    """

    def __init__(self):
        self._init_standard_2obj(generate_arms())

    def reset(self):
        self._init_standard_2obj(generate_arms())
