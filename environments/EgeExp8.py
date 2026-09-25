import numpy as np
from environments.BaseEnvironment import BaseEnvironment
from environments.distributions import std_from_variance

# Manuscript spec: Gaussian rewards with diagonal covariance sigma^2 I_D, sigma^2=0.25.
GAUSSIAN_VARIANCE = 0.25


def generate_arms():
    """
    Generate the arms for the EgeExp8 environment.
    :return: A numpy array of arms.
    """
    arms = []
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

    def __init__(self, dist: str = "gaussian", noise_scale: float = 1.0):
        self.dist = dist
        self.noise_scale = noise_scale
        self._init_standard_2obj(
            generate_arms(), dist=dist,
            gaussian_std=std_from_variance(GAUSSIAN_VARIANCE) * noise_scale)
