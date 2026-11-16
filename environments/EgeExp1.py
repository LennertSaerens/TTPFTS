import numpy as np
from environments.BaseEnvironment import BaseEnvironment
from environments.distributions import std_from_variance
from environments.instances.frozen_instance_loader import load_frozen_arms

# Manuscript spec: Gaussian rewards with diagonal covariance sigma^2 I_D, sigma^2=0.25.
GAUSSIAN_VARIANCE = 0.25


def generate_arms():
    """
    Non-frozen, random re-derivation of the EgeExp1 arm-generating process
    (kept for reference / regenerating the frozen instance; NOT used by
    __init__/reset(), which load the preregistered frozen instance instead
    -- see environments/instances/freeze_instances.py).
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

    For the paper's final experiments, arms 11-60 are NOT redrawn on every
    instantiation/reset(): a single instance was frozen from master seed 0
    (see environments/instances/freeze_instances.py and CHECKSUMS.sha256) so
    that every trajectory/algorithm faces the exact same problem instance.
    """

    def __init__(self, dist: str = "gaussian"):
        self.dist = dist
        pareto_indices = np.arange(10)  # The first 10 arms are Pareto optimal
        frozen_arms = load_frozen_arms("EgeExp1")
        self._init_standard_2obj(frozen_arms, pareto_indices, dist=dist,
                                 gaussian_std=std_from_variance(GAUSSIAN_VARIANCE))

    def reset(self) -> None:
        """Re-derive the distributions from the same frozen arm means.

        Paper mode: does NOT redraw arm means. This only exists so that
        callers seeding a fresh RNG stream around reset() (see
        momab_main.py's common-random-numbers seeding) get a clean,
        deterministic re-initialization of this environment's state.
        """
        pareto_indices = np.arange(10)
        frozen_arms = load_frozen_arms("EgeExp1")
        self._init_standard_2obj(frozen_arms, pareto_indices, dist=self.dist,
                                 gaussian_std=std_from_variance(GAUSSIAN_VARIANCE))
