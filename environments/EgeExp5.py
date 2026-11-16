import numpy as np
from environments.BaseEnvironment import BaseEnvironment
from environments.distributions import std_from_variance
from environments.instances.frozen_instance_loader import load_frozen_arms

# Manuscript spec: Gaussian rewards with diagonal covariance sigma^2 I_D, sigma^2=0.25.
GAUSSIAN_VARIANCE = 0.25


def generate_arms():
    """
    Non-frozen, random re-derivation of the EgeExp5 arm-generating process
    (kept for reference / regenerating the frozen instance; NOT used by
    __init__/reset(), which load the preregistered frozen instance instead
    -- see environments/instances/freeze_instances.py).
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

    For the paper's final experiments, arm means are NOT redrawn on every
    instantiation/reset(): a single instance was frozen from master seed 0
    (see environments/instances/freeze_instances.py and CHECKSUMS.sha256) so
    that every trajectory/algorithm faces the exact same problem instance.
    The Pareto set is computed from the frozen means (not hardcoded), since
    with two well-separated uniform clusters the low cluster is generically
    non-dominated in one objective and the high cluster in the other.
    """

    def __init__(self, dist: str = "gaussian"):
        self.dist = dist
        frozen_arms = load_frozen_arms("EgeExp5")
        self._init_standard_2obj(frozen_arms, dist=dist,
                                 gaussian_std=std_from_variance(GAUSSIAN_VARIANCE))

    def reset(self):
        """Re-derive the distributions from the same frozen arm means.

        Paper mode: does NOT redraw arm means. This only exists so that
        callers seeding a fresh RNG stream around reset() (see
        momab_main.py's common-random-numbers seeding) get a clean,
        deterministic re-initialization of this environment's state.
        """
        frozen_arms = load_frozen_arms("EgeExp5")
        self._init_standard_2obj(frozen_arms, dist=self.dist,
                                 gaussian_std=std_from_variance(GAUSSIAN_VARIANCE))
