import numpy as np
from environments.BaseEnvironment import BaseEnvironment
from environments.distributions import make_distributions, std_from_variance
from environments.instances.frozen_instance_loader import load_frozen_arms

# Manuscript spec: Gaussian rewards with diagonal covariance sigma^2 I_D, sigma^2=0.25.
GAUSSIAN_VARIANCE = 0.25


def generate_arms():
    """
    Non-frozen, random re-derivation of the EgeExp4 arm-generating process
    (kept for reference / regenerating the frozen instance; NOT used by
    __init__/reset(), which load the preregistered frozen instance instead
    -- see environments/instances/freeze_instances.py).
    :return: A numpy array of arms.
    """
    arms_low = np.random.uniform(0.2, 0.45, size=(30, 10))
    arms_high = np.random.uniform(0.55, 0.75, size=(20, 10))
    return np.vstack([arms_low, arms_high])


class EgeExp4(BaseEnvironment):
    """
    Experiment 4: High dimension
    There are 50 arms and 10 objectives.
    We generate mu_1,...,mu_30 uniformly in [0.2, 0.45]^10 and mu_31,...,mu_50 uniformly in [0.55, 0.75]^10

    For the paper's final experiments, arm means are NOT redrawn on every
    instantiation/reset(): a single instance was frozen from master seed 0
    (see environments/instances/freeze_instances.py and CHECKSUMS.sha256) so
    that every trajectory/algorithm faces the exact same problem instance.
    """

    def __init__(self, dist: str = "gaussian"):
        self.dist = dist
        arm_means = load_frozen_arms("EgeExp4")
        distributions = make_distributions(arm_means, dist=dist,
                                           gaussian_std=std_from_variance(GAUSSIAN_VARIANCE))
        pareto_indices = BaseEnvironment._compute_pareto_indices(arm_means)
        reference_point = np.array([1.0] * 10)
        inverted_arms = 1.0 - arm_means
        super().__init__(len(arm_means), 10, pareto_indices, inverted_arms, reference_point,
                         distributions=distributions)

    def plot(self, save_png=False, save_file=None):
        """Cannot plot 10-dimensional arms."""
        pass

    def reset(self):
        """Re-derive the distributions from the same frozen arm means.

        Paper mode: does NOT redraw arm means. This only exists so that
        callers seeding a fresh RNG stream around reset() (see
        momab_main.py's common-random-numbers seeding) get a clean,
        deterministic re-initialization of this environment's state.
        """
        arm_means = load_frozen_arms("EgeExp4")
        self.distributions = make_distributions(arm_means, dist=self.dist,
                                                 gaussian_std=std_from_variance(GAUSSIAN_VARIANCE))
        self.pareto_indices = BaseEnvironment._compute_pareto_indices(arm_means)
        self._update_pareto_cache()
        self.inverted_arms = 1.0 - arm_means
