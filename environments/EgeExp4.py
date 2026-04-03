import numpy as np
from environments.BaseEnvironment import BaseEnvironment


def generate_arms():
    """
    Generate the arms for the EgeExp4 environment.
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
    """

    def __init__(self):
        self.arms = generate_arms()
        self.stds = np.array([0.25] * 10)
        pareto_indices = BaseEnvironment._compute_pareto_indices(self.arms)
        reference_point = np.array([1.0] * 10)
        inverted_arms = 1.0 - self.arms
        super().__init__(len(self.arms), 10, pareto_indices, inverted_arms, reference_point)

    def plot(self, save_png=False, save_file=None):
        """Cannot plot 10-dimensional arms."""
        pass

    def reset(self):
        self.arms = generate_arms()
        self.pareto_indices = BaseEnvironment._compute_pareto_indices(self.arms)
        self._update_pareto_cache()
        self.inverted_arms = 1.0 - self.arms
