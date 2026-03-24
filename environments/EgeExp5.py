import numpy as np
from environments.BaseEnvironment import BaseEnvironment


def generate_arms():
    """
    Generate the arms for the EgeExp5 environment.
    :return: A list of arms.
    """
    arms = []
    # Generate the first 10 arms in the range [0.2, 0.4]^2
    for i in range(10):
        arm = np.random.uniform(0.2, 0.4, size=2)
        arms.append(tuple(arm))

    # Generate the next 10 arms in the range [0.5, 0.7]^2
    for i in range(10):
        arm = np.random.uniform(0.5, 0.7, size=2)
        arms.append(tuple(arm))

    return np.array(arms)


class EgeExp5(BaseEnvironment):
    """
    Experiment 5: 2 clusters of arms.
    There are 20 arms and 2 objectives.
    We choose mu_1, ..., mu_10 uniformly in [0.2, 0.4]^2 and mu_11, ..., mu_20 uniformly in [0.5, 0.7]^2.
    """

    def __init__(self):
        self.arms = generate_arms()
        self.stds = [0.25, 0.25]  # Standard deviation for the normal distribution
        pareto_indices = BaseEnvironment._compute_pareto_indices(self.arms)
        reference_point = np.array([1.0, 1.0])
        inverted_arms = [(1 - arm[0], 1 - arm[1]) for arm in self.arms]
        super().__init__(len(self.arms), 2, pareto_indices, inverted_arms, reference_point)

    def reset(self):
        self.arms = generate_arms()
        self.pareto_indices = BaseEnvironment._compute_pareto_indices(self.arms)
