import numpy as np
from environments.BaseEnvironment import BaseEnvironment


def generate_arms():
    """
    Generate the arms for the EgeExp4 environment.
    :return: A list of arms.
    """
    arms = []
    # Generate the first 30 arms in the range [0.2, 0.45]^10
    for i in range(30):
        arm = np.random.uniform(0.2, 0.45, size=10)
        arms.append(tuple(arm))

    # Generate the next 20 arms in the range [0.55, 0.75]^10
    for i in range(20):
        arm = np.random.uniform(0.55, 0.75, size=10)
        arms.append(tuple(arm))

    return np.array(arms)


class EgeExp4(BaseEnvironment):
    """
    Experiment 4: High dimension
    There are 50 arms and 10 objectives.
    We generate mu_1,...,mu_30 uniformly in [0.2, 0.45]^10 and mu_31,...,mu_50 uniformly in [0.55, 0.75]^10
    """

    def __init__(self):
        self.arms = generate_arms()
        self.stds = [0.25] * 10  # Standard deviation for the normal distribution
        is_strictly_worse = np.all(self.arms[:, None, :] < self.arms[None, :, :], axis=2)
        pareto_indices = np.where(~np.any(is_strictly_worse, axis=1))[0]
        reference_point = np.array([1.0] * 10)
        inverted_arms = [(1 - arm[i] for i in range(10)) for arm in self.arms]
        super().__init__(len(self.arms), 10, pareto_indices, inverted_arms, reference_point)

    def pull_arm(self, arm):
        """
        Pull the specified arm and return the reward.
        :param arm: The index of the arm to pull.
        :return: The reward for the pulled arm.
        """
        mu = self.arms[arm]
        return [np.random.normal(mu[i], self.stds[i]) for i in range(10)]

    def plot(self, save_png=False, save_file=None):
        """
        Plot the arms and the Pareto front.
        """
        pass

    def reset(self):
        self.arms = generate_arms()
        is_strictly_worse = np.all(self.arms[:, None, :] < self.arms[None, :, :], axis=2)
        self.pareto_indices = np.where(~np.any(is_strictly_worse, axis=1))[0]
