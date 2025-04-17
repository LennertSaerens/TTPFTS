import numpy as np


class PLUCBBandit:
    """
    Pareto Lower Upper Confidence Bound (PLUCB) Bandit
    From "PAC models in stochastic multi-objective multi-armed bandits" by Madalina M. Drugan
    """

    def __init__(self, num_arms, num_objectives, epsilon, delta):
        self.num_arms = num_arms
        self.num_objectives = num_objectives

        self.epsilon = epsilon
        self.delta = delta

        self.upper_critical_arms = np.zeros((num_arms, num_objectives))  # worst Pareto optimal arms
        self.lower_critical_arms = np.zeros((num_arms, num_objectives))  # best suboptimal arms
        self.optimal = []

        self.arm_means = np.zeros((num_arms, num_objectives))
        self.arm_counts = np.zeros((num_arms, num_objectives))
        self.n = 0
        self.current_init_arm = 0

    def choose_arm(self):
        """
        Choose an arm to pull based on the PLUCB algorithm.

        :return: The arm to pull.
        """
        if np.all(self.arm_counts > 0):
            pass
        else:
            arm = self.current_init_arm
            self.current_init_arm += 1

        self.arm_counts[arm] += 1
        self.n += 1
        return arm

    def compute_upper_critical_arms(self):
        """
        Compute the upper critical arms based on the current arm counts and epsilon.
        """
        pass

    def confidence_bound(self, arm):
        return np.sqrt(
            (1 / (2 * self.arm_counts[arm])) * np.log(4 * self.num_arms * self.num_objectives * self.n**4 / self.delta)
        )

    def get_top_arms(self):
        pass

    def learn(self, arm, reward):
        pass

    def reset(self):
        pass
