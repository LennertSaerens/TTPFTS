import numpy as np
from paretoset import paretoset
from bandits.InterfaceMOMABPFI import BaseMOMABAlgorithm


class UniformBandit(BaseMOMABAlgorithm):
    """Uniform sampling baseline: cycles through all arms in round-robin order."""

    def __init__(self, num_arms, num_objectives):
        super().__init__(num_arms, num_objectives)
        self.arm_means = np.zeros((num_arms, num_objectives))
        self.arm_counts = np.zeros(num_arms)
        self.current_arm = np.random.randint(num_arms)
        self._pareto_sense = ["max"] * num_objectives

    def choose_arm(self):
        """Select the next arm in round-robin order."""
        arm = self.current_arm
        self.arm_counts[arm] += 1
        self.current_arm = (self.current_arm + 1) % self.num_arms
        return arm

    def get_top_arms(self):
        """Get the Pareto optimal arms based on the estimated arm means."""
        pareto_mask = paretoset(self.arm_means, sense=self._pareto_sense)
        return np.where(pareto_mask)[0]

    def learn(self, arm, reward):
        """Update the running mean with the observed reward."""
        self.arm_means[arm] += (reward - self.arm_means[arm]) / self.arm_counts[arm]

    def reset(self, _):
        """Reset the agent."""
        self.arm_means = np.zeros((self.num_arms, self.num_objectives))
        self.arm_counts = np.zeros(self.num_arms)
        self.current_arm = np.random.randint(self.num_arms)
