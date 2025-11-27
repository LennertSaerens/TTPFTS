import numpy as np
from paretoset import paretoset
from bandits.InterfaceMOMABPFI import BaseMOMABAlgorithm


class UniformBandit(BaseMOMABAlgorithm):
    def __init__(self, num_arms, num_objectives):
        super().__init__(num_arms, num_objectives)
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.arm_means = np.zeros((num_arms, num_objectives))
        self.arm_counts = np.zeros(num_arms)
        self.current_arm = np.random.randint(num_arms)

    def choose_arm(self):
        arm = self.current_arm
        self.arm_counts[arm] += 1
        self.current_arm = (self.current_arm + 1) % self.num_arms
        return arm

    def get_top_arms(self):
        pareto_mask = paretoset(self.arm_means, sense=["max"] * self.num_objectives)
        pareto_indices = np.where(pareto_mask)[0]
        return pareto_indices

    def learn(self, arm, reward):
        self.arm_means[arm] += (reward - self.arm_means[arm]) / self.arm_counts[arm]

    def reset(self, _):
        self.arm_means = np.zeros((self.num_arms, self.num_objectives))
        self.arm_counts = np.zeros(self.num_arms)
        self.current_arm = np.random.randint(self.num_arms)
