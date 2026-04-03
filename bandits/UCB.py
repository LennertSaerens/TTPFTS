import numpy as np
import random

from paretoset import paretoset
from bandits.InterfaceMOMABPFI import BaseMOMABAlgorithm


class PUCB1Bandit(BaseMOMABAlgorithm):
    """
    Empirical Pareto UCB1 Bandit
    From "Designing multi-objective multi-armed bandits algorithms: a study" by Madalina M. Drugan and Ann Nowe
    """

    def __init__(self, num_arms, num_objectives, kappa=1):
        super().__init__(num_arms, num_objectives)
        self.kappa = kappa
        self.n = 0
        self.arm_means = np.zeros((num_arms, num_objectives))
        self.arm_counts = np.zeros((num_arms, num_objectives))
        self.current_init_arm = 0
        self._pareto_sense = ["max"] * num_objectives
        # Pre-compute the constant part of the UCB log term
        self._log_arm_obj_factor = 0.25 * np.log(num_objectives * num_arms)

    def choose_arm(self):
        """Choose an arm based on the Upper Confidence Bound (UCB) strategy."""
        if np.all(self.arm_counts > 0):
            ucb_values = self.arm_means + self.kappa * np.sqrt(
                (2 * (np.log(self.n) + self._log_arm_obj_factor)) / self.arm_counts
            )
            pareto_mask = paretoset(ucb_values, sense=self._pareto_sense)
            pareto_indices = np.where(pareto_mask)[0]
            arm = random.choice(pareto_indices)
        else:
            arm = self.current_init_arm
            self.current_init_arm += 1
        self.arm_counts[arm] += 1
        self.n += 1
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
        self.n = 0
        self.arm_means = np.zeros((self.num_arms, self.num_objectives))
        self.arm_counts = np.zeros((self.num_arms, self.num_objectives))
        self.current_init_arm = 0
