from bandits.InterfaceMOMABPFI import BaseMOMABAlgorithm

import math
import numpy as np


def calc_round_budget(r: int, total_budget: int, num_arms: int) -> int:
    """
    Calculate the budget for the current round.
    :param r: The current round number.
    :param total_budget: The total budget available.
    :param num_arms: The number of arms.
    :return: The budget for the current round.
    """
    return math.ceil(
        (1 / np.log(num_arms)) * ((total_budget - num_arms) / (num_arms + 1 - r))
    ) - math.ceil(
        (1 / np.log(num_arms)) * ((total_budget - num_arms) / (num_arms + 1 - r + 1))
    )


class EGE(BaseMOMABAlgorithm):
    """
    Empirical Gap Elimination Bandit
    From "Bandit Pareto Set Identification: the Fixed Budget Setting" by Kone et al.
    """
    def __init__(self, num_arms, num_objectives, budget):
        super().__init__(num_arms, num_objectives)
        self.budget = budget

        self.arm_means = np.zeros((num_arms, num_objectives))
        self.total_arm_counts = np.zeros((num_arms, num_objectives))
        self.round_arm_counts = np.zeros((num_arms, num_objectives))

        self.active_arms = np.arange(num_arms)
        self.optimal_arms = []
        self.suboptimal_arms = []

        self.curr_round = 1
        self.round_budget = calc_round_budget(self.curr_round, self.budget, self.num_arms)
        self.current_arm = 0

    def choose_arm(self):
        """
        Choose an arm to pull based on the Empirical Gap Elimination (EGE) strategy.
        :return: The arm to pull.
        """
        # Check whether each of the active arms has been pulled round_budget times
        if np.all(self.round_arm_counts[self.active_arms] >= self.round_budget):
            self.go_to_next_round()

        arm = self.active_arms[self.current_arm]
        # Increment the pull count for the current arm
        self.round_arm_counts[arm] += 1
        self.total_arm_counts[arm] += 1
        # Update the current arm to the next one in the active arms
        self.current_arm = (self.current_arm + 1) % len(self.active_arms)
        return arm

    def get_top_arms(self):
        """
        Get the arms that are considered to be Pareto optimal by the bandit.
        :return: The top arms.
        """
        return np.union1d(self.active_arms, self.optimal_arms)

    def learn(self, arm, reward):
        """
        Update the model with the observed reward from the chosen arm.
        :param arm: The arm that was pulled.
        :param reward: The observed reward.
        """
        self.arm_means[arm] += (reward - self.arm_means[arm]) / self.total_arm_counts[arm]

    def reset(self):
        """
        Reset the internal state of the algorithm.
        """
        self.arm_means = np.zeros((self.num_arms, self.num_objectives))
        self.total_arm_counts = np.zeros((self.num_arms, self.num_objectives))
        self.active_arms = np.arange(self.num_arms)
        self.optimal_arms = []
        self.suboptimal_arms = []

    def min_gap(self, arm_i: int, arm_j: int) -> float:
        """
        Calculate the minimum gap between two arms.
        :param arm_i: The first arm.
        :param arm_j: The second arm.
        :return: The minimum gap between the two arms.
        """
        return np.min(self.arm_means[arm_j] - self.arm_means[arm_i])

    def max_gap(self, arm_i: int, arm_j: int) -> float:
        """
        Calculate the maximum gap between two arms.
        :param arm_i: The first arm.
        :param arm_j: The second arm.
        :return: The maximum gap between the two arms.
        """
        return np.max(self.arm_means[arm_i] - self.arm_means[arm_j])

    def go_to_next_round(self) -> None:
        """
        Move to the next round of the algorithm.
        """
        self.update_active_arms()

        self.curr_round += 1
        self.round_budget = calc_round_budget(self.curr_round, self.budget, self.num_arms)
        self.round_arm_counts = np.zeros(len(self.active_arms))

    def update_active_arms(self) -> None:
        """
        Update the active arms based on the current arm means.
        """
        max_arms = self.num_arms - self.curr_round

        # Calculate the empirical Pareto set
        is_strictly_worse = np.all(self.arm_means[:, None, :] < self.arm_means[None, :, :], axis=2)
        emp_Pareto_arms = np.where(~np.any(is_strictly_worse, axis=1))[0]

        diff = np.setdiff1d(self.active_arms, emp_Pareto_arms)

        empirical_gaps = np.zeros(len(self.active_arms))
        for i, arm in enumerate(self.active_arms):
            if arm in diff:
                empirical_gaps[i] = np.max([self.min_gap(arm, arm_j) for arm_j in self.active_arms if arm_j != arm])
            else:
                empirical_gaps[i] = np.min([self.max_gap(arm, arm_j) for arm_j in self.active_arms if arm_j != arm])

        # Sort the arms by increasing empirical gaps
        sorted_indices = np.argsort(empirical_gaps)
        sorted_arms = self.active_arms[sorted_indices]

        # Select the top arms based on the maximum number of arms allowed
        new_active_arms = sorted_arms[:max_arms]
        active_diff = np.setdiff1d(self.active_arms, new_active_arms)
        # Update the active arms
        self.active_arms = new_active_arms

        self.optimal_arms = np.union1d(self.optimal_arms, (np.intersect1d(emp_Pareto_arms, active_diff)))
        self.suboptimal_arms = np.union1d(self.suboptimal_arms, np.setdiff1d(active_diff, emp_Pareto_arms))