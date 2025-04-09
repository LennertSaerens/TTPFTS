import numpy as np


class Auer2013Bandit:
    """
    Implements the MOMAB PFI algorithm described by Auer et al. (2013) in the paper called "Pareto Front Identification
     from Stochastic Bandit Feedback".

    The algorithm is an elimination algorithm based on confidence intervals, that maintains a set A of active_arms arms.
    It eliminates points that are suboptimal with high probability, or points that are (almost) Pareto optimal with
    high probability (set P1). Optimal points in P1 are not removed, though, if they are needed to establish the
    status of other points they may dominate. The algorithm stops when no active_arms operating points remain and outputs
    the set of (almost) Pareto optimal points.
    """

    def __init__(self, num_arms: int, num_objectives: int, epsilon: float, delta: float):
        """
        Initialize the Auer2013Bandit instance.

        :param num_arms: Number of arms in the bandit problem.
        :param epsilon: Accuracy parameter.
        :param delta: Confidence parameter.
        """
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.epsilon = epsilon
        self.delta = delta

        self.active_arms = np.arange(num_arms)
        self.optimal = []

        self.arm_means = np.zeros((num_arms, num_objectives))
        self.itt_arm_counts = np.zeros((num_arms, num_objectives))
        self.total_arm_counts = np.zeros((num_arms, num_objectives))
        self.current_arm = 0

    def choose_arm(self):
        """
        Choose an arm to pull based on the Auer2013 algorithm.

        :return: The arm to pull.
        """
        if len(self.active_arms) == 0:
            # If there are no active arms left, return 0
            return 0

        # Check if all active arms have been pulled once
        if np.all(self.itt_arm_counts[self.active_arms] > 0):
            # If all active arms have been pulled once, update the optimal and active arms
            self.update_optimal_and_active_arms()

        if len(self.active_arms) == 0:
            # If there are no active arms left, return 0
            return 0

        arm = self.active_arms[self.current_arm]
        self.itt_arm_counts[arm] += 1
        self.total_arm_counts[arm] += 1
        self.current_arm = (self.current_arm + 1) % len(self.active_arms)
        return arm

    def update_optimal_and_active_arms(self):
        """
        Update the optimal and active arms based on the strategy described in the Auer2013 paper.

        Pseudocode:
        A_1 = {i \in A | \forall j \in A, min_gap(i,j) <= CI(i) + CI(j)}
        P_1 = {i \in A_1 | \forall j \in A_1 \ {i}, max_gap(i,j) >= CI(i) + CI(j)}
        P_2 = {j \in P_1 | \nexists i \in A_1 \ {P_1}, max_gap(i,j) <= CI(i) + CI(j)}
        A = A_1 \ {P_2}
        P = P_2 \cup P_1

        :return: None
        """
        # print(f"Arm means:\n {self.arm_means}")
        # print(f"Total arm counts:\n {self.total_arm_counts}")
        # Get the active_arms arms i for which the min_gap between the arm and all other active_arms arms j is
        # lesser than or equal to CI(i) + CI(j)
        A_1 = []
        for i in self.active_arms:
            valid = True
            for j in self.active_arms:
                if i != j:
                    min_gap = self.get_min_gap(i, j)
                    ci_sum = (self.get_arm_ci(i) + self.get_arm_ci(j))[0]
                    # print(f"min_gap({i}, {j}): {min_gap}, CI({i}) + CI({j}): {ci_sum}")
                    if min_gap > ci_sum:
                        valid = False
                        break
            if valid:
                A_1.append(i)

        # print(f"A_1: {A_1}")

        # Get the arms i from A_1 \ {i} for which the max_gap between the arm and all other active_arms arms j is
        # greater than or equal to CI(i) + CI(j)
        P_1 = []
        for i in A_1:
            valid = True
            for j in A_1:
                if i != j:
                    max_gap = self.get_eps_max_gap(i, j)
                    ci_sum = (self.get_arm_ci(i) + self.get_arm_ci(j))[0]
                    # print(f"max_gap({i}, {j}): {max_gap}, CI({i}) + CI({j}): {ci_sum}")
                    if max_gap < ci_sum:
                        valid = False
                        break
            if valid:
                P_1.append(i)

        # print(f"P_1: {P_1}")

        # Get the arms j from P_1 for which there exists no arm i in A_1 \ P_1 such that the max_gap
        # between i and j is less than or equal to CI(i) + CI(j)
        P_2 = []
        for j in P_1:
            valid = True
            for i in np.setdiff1d(A_1, P_1):
                if i != j:
                    max_gap = self.get_eps_max_gap(i, j)
                    ci_sum = (self.get_arm_ci(i) + self.get_arm_ci(j))[0]
                    # print(f"max_gap({i}, {j}): {max_gap}, CI({i}) + CI({j}): {ci_sum}")
                    if max_gap <= ci_sum:
                        valid = False
                        break
            if valid:
                P_2.append(j)

        # print(f"P_2: {P_2}")

        # Set the active_arms arms to the difference between A_1 and P_2
        self.active_arms = np.setdiff1d(A_1, P_2)
        # print(f"New active_arms: {self.active_arms}")
        # Set the optimal arms to the union of the current optimal arms and P_2
        self.optimal = np.union1d(self.optimal, P_2)
        # print(f"New optimal arms: {self.optimal}")

        # Reset the arm counts
        self.itt_arm_counts[self.active_arms] = 0
        # Reset the current arm to 0
        self.current_arm = 0

    def get_arm_ci(self, arm: int):
        """
        Calculate the confidence interval for the given arm.

        :param arm: The arm to calculate the confidence interval for.
        :return: The confidence interval for the arm.
        """
        ci = np.sqrt(
            (2 * np.log(
                4 * self.num_arms * self.num_objectives * self.total_arm_counts[arm] ** 2 / self.delta
            )) / self.total_arm_counts[arm]
        )
        return ci

    def get_min_gap(self, arm_i: int, arm_j: int):
        """
        Calculate by how much an arm i is dominated by an arm j.

        :param arm_i: The first arm.
        :param arm_j: The second arm.
        :return: The gap between the two arms.
        """
        # Find the smallest difference between the two arms across all objectives
        return max(0, np.min(self.arm_means[arm_j] - self.arm_means[arm_i]))

    def get_eps_max_gap(self, arm_i: int, arm_j: int):
        """
        Calculate the maximum gap between two arms.

        :param arm_i: The first arm.
        :param arm_j: The second arm.
        :return: The maximum gap between the two arms.
        """
        return max(0, np.max((self.arm_means[arm_i] + self.epsilon) - self.arm_means[arm_j]))

    def get_top_arms(self):
        return self.optimal

    def learn(self, arm: int, reward: float):
        """
        Learn from the reward that was received for pulling the arm.

        :param arm: The arm that was pulled.
        :param reward: The reward for each objective.
        :return: None
        """
        # Update the arm means based on the reward received
        self.arm_means[arm] += (reward - self.arm_means[arm]) / self.total_arm_counts[arm]

    def reset(self):
        """
        Reset the bandit instance to its initial state.
        """
        self.arm_means = np.zeros((self.num_arms, self.num_objectives))
        self.itt_arm_counts = np.zeros((self.num_arms, self.num_objectives))
        self.total_arm_counts = np.zeros((self.num_arms, self.num_objectives))
        self.active_arms = np.arange(self.num_arms)
        self.optimal = []
        self.current_arm = 0
