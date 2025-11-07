import numpy as np
import random
from scipy.stats import invgamma, norm
from paretoset import paretoset


class TTPFTSBandit:
    """
    Top Two Pareto Front Thompson Sampling Bandit

    MO adaptation of the Top Two Thompson Sampling Bandit for expansion of the best-arm identification problem to the
    multi-objective case: Pareto front identification.
    """

    def __init__(self, num_arms, num_objectives, p):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.alphas = np.ones((num_arms, num_objectives))
        self.betas = np.ones((num_arms, num_objectives))
        self.p = p

    def choose_arm(self):
        """
        Find all Pareto optimal arms. With a chance of p, return one of them. Otherwise, find the non-dominated arms
        in the non-Pareto optimal set and return one of them.
        :return: The arm to pull.
        """
        # 1. Sample from the posterior
        samples = np.random.beta(self.alphas, self.betas)

        # 2. Find Pareto front
        pareto_mask = paretoset(samples, sense=["max"] * self.num_objectives)
        pareto_indices = np.where(pareto_mask)[0]

        # 3. Decide which set to pull from
        if np.random.random() < self.p:
            return random.choice(pareto_indices)
        else:
            non_pareto_indices = np.where(~pareto_mask)[0]
            if len(non_pareto_indices) == 0:
                return random.choice(pareto_indices)

            non_pareto_samples = samples[non_pareto_indices]
            non_pareto_pareto_mask = paretoset(non_pareto_samples, sense=["max"] * self.num_objectives)
            non_dominated_indices = np.where(non_pareto_pareto_mask)[0]

            # Map non-dominated indices back to original arm indices
            non_dominated_indices = non_pareto_indices[non_dominated_indices]

            return random.choice(non_dominated_indices)

    def get_top_arms(self):
        """
        Get the arms that are considered to be Pareto optimal by the bandit.
        :return: The top arms.
        """
        pareto_mask = paretoset(self.alphas / (self.alphas + self.betas), sense=["max"] * self.num_objectives)
        pareto_indices = np.where(pareto_mask)[0]
        return pareto_indices

    def learn(self, arm, reward):
        """
        Learn from the reward that was received for pulling the arm.
        :param arm: The arm that was pulled.
        :param reward: The reward for each objective.
        :return: None
        """
        for o in range(self.num_objectives):
            self.alphas[arm][o] += reward[o]
            self.betas[arm][o] += 1 - reward[o]

    def reset(self):
        """
        Reset the agent.
        :return: None
        """
        self.alphas = np.ones((self.num_arms, self.num_objectives))
        self.betas = np.ones((self.num_arms, self.num_objectives))


class NormalTTPFTSBandit:
    """
    Variant of the TTPFTSBandit that uses Normal-Inverse-Gamma Distribution instead of Beta.
    """

    def __init__(self, num_arms, num_objectives, p):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.mu = np.zeros((num_arms, num_objectives))  # mean
        self.lambdas = np.ones((num_arms, num_objectives))  # precision
        self.alpha = np.full((num_arms, num_objectives), 2.0, dtype=np.float64)  # shape
        self.beta = np.full((num_arms, num_objectives), 2.0, dtype=np.float64)  # scale
        self.p = p

    def choose_arm(self):
        """
        Find all Pareto optimal arms. With a chance of p, return one of them. Otherwise, find the non-dominated arms
        in the non-Pareto optimal set and return one of them.
        :return: The arm to pull.
        """
        # 1. Sample from the posterior
        variances = invgamma.rvs(a=self.alpha, scale=self.beta)
        samples = norm.rvs(loc=self.mu, scale=np.sqrt(variances / self.lambdas))

        # 2. Find Pareto front
        pareto_mask = paretoset(samples, sense=["max"] * self.num_objectives)
        pareto_indices = np.where(pareto_mask)[0]

        # 3. Decide which set to pull from
        if np.random.random() < self.p:
            return random.choice(pareto_indices)
        else:
            non_pareto_indices = np.where(~pareto_mask)[0]
            if len(non_pareto_indices) == 0:
                return random.choice(pareto_indices)

            non_pareto_samples = samples[non_pareto_indices]
            non_pareto_pareto_mask = paretoset(non_pareto_samples, sense=["max"] * self.num_objectives)
            non_dominated_indices = np.where(non_pareto_pareto_mask)[0]

            # Map non-dominated indices back to original arm indices
            non_dominated_indices = non_pareto_indices[non_dominated_indices]

            return random.choice(non_dominated_indices)

    def get_top_arms(self):
        """
        Get the arms that are considered to be Pareto optimal by the bandit based on the estimated means
        :return: The top arms.
        """
        pareto_mask = paretoset(self.mu, sense=["max"] * self.num_objectives)
        pareto_indices = np.where(pareto_mask)[0]
        return pareto_indices

    def learn(self, arm, reward):
        """
        Learn from the reward that was received for pulling the arm.
        :param arm: The arm that was pulled.
        :param reward: The reward for each objective. (This is a 1D NumPy array)
        :return: None
        """
        # Get the current parameters for the pulled arm
        mu_0 = self.mu[arm]
        lambda_0 = self.lambdas[arm]

        # Perform vectorized updates
        # Note: The order matters. We must calculate new beta and mu
        # using the *old* mu_0 and lambda_0.

        self.beta[arm] += 0.5 * (reward - mu_0) ** 2 * (lambda_0 / (lambda_0 + 1))
        self.mu[arm] = (mu_0 * lambda_0 + reward) / (lambda_0 + 1)
        self.lambdas[arm] += 1
        self.alpha[arm] += 0.5

    def reset(self):
        """
        Reset the agent.
        :return: None
        """
        self.mu = np.zeros((self.num_arms, self.num_objectives))
        self.lambdas = np.ones((self.num_arms, self.num_objectives))
        self.alpha = np.full((self.num_arms, self.num_objectives), 2.0, dtype=np.float64)  # shape
        self.beta = np.full((self.num_arms, self.num_objectives), 2.0, dtype=np.float64)  # scale
