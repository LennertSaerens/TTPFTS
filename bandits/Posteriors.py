import numpy as np
from scipy.stats import invgamma, norm, t
from abc import ABC, abstractmethod
import pandas as pd


class PosteriorBase(ABC):
    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def update(self, arm, reward):
        raise NotImplementedError

    @abstractmethod
    def get_mean(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self, env_stds):
        raise NotImplementedError

    @abstractmethod
    def log(self, file):
        raise NotImplementedError


class BetaBernoulliPosterior(PosteriorBase):
    def __init__(self, num_arms, num_objectives):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.alphas = np.ones((num_arms, num_objectives))
        self.betas = np.ones((num_arms, num_objectives))

    def sample(self):
        return np.random.beta(self.alphas, self.betas)

    def update(self, arm, reward):
        self.alphas[arm] += reward
        self.betas[arm] += 1 - reward

    def get_mean(self):
        return self.alphas / (self.alphas + self.betas)

    def reset(self, _):
        self.alphas = np.ones((self.num_arms, self.num_objectives))
        self.betas = np.ones((self.num_arms, self.num_objectives))


class NormalIGPosterior(PosteriorBase):
    def __init__(self, num_arms, num_objectives):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.mu = np.zeros((num_arms, num_objectives))
        self.lambdas = np.ones((num_arms, num_objectives))
        self.alpha = np.full((num_arms, num_objectives), 2.0)
        self.beta = np.full((num_arms, num_objectives), 2.0)

    def sample(self):
        variances = invgamma.rvs(a=self.alpha, scale=self.beta)
        return norm.rvs(loc=self.mu, scale=np.sqrt(variances / self.lambdas))

    def update(self, arm, reward):
        mu_0 = self.mu[arm]
        lambda_0 = self.lambdas[arm]
        self.beta[arm] += 0.5 * (reward - mu_0) ** 2 * (lambda_0 / (lambda_0 + 1))
        self.mu[arm] = (mu_0 * lambda_0 + reward) / (lambda_0 + 1)
        self.lambdas[arm] += 1
        self.alpha[arm] += 0.5

    def get_mean(self):
        return self.mu

    def reset(self, _):
        self.mu = np.zeros((self.num_arms, self.num_objectives))
        self.lambdas = np.ones((self.num_arms, self.num_objectives))
        self.alpha = np.full((self.num_arms, self.num_objectives), 2.0)
        self.beta = np.full((self.num_arms, self.num_objectives), 2.0)


class TPosterior(PosteriorBase):
    """
    Posterior based on "Optimality of Thompson Sampling for Gaussian Bandits Depends on Priors" paper by Honda and Takemura.
    """

    def __init__(self, num_arms, num_objectives, alpha=0):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.alpha = alpha
        self.mu = np.zeros((num_arms, num_objectives))
        self.n = np.zeros((num_arms, num_objectives))  # Number of observations per arm-objective
        self.sum = np.zeros((num_arms, num_objectives))  # Sum of rewards per arm-objective
        self.S = np.zeros((num_arms, num_objectives))  # Sum of squared deviations per arm-objective

    def sample(self):
        nu_df = self.n + (2 * self.alpha) - 1
        z = t.rvs(df=nu_df)
        return self.mu + (self.S / np.sqrt(self.n * nu_df)) * z

    def update(self, arm, reward):
        self.n[arm] += 1
        self.sum[arm] += reward
        self.mu[arm] = self.sum[arm] / self.n[arm]
        self.S[arm] += (reward - self.mu[arm]) ** 2

    def get_mean(self):
        return self.mu

    def reset(self, _):
        self.mu = np.zeros((self.num_arms, self.num_objectives))
        self.n = np.zeros((self.num_arms, self.num_objectives))
        self.sum = np.zeros((self.num_arms, self.num_objectives))
        self.S = np.zeros((self.num_arms, self.num_objectives))


# class NormalPosterior(PosteriorBase):
#     """
#     Posterior for arms with normal rewards with known variance as described in "Thompson Sampling - An Efficient Method
#      for Searching Ultralarge Synthesis on Demand Databases" by Klarich et al.
#     """
#     def __init__(self, num_arms, num_objectives, known_stds):
#         self.num_arms = num_arms
#         self.num_objectives = num_objectives
#         self.means = np.zeros((self.num_arms, self.num_objectives))
#         self.known_stds = np.array(known_stds)
#         known_variances = self.known_stds ** 2
#         self.known_variances = np.full((self.num_arms, self.num_objectives), known_variances, dtype=np.float64)
#         self.empirical_variances = np.full((self.num_arms, self.num_objectives), known_variances, dtype=np.float64)
#
#     def sample(self):
#         return np.random.normal(self.means, np.sqrt(self.empirical_variances))
#
#     def update(self, arm, reward):
#         self.means[arm] = (self.empirical_variances[arm] * reward + self.known_variances[arm] * self.means[arm]) / (self.empirical_variances[arm] + self.known_variances[arm])
#         self.empirical_variances[arm] = (self.empirical_variances[arm] * self.known_variances[arm]) / (self.empirical_variances[arm] + self.known_variances[arm])
#
#     def get_mean(self):
#         return self.means
#
#     def reset(self):
#         known_variances = self.known_stds ** 2
#         self.means = np.zeros((self.num_arms, self.num_objectives))
#         self.known_variances = np.full((self.num_arms, self.num_objectives), known_variances, dtype=np.float64)
#         self.empirical_variances = np.full((self.num_arms, self.num_objectives), known_variances, dtype=np.float64)


class NormalPosterior(PosteriorBase):
    """
    Bayesian posterior for normal rewards with known variance, per arm/objective.
    """
    def __init__(self, num_arms, num_objectives, known_stds):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.known_stds = np.full((self.num_arms, self.num_objectives), known_stds, dtype=np.float64)

        # Each arm/objective: track count (n), mean, and posterior variance
        self.counts = np.zeros((num_arms, num_objectives), dtype=int)
        self.means = np.zeros((num_arms, num_objectives), dtype=float)

    def sample(self):
        # Posterior std-dev: known std / sqrt(n) (n >= 1), otherwise infinity
        stds = np.where(
            self.counts > 0,
            self.known_stds / np.sqrt(self.counts),
            1e6  # Large std-dev for unpulled arms
        )

        return np.random.normal(self.means, stds)

    def update(self, arm, reward):
        n = self.counts[arm]
        old_mean = self.means[arm]
        # Running mean
        new_mean = (old_mean * n + reward) / (n + 1)
        self.means[arm] = new_mean
        self.counts[arm] += 1

    def get_mean(self):
        return self.means

    def get_stds(self):
        stds = np.where(
            self.counts > 0,
            self.known_stds / np.sqrt(self.counts),
            1e6
        )
        return stds

    def reset(self, env_stds):
        self.known_stds = np.full((self.num_arms, self.num_objectives), env_stds, dtype=np.float64)
        self.counts = np.zeros((self.num_arms, self.num_objectives), dtype=int)
        self.means = np.zeros((self.num_arms, self.num_objectives), dtype=float)

    def log(self, file):
        stds = np.where(
            self.counts > 0,
            self.known_stds / np.sqrt(self.counts),
            1e6
        )

        # Aggregate per arm by averaging over objectives
        arm_indices = np.arange(self.num_arms)
        means_lists = [[self.means[i, j] for j in range(self.num_objectives)] for i in range(self.num_arms)]
        stds_lists = [[stds[i, j] for j in range(self.num_objectives)] for i in range(self.num_arms)]

        df = pd.DataFrame({
            "arm": arm_indices,
            "means": means_lists,
            "stds": stds_lists,
        })

        df.to_parquet(file, index=False)
