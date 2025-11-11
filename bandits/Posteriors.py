import numpy as np
from scipy.stats import invgamma, norm, t
from abc import ABC, abstractmethod


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
    def reset(self):
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

    def reset(self):
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

    def reset(self):
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

    def reset(self):
        self.mu = np.zeros((self.num_arms, self.num_objectives))
        self.n = np.zeros((self.num_arms, self.num_objectives))
        self.sum = np.zeros((self.num_arms, self.num_objectives))
        self.S = np.zeros((self.num_arms, self.num_objectives))


class NormalPosterior(PosteriorBase):
    """
    Posterior for arms with normal rewards with known variance as described in "Thompson Sampling - An Efficient Method
     for Searching Ultralarge Synthesis on Demand Databases" by Klarich et al.
    """
    def __init__(self, num_arms, num_objectives, known_variance=0.25):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.known_variance = known_variance
        self.means = np.zeros((self.num_arms, self.num_objectives))
        self.known_variances = np.full((self.num_arms, self.num_objectives), self.known_variance, dtype=np.float64)
        self.empirical_variances = np.full((self.num_arms, self.num_objectives), self.known_variance, dtype=np.float64)

    def sample(self):
        return np.random.normal(self.means, np.sqrt(self.empirical_variances))

    def update(self, arm, reward):
        self.means[arm] = (self.empirical_variances[arm] * reward + self.known_variances[arm] * self.means[arm]) / (self.empirical_variances[arm] + self.known_variances[arm])
        self.empirical_variances[arm] = (self.empirical_variances[arm] * self.known_variances[arm]) / (self.empirical_variances[arm] + self.known_variances[arm])

    def get_mean(self):
        return self.means

    def reset(self):
        self.means = np.zeros((self.num_arms, self.num_objectives))
        self.known_variances = np.full((self.num_arms, self.num_objectives), self.known_variance, dtype=np.float64)
        self.empirical_variances = np.full((self.num_arms, self.num_objectives), self.known_variance, dtype=np.float64)
