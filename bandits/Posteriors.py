import numpy as np
from scipy.stats import invgamma, norm
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
