import numpy as np
from scipy.stats import invgamma, norm, t
from abc import ABC, abstractmethod
import pandas as pd


class PosteriorBase(ABC):
    @abstractmethod
    def sample(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def update(self, arm: int, reward: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_mean(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def reset(self, env_stds) -> None:
        raise NotImplementedError

    @abstractmethod
    def log(self, file: str) -> None:
        raise NotImplementedError


class BetaBernoulliPosterior(PosteriorBase):
    """Beta-Bernoulli posterior for binary rewards."""

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

    def log(self, file):
        means = self.get_mean()
        # Variance of Beta distribution: alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
        total = self.alphas + self.betas
        variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
        stds = np.sqrt(variances)
        df = pd.DataFrame({
            "arm": np.arange(self.num_arms),
            "means": means.tolist(),
            "stds": stds.tolist(),
        })
        df.to_parquet(file, index=False)


class GammaExponentialPosterior(PosteriorBase):
    """Gamma posterior for exponential rewards (conjugate pair).

    Models each arm/objective rate as lambda ~ Gamma(alpha, beta).
    The reward mean is 1/lambda. Thompson samples are drawn as
    1 / Gamma(alpha, beta) so that higher means are preferred.

    Default prior is the Jeffreys prior: Gamma(alpha0=0, beta0=0),
    i.e. pi(lambda) proportional to 1/lambda.  This is improper, so at least
    one warmup pull per arm is required before sampling (use
    num_warmup_pulls >= 1 in TTPFTS).
    """

    def __init__(self, num_arms, num_objectives, alpha0=0.0, beta0=0.0):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self._alpha0 = alpha0
        self._beta0 = beta0
        self.alpha = np.full((num_arms, num_objectives), alpha0)
        self.beta = np.full((num_arms, num_objectives), beta0)
        self._counts = np.zeros((num_arms, num_objectives))

    def sample(self):
        pulled = self._counts > 0
        safe_alpha = np.where(pulled, self.alpha, 1.0)
        safe_beta = np.where(pulled, self.beta, 1.0)
        lam = np.random.gamma(safe_alpha, 1.0 / safe_beta)
        lam = np.maximum(lam, 1e-12)
        return np.where(pulled, 1.0 / lam, 1e6)

    def update(self, arm, reward):
        self.alpha[arm] += 1
        self.beta[arm] += reward
        self._counts[arm] += 1

    def get_mean(self):
        pulled = self._counts > 0
        # Posterior mean of 1/lambda = beta / (alpha - 1) for alpha > 1
        safe_alpha = np.where(self.alpha > 1, self.alpha, 2.0)
        proper_mean = np.where(self.alpha > 1, self.beta / (safe_alpha - 1), self.beta)
        return np.where(pulled, proper_mean, 1e6)

    def reset(self, _):
        self.alpha = np.full((self.num_arms, self.num_objectives), self._alpha0)
        self.beta = np.full((self.num_arms, self.num_objectives), self._beta0)
        self._counts = np.zeros((self.num_arms, self.num_objectives))

    def log(self, file):
        means = self.get_mean()
        safe_alpha = np.maximum(self.alpha, 2.01)
        variances = self.beta ** 2 / ((safe_alpha - 1) ** 2 * (safe_alpha - 2))
        stds = np.sqrt(variances)
        df = pd.DataFrame({
            "arm": np.arange(self.num_arms),
            "means": means.tolist(),
            "stds": stds.tolist(),
        })
        df.to_parquet(file, index=False)


class GammaPoissonPosterior(PosteriorBase):
    """Gamma posterior for Poisson rewards (conjugate pair).

    Models each arm/objective rate as lambda ~ Gamma(alpha, beta).
    The reward mean IS lambda.  Thompson samples are drawn directly
    from Gamma(alpha, 1/beta).

    Default prior is the Jeffreys prior: Gamma(alpha0=0.5, beta0=0),
    i.e. pi(lambda) proportional to lambda^(-1/2).  This is improper, so
    at least one warmup pull per arm is required before sampling (use
    num_warmup_pulls >= 1 in TTPFTS).

    Update rules:  alpha += observed_count,  beta += 1.
    """

    def __init__(self, num_arms, num_objectives, alpha0=0.5, beta0=0.0):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self._alpha0 = alpha0
        self._beta0 = beta0
        self.alpha = np.full((num_arms, num_objectives), alpha0)
        self.beta = np.full((num_arms, num_objectives), beta0)
        self._counts = np.zeros((num_arms, num_objectives))

    def sample(self):
        pulled = self._counts > 0
        safe_beta = np.where(pulled, self.beta, 1.0)
        return np.where(pulled, np.random.gamma(self.alpha, 1.0 / safe_beta), 1e6)

    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += 1
        self._counts[arm] += 1

    def get_mean(self):
        pulled = self._counts > 0
        safe_beta = np.where(pulled, self.beta, 1.0)
        return np.where(pulled, self.alpha / safe_beta, 1e6)

    def reset(self, _):
        self.alpha = np.full((self.num_arms, self.num_objectives), self._alpha0)
        self.beta = np.full((self.num_arms, self.num_objectives), self._beta0)
        self._counts = np.zeros((self.num_arms, self.num_objectives))

    def log(self, file):
        means = self.get_mean()
        pulled = self._counts > 0
        safe_beta = np.where(pulled, self.beta, 1.0)
        variances = np.where(pulled, self.alpha / (safe_beta ** 2), 1e12)
        stds = np.sqrt(variances)
        df = pd.DataFrame({
            "arm": np.arange(self.num_arms),
            "means": means.tolist(),
            "stds": stds.tolist(),
        })
        df.to_parquet(file, index=False)


class NormalIGPosterior(PosteriorBase):
    """Normal-Inverse-Gamma posterior for Gaussian rewards with unknown variance."""

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

    def log(self, file):
        # Posterior std: sqrt(beta / (lambdas * (alpha - 1))) when alpha > 1
        stds = np.where(
            self.alpha > 1,
            np.sqrt(self.beta / (self.lambdas * (self.alpha - 1))),
            1e6
        )
        df = pd.DataFrame({
            "arm": np.arange(self.num_arms),
            "means": self.mu.tolist(),
            "stds": stds.tolist(),
        })
        df.to_parquet(file, index=False)


class TPosterior(PosteriorBase):
    """
    Posterior based on "Optimality of Thompson Sampling for Gaussian Bandits Depends on Priors"
    paper by Honda and Takemura.
    """

    def __init__(self, num_arms, num_objectives, alpha=0):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.alpha = alpha
        self.mu = np.zeros((num_arms, num_objectives))
        self.n = np.zeros((num_arms, num_objectives))
        self.sum = np.zeros((num_arms, num_objectives))
        self.S = np.zeros((num_arms, num_objectives))

    def sample(self):
        # Guard unpulled arms: return large-variance samples
        pulled = self.n > 0
        nu_df = np.where(pulled, self.n + (2 * self.alpha) - 1, 1.0)
        nu_df = np.maximum(nu_df, 0.01)
        z = t.rvs(df=nu_df)
        # Use safe denominators to avoid division-by-zero warnings
        safe_n = np.where(pulled, self.n, 1.0)
        scale = np.where(
            pulled,
            self.S / np.sqrt(safe_n * nu_df),
            1e6
        )
        return self.mu + scale * z

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

    def log(self, file):
        pulled = self.n > 0
        nu_df = np.where(pulled, self.n + (2 * self.alpha) - 1, 1.0)
        nu_df = np.maximum(nu_df, 0.01)
        safe_n = np.where(pulled, self.n, 1.0)
        stds = np.where(pulled, np.sqrt(self.S / (safe_n * nu_df)), 1e6)

        df = pd.DataFrame({
            "arm": np.arange(self.num_arms),
            "means": self.mu.tolist(),
            "stds": stds.tolist(),
        })
        df.to_parquet(file, index=False)


class NormalPosterior(PosteriorBase):
    """Bayesian posterior for normal rewards with known variance, per arm/objective."""

    def __init__(self, num_arms, num_objectives, known_stds):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.known_stds = np.full((self.num_arms, self.num_objectives), known_stds, dtype=np.float64)
        self.counts = np.zeros((num_arms, num_objectives), dtype=int)
        self.means = np.zeros((num_arms, num_objectives), dtype=float)

    def sample(self):
        stds = np.where(
            self.counts > 0,
            self.known_stds / np.sqrt(self.counts),
            1e6
        )
        return np.random.normal(self.means, stds)

    def update(self, arm, reward):
        n = self.counts[arm]
        old_mean = self.means[arm]
        new_mean = (old_mean * n + reward) / (n + 1)
        self.means[arm] = new_mean
        self.counts[arm] += 1

    def get_mean(self):
        return self.means

    def get_stds(self):
        return np.where(
            self.counts > 0,
            self.known_stds / np.sqrt(self.counts),
            1e6
        )

    def reset(self, env_stds):
        self.known_stds = np.full((self.num_arms, self.num_objectives), env_stds, dtype=np.float64)
        self.counts = np.zeros((self.num_arms, self.num_objectives), dtype=int)
        self.means = np.zeros((self.num_arms, self.num_objectives), dtype=float)

    def log(self, file):
        stds = self.get_stds()
        df = pd.DataFrame({
            "arm": np.arange(self.num_arms),
            "means": self.means.tolist(),
            "stds": stds.tolist(),
        })
        df.to_parquet(file, index=False)
