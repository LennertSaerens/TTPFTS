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
    def get_stds(self) -> np.ndarray:
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

    def get_stds(self):
        # Variance of Beta distribution: alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
        total = self.alphas + self.betas
        variances = (self.alphas * self.betas) / (total ** 2 * (total + 1))
        return np.sqrt(variances)

    def log(self, file):
        df = pd.DataFrame({
            "arm": np.arange(self.num_arms),
            "means": self.get_mean().tolist(),
            "stds": self.get_stds().tolist(),
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

    def get_stds(self):
        safe_alpha = np.maximum(self.alpha, 2.01)
        variances = self.beta ** 2 / ((safe_alpha - 1) ** 2 * (safe_alpha - 2))
        return np.sqrt(variances)

    def log(self, file):
        df = pd.DataFrame({
            "arm": np.arange(self.num_arms),
            "means": self.get_mean().tolist(),
            "stds": self.get_stds().tolist(),
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

    def get_stds(self):
        pulled = self._counts > 0
        safe_beta = np.where(pulled, self.beta, 1.0)
        variances = np.where(pulled, self.alpha / (safe_beta ** 2), 1e12)
        return np.sqrt(variances)

    def log(self, file):
        df = pd.DataFrame({
            "arm": np.arange(self.num_arms),
            "means": self.get_mean().tolist(),
            "stds": self.get_stds().tolist(),
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

    def get_stds(self):
        # Posterior std: sqrt(beta / (lambdas * (alpha - 1))) when alpha > 1
        return np.where(
            self.alpha > 1,
            np.sqrt(self.beta / (self.lambdas * (self.alpha - 1))),
            1e6
        )

    def log(self, file):
        df = pd.DataFrame({
            "arm": np.arange(self.num_arms),
            "means": self.get_mean().tolist(),
            "stds": self.get_stds().tolist(),
        })
        df.to_parquet(file, index=False)


class TPosterior(PosteriorBase):
    """
    Posterior for Gaussian rewards with *unknown mean and unknown variance*,
    generalizing over the improper power-law prior family

        p(mu, sigma^2) ~ (sigma^2)^{-(alpha + 1)},      -inf < mu < inf,

    studied by Honda & Takemura, "Optimality of Thompson Sampling for
    Gaussian Bandits Depends on Priors" (AISTATS 2014). ``alpha=0`` (the
    default) is exactly the standard *reference* (Jeffreys) prior

        p(mu, sigma^2) ~ 1 / sigma^2,

    for which the textbook result after n >= 2 i.i.d. observations x_1..x_n
    (sample mean xbar, sum of squared deviations S = sum_i (x_i - xbar)^2) is

        mu | data ~ Student-t(df = n - 1, loc = xbar, scale = sqrt(S / (n (n-1)))).

    More generally, df = n + 2*alpha - 1 (df=n-1 at alpha=0). The posterior
    is only well-defined once df > 0; states with df <= 0 (too few
    observations for the chosen alpha) are surfaced *explicitly* via
    ``ValueError`` rather than silently patched with an ad-hoc sentinel.

    Sufficient statistics (count n, running mean, sum of squared deviations)
    are tracked with Welford's numerically stable one-pass streaming
    algorithm, which is exact (up to floating-point rounding) for any
    number/order of updates and matches a batch computation from raw data.

    Note: the Student-t *scale* parameter (used for sampling) is **not**
    the actual standard deviation of the posterior distribution of mu -
    the latter is scale * sqrt(df / (df - 2)) and is only finite for
    df > 2. ``get_scale()`` exposes the former, ``get_stds()`` the latter
    (returning ``np.inf`` where the true posterior variance is infinite,
    which is the mathematically correct value, not an error).
    """

    def __init__(self, num_arms, num_objectives, alpha: float = 0.0):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.alpha = alpha
        self.n = np.zeros((num_arms, num_objectives), dtype=np.int64)
        self.mean = np.zeros((num_arms, num_objectives), dtype=np.float64)
        # Welford's M2: running sum of squared deviations from the current mean.
        self.M2 = np.zeros((num_arms, num_objectives), dtype=np.float64)

    def _df(self) -> np.ndarray:
        """Student-t degrees of freedom for every (arm, objective)."""
        return self.n + 2.0 * self.alpha - 1.0

    def _valid_mask(self) -> np.ndarray:
        """True where the posterior is well-defined (df > 0)."""
        return self._df() > 0

    def _require_valid(self, context: str) -> np.ndarray:
        mask = self._valid_mask()
        if not np.all(mask):
            bad = [tuple(idx) for idx in np.argwhere(~mask)]
            raise ValueError(
                f"TPosterior.{context}: no well-defined posterior for "
                f"(arm, objective) pairs {bad} - insufficient observations "
                f"(n={self.n[~mask].tolist()}) for alpha={self.alpha} "
                f"(need df = n + 2*alpha - 1 > 0). Ensure at least "
                f"ceil(1 - 2*alpha) observations per arm before sampling."
            )
        return mask

    def _scale(self, df: np.ndarray) -> np.ndarray:
        # df > 0 guaranteed by caller via _require_valid; n > 0 follows from df > 0
        # whenever alpha >= 0, and is enforced explicitly otherwise via the n>0 check.
        n_safe = np.where(self.n > 0, self.n, 1)
        return np.sqrt(self.M2 / (n_safe * df))

    def sample(self):
        self._require_valid("sample")
        df = self._df()
        scale = self._scale(df)
        z = t.rvs(df=df)
        return self.mean + scale * z

    def update(self, arm, reward):
        reward = np.asarray(reward, dtype=np.float64)
        self.n[arm] += 1
        # Welford's online algorithm for mean and sum-of-squared-deviations.
        delta = reward - self.mean[arm]
        self.mean[arm] = self.mean[arm] + delta / self.n[arm]
        delta2 = reward - self.mean[arm]
        self.M2[arm] = self.M2[arm] + delta * delta2

    def get_mean(self):
        return self.mean

    def reset(self, _):
        self.n = np.zeros((self.num_arms, self.num_objectives), dtype=np.int64)
        self.mean = np.zeros((self.num_arms, self.num_objectives), dtype=np.float64)
        self.M2 = np.zeros((self.num_arms, self.num_objectives), dtype=np.float64)

    def get_scale(self):
        """Student-t *scale* parameter used for sampling (not the actual std)."""
        self._require_valid("get_scale")
        return self._scale(self._df())

    def get_stds(self):
        """
        Actual marginal posterior standard deviation of mu:
        Var(Student-t_df) = scale^2 * df / (df - 2) for df > 2, and is
        infinite (returned explicitly as ``np.inf``) for 0 < df <= 2.
        """
        self._require_valid("get_stds")
        df = self._df()
        scale = self._scale(df)
        with np.errstate(divide="ignore", invalid="ignore"):
            variance_factor = np.where(df > 2, df / np.where(df > 2, df - 2, 1.0), np.inf)
        return scale * np.sqrt(variance_factor)

    def log(self, file):
        df = pd.DataFrame({
            "arm": np.arange(self.num_arms),
            "means": self.get_mean().tolist(),
            "stds": self.get_stds().tolist(),
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
