"""
Bi-objective Poisson environment derived from the BFTS benchmark.

The original single-objective Poisson environment is from:
    Libin et al., "Bayesian Anytime m-Top Exploration", ICTAI 2019.
    https://github.com/plibin/bfts

It contains 1000 Poisson arms with lambda values drawn from
epidemiological simulation outputs.  This module adapts it to
a bi-objective Pareto set identification problem by introducing
a conflicting second objective:

    lambda_2[i] = max(lambda_1) - lambda_1[i] + noise[i]

where noise ~ N(0, 0.3) with seed 42 for reproducibility.
The noise creates a non-trivial Pareto set where most arms are
dominated but a meaningful subset is Pareto-optimal.

The number of arms can be configured (default 100, max 1000).
"""

import os
from typing import Optional

import numpy as np

from environments.BaseEnvironment import BaseEnvironment
from environments.distributions import PoissonReward

_DATA_FILE = os.path.join(os.path.dirname(__file__), "poisson_lambdas.npy")
_ALL_LAMBDAS = np.load(_DATA_FILE)

_NOISE_SEED = 42
_NOISE_STD = 0.3
_MIN_LAMBDA = 0.01  # floor to keep Poisson rate positive


def _make_bi_objective_lambdas(n_arms: int) -> np.ndarray:
    """Build (n_arms, 2) array of Poisson lambda values."""
    if n_arms > len(_ALL_LAMBDAS):
        raise ValueError(f"n_arms={n_arms} exceeds available lambdas ({len(_ALL_LAMBDAS)})")
    lam1 = _ALL_LAMBDAS[:n_arms].copy()
    rng = np.random.RandomState(_NOISE_SEED)
    noise = rng.normal(0, _NOISE_STD, size=n_arms)
    lam2 = _ALL_LAMBDAS.max() - lam1 + noise
    lam2 = np.clip(lam2, _MIN_LAMBDA, None)
    return np.column_stack([lam1, lam2])


class PoissonExp(BaseEnvironment):
    """Bi-objective Poisson benchmark (BFTS-derived).

    Each arm yields independent Poisson rewards on two conflicting
    objectives.  The first objective uses lambda values from the BFTS
    epidemiological benchmark; the second uses a conflicting linear
    transformation with Gaussian noise to create a non-trivial Pareto set.

    Parameters
    ----------
    n_arms : int
        Number of arms to use from the 1000 available (default 100).
    """

    def __init__(self, n_arms: int = 100):
        self.n_arms_requested = n_arms
        lambdas = _make_bi_objective_lambdas(n_arms)
        distributions = [PoissonReward(lam) for lam in lambdas]
        pareto_indices = BaseEnvironment._compute_pareto_indices(lambdas)
        reference_point = np.array([lambdas.max(axis=0)[0] + 1, lambdas.max(axis=0)[1] + 1])
        inverted = reference_point - lambdas

        super().__init__(
            num_arms=n_arms,
            num_objectives=2,
            pareto_indices=pareto_indices,
            inverted_arms=inverted,
            reference_point=reference_point,
            distributions=distributions,
        )
