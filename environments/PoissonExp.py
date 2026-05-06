"""
Bi-objective Poisson environment for organic agriculture benchmarking.

This environment is based on the "Organic Agriculture" benchmark from:
    Cappé, Olivier, et al. "Kullback–leibler upper confidence bounds for
    optimal sequential allocation." The Annals of Statistics 41.3 (2013).
    https://github.com/plibin/bfts/blob/master/environments/poisson_olivier.py

The original single-objective environment uses linearly increasing Poisson
rates from 0.5 to 5.0 across arms, shuffled for reproducibility. This module
extends it to a bi-objective Pareto set identification problem by introducing
a conflicting second objective:

    lambda_2[i] = max(lambda_1) - lambda_1[i] + noise[i]

where noise ~ N(0, 0.6) with seed 42 for reproducibility.
The larger noise creates a more challenging Pareto set where most arms are
dominated but a meaningful subset is Pareto-optimal with varying conflicts.
"""

import numpy as np

from environments.BaseEnvironment import BaseEnvironment
from environments.distributions import PoissonReward

_NOISE_SEED = 42
_NOISE_STD = 0.6  # increased from 0.3 for more challenging Pareto set
_MIN_LAMBDA = 0.01


def _make_linear_lambdas(n_arms: int) -> np.ndarray:
    """Generate linearly increasing lambda values from 0.5 to 5.0, shuffled."""
    min_lambda = 0.5
    max_lambda = 5.0
    lam1 = np.array([min_lambda + i * (max_lambda - min_lambda) / (n_arms - 1) 
                     for i in range(n_arms)], dtype=np.float64)
    # Shuffle with seed 42 for reproducibility (matching BFTS)
    rng = np.random.RandomState(42)
    rng.shuffle(lam1)
    return lam1


def _make_bi_objective_lambdas(n_arms: int) -> np.ndarray:
    """Build (n_arms, 2) array of conflicting Poisson lambda values."""
    lam1 = _make_linear_lambdas(n_arms)
    rng = np.random.RandomState(_NOISE_SEED)
    noise = rng.normal(0, _NOISE_STD, size=n_arms)
    lam2 = lam1.max() - lam1 + noise
    lam2 = np.clip(lam2, _MIN_LAMBDA, None)
    return np.column_stack([lam1, lam2])


class PoissonExp(BaseEnvironment):
    """Bi-objective Poisson benchmark (organic agriculture inspired).

    Single-objective design: linearly increasing Poisson rates [0.5, 5.0].
    Bi-objective extension: second objective creates Pareto trade-off.

    Each arm yields independent Poisson rewards on two objectives with
    conflicting lambda values. Most arms are dominated, but a meaningful
    subset is Pareto-optimal.

    Parameters
    ----------
    n_arms : int
        Number of arms (default 100). Must be >= 2.
    """

    def __init__(self, n_arms: int = 100):
        if n_arms < 2:
            raise ValueError(f"n_arms must be >= 2, got {n_arms}")
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
