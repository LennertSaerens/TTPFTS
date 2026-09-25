"""Fixed-budget Adaptive Pareto Exploration (APE).

The sampling rule is a faithful port of ``psi_ape::loop`` in the bundled
reference implementation. The reference confidence-stops; this class records
that stopping decision but continues applying the same selector for a fixed
budget whenever its candidate set remains non-empty. If every arm is already
certified, the reference selector is undefined and balanced sampling is the
explicit fixed-budget extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from bandits.InterfaceMOMABPFI import BaseMOMABAlgorithm

_REFERENCE_STOPPING_TARGET = 1 << 16
_REFERENCE_INFINITY = 1e12


def confidence_constant(delta: float) -> float:
    """Port of ``Cg`` in ``bandit_lib_c++/src/cpp/utils.hpp``."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be strictly between zero and one")
    log_inverse_delta = np.log(1.0 / delta)
    return log_inverse_delta / 2.0 + max(np.log(log_inverse_delta / 2.0), 0.0)


def beta_ij(T_i: np.ndarray | int, T_j: np.ndarray | int, cg: float, sigma: float) -> np.ndarray:
    """Port of ``betaij`` in ``bandit_lib_c++/src/cpp/utils.hpp``."""
    T_i = np.asarray(T_i, dtype=float)
    T_j = np.asarray(T_j, dtype=float)
    return sigma * np.sqrt(
        2.0 * (2.0 * cg + 2.0 * np.log(4.0 + np.log(T_i)) + 2.0 * np.log(4.0 + np.log(T_j)))
        * (1.0 / T_i + 1.0 / T_j)
    )


def _minimum_quantity_dom(x_i: np.ndarray, x_j: np.ndarray, epsilon: float) -> float:
    """Port of ``minimum_quantity_dom`` in ``utils.hpp``."""
    return float(np.max(x_i + epsilon - x_j))


def _minimum_quantity_non_dom(x_i: np.ndarray, x_j: np.ndarray, epsilon: float) -> float:
    """Port of ``minimum_quantity_non_dom`` in ``utils.hpp``."""
    return float(np.min(x_j - x_i + epsilon))


def _pareto_optimal_arms_mask(means: np.ndarray) -> np.ndarray:
    """Port of ``pareto_optimal_arms_mask(..., null, 0.)`` in ``utils.cxx``.

    Vectorized over the arm pairs; elementwise comparisons are exact, so this
    returns the same mask as the reference's nested loop.
    """
    means = np.asarray(means, dtype=float)
    # pair[i, j] is True iff j dominates i (>= everywhere, > somewhere).
    no_better = np.all(means[:, None, :] <= means[None, :, :], axis=2)
    strictly_worse = np.any(means[:, None, :] < means[None, :, :], axis=2)
    dominated_by = no_better & strictly_worse
    np.fill_diagonal(dominated_by, False)
    return ~np.any(dominated_by, axis=1)


def _pairwise_dom(means: np.ndarray, epsilon: float) -> np.ndarray:
    """``_minimum_quantity_dom(means[i], means[j], epsilon)`` for every (i, j)."""
    return np.max(means[:, None, :] + epsilon - means[None, :, :], axis=2)


def _pairwise_non_dom(means: np.ndarray, epsilon: float) -> np.ndarray:
    """``_minimum_quantity_non_dom(means[i], means[j], epsilon)`` for every (i, j)."""
    return np.min(means[None, :, :] - means[:, None, :] + epsilon, axis=2)


@dataclass(frozen=True)
class APEDecision:
    """State used to select one arm, exposed for deterministic trace tests."""

    selected_arm: int
    counts: np.ndarray
    means: np.ndarray
    beta: Optional[np.ndarray]
    pareto_arms: np.ndarray
    opt_arms: np.ndarray
    opt_comp_arms: np.ndarray
    b_t: Optional[int]
    c_t: Optional[int]
    z1_t: Optional[float]
    z2_t: Optional[float]
    z3_t: Optional[int]
    z4_t: Optional[float]
    reference_should_stop: bool
    fixed_budget_continuation: bool


class APEAnytimeBandit(BaseMOMABAlgorithm):
    """APE sampling with reference stopping diagnostics and fixed-budget pulls.

    ``sigma`` is the sub-Gaussian bound and is a required argument: the C++
    reference hardcodes ``1.0`` for its Gaussian bandit and ``0.5`` for its
    Bernoulli bandit, so no default is safe.  Equivalence tests reproduce the
    reference by passing ``sigma=1.0``; the EgeExp experiments pass the
    validated reward standard deviation ``0.5``.  It is never inferred from
    ``env_stds``.  ``epsilon_1``, ``epsilon_2``, and ``target_optimal_arms``
    are the ``psi_ape::loop`` parameters ``eps_1``, ``eps_2``, and ``m``.
    """

    def __init__(
        self,
        num_arms: int,
        num_objectives: int,
        *,
        sigma: float,
        delta: float = 0.1,
        epsilon_1: float = 0.0,
        epsilon_2: float = 0.0,
        target_optimal_arms: int = _REFERENCE_STOPPING_TARGET,
    ) -> None:
        super().__init__(num_arms, num_objectives)
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if target_optimal_arms < 1:
            raise ValueError("target_optimal_arms must be at least one")
        self.delta = delta
        self.sigma = float(sigma)
        self.epsilon_1 = float(epsilon_1)
        self.epsilon_2 = float(epsilon_2)
        self.target_optimal_arms = int(target_optimal_arms)
        self._cg = confidence_constant(delta)
        self.pulls = np.zeros(num_arms, dtype=int)
        self.means = np.zeros((num_arms, num_objectives), dtype=float)
        self.last_decision: Optional[APEDecision] = None

    def reset(self, env_stds: np.ndarray) -> None:
        del env_stds
        self.pulls.fill(0)
        self.means.fill(0.0)
        self.last_decision = None

    def _beta(self) -> np.ndarray:
        return beta_ij(self.pulls[:, None], self.pulls[None, :], self._cg, self.sigma)

    def _get_h(self, arm: int, beta: np.ndarray) -> float:
        values = [
            _minimum_quantity_dom(self.means[arm], self.means[other], self.epsilon_1)
            - beta[arm, other] + _REFERENCE_INFINITY * float(arm == other)
            for other in range(self.num_arms)
        ]
        return min(values)

    def _all_h(self, beta: np.ndarray) -> np.ndarray:
        """Vectorized ``[self._get_h(i, beta) for i in range(K)]``."""
        values = _pairwise_dom(self.means, self.epsilon_1) - beta
        values[np.diag_indices(self.num_arms)] += _REFERENCE_INFINITY
        return np.min(values, axis=1)

    def _all_g(self, beta: np.ndarray, opt_arms: np.ndarray) -> np.ndarray:
        """Vectorized ``[self._get_g(i, beta, opt_arms) for i in range(K)]``."""
        values = _pairwise_non_dom(self.means, 0.0) - beta
        if self.epsilon_2 > 0.0 and opt_arms.size:
            values[:, opt_arms] += self.epsilon_2
        values[np.diag_indices(self.num_arms)] -= _REFERENCE_INFINITY
        return np.max(values, axis=1)

    def _get_g(self, arm: int, beta: np.ndarray, opt_arms: np.ndarray) -> float:
        opt_set = set(opt_arms.tolist())
        values = [
            _minimum_quantity_non_dom(self.means[arm], self.means[other], 0.0)
            - beta[arm, other]
            + self.epsilon_2 * float(self.epsilon_2 > 0.0 and other in opt_set)
            - _REFERENCE_INFINITY * float(arm == other)
            for other in range(self.num_arms)
        ]
        return max(values)

    def _stopping_diagnostics(self, beta: np.ndarray, pareto_arms: np.ndarray,
                              opt_arms: np.ndarray,
                              h_values: Optional[np.ndarray] = None) -> tuple[float, float, int, float]:
        if h_values is None:
            h_values = self._all_h(beta)
        g_plain = self._all_g(beta, np.empty(0, dtype=int))
        g_opt = self._all_g(beta, opt_arms)

        non_pareto = np.setdiff1d(np.arange(self.num_arms), pareto_arms, assume_unique=False)
        z1_t = float(np.min(h_values[pareto_arms])) if pareto_arms.size else _REFERENCE_INFINITY
        z2_t = (float(np.min(np.maximum(h_values[non_pareto], g_plain[non_pareto])))
                if non_pareto.size else _REFERENCE_INFINITY)
        z3_t = len(opt_arms)
        z4_t = float(np.min(np.maximum(h_values, g_opt)))
        return float(z1_t), float(z2_t), z3_t, float(z4_t)

    def _reference_selection(self, beta: np.ndarray, opt_comp_arms: np.ndarray) -> tuple[int, int, int]:
        dom = _pairwise_dom(self.means, 0.0)

        b_matrix = dom[opt_comp_arms] + beta.T[opt_comp_arms]
        b_matrix[np.arange(opt_comp_arms.size), opt_comp_arms] += _REFERENCE_INFINITY
        b_values = np.min(b_matrix, axis=1)
        b_t = int(opt_comp_arms[int(np.argmax(b_values))])

        c_values = dom[b_t] - beta[b_t]
        c_values[b_t] += _REFERENCE_INFINITY
        c_t = int(np.argmin(c_values))

        selected_arm = c_t if self.pulls[b_t] > self.pulls[c_t] else b_t
        return b_t, c_t, selected_arm

    def choose_arm(self) -> int:
        warmup_arms = np.flatnonzero(self.pulls == 0)
        if warmup_arms.size:
            selected_arm = int(warmup_arms[0])
            self.last_decision = APEDecision(
                selected_arm, self.pulls.copy(), self.means.copy(), None,
                np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=int),
                None, None, None, None, None, None, False, False,
            )
            return selected_arm

        beta = self._beta()
        pareto_arms = np.flatnonzero(_pareto_optimal_arms_mask(self.means))
        h_values = self._all_h(beta)
        opt_arms = np.flatnonzero(h_values > 0.0)
        opt_comp_arms = np.flatnonzero(h_values <= 0.0)
        z1_t, z2_t, z3_t, z4_t = self._stopping_diagnostics(beta, pareto_arms, opt_arms, h_values)
        reference_should_stop = not ((z1_t < 0.0 or z2_t < 0.0) and z3_t < self.target_optimal_arms and z4_t < 0.0)

        if not opt_comp_arms.size:
            # ``get_bt`` is undefined on an empty candidate set. The reference
            # loop has necessarily stopped before this point, so balanced
            # sampling is an explicit fixed-budget extension.
            selected_arm = int(np.argmin(self.pulls))
            b_t = c_t = None
            fixed_budget_continuation = True
        else:
            b_t, c_t, selected_arm = self._reference_selection(beta, opt_comp_arms)
            fixed_budget_continuation = reference_should_stop

        self.last_decision = APEDecision(
            selected_arm, self.pulls.copy(), self.means.copy(), beta.copy(), pareto_arms,
            opt_arms, opt_comp_arms, b_t, c_t, z1_t, z2_t, z3_t, z4_t,
            reference_should_stop, fixed_budget_continuation,
        )
        return selected_arm

    def learn(self, arm: int, reward: np.ndarray) -> None:
        reward = np.asarray(reward, dtype=float)
        if reward.shape != (self.num_objectives,):
            raise ValueError(f"reward must have shape ({self.num_objectives},), got {reward.shape}")
        n = self.pulls[arm]
        self.means[arm] = (reward + n * self.means[arm]) / (n + 1)
        self.pulls[arm] += 1

    def get_top_arms(self) -> np.ndarray:
        return np.flatnonzero(_pareto_optimal_arms_mask(self.means))
