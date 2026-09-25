"""Independent APE trace oracle for validating ``bandits.APE``.

This module intentionally mirrors the supplied source rather than importing
production helpers.  Citations: ``psi_ape::loop``, ``psi_ape::get_bt``, and
``psi_ape::get_ct`` in
``8606_Adaptive_Algorithms_for_R_Supplementary Material/bandit_lib_c++/src/cpp/policies.{cxx,hpp}``; and
``Cg``, ``betaij``, ``get_g``, ``get_h``, ``get_z1t``, ``get_z2t``, and
``get_z_tilde`` in ``src/cpp/utils.{hpp,cxx}``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INF = 1e12


def cg(delta: float) -> float:
    log_inverse_delta = np.log(1.0 / delta)
    return log_inverse_delta / 2.0 + max(np.log(log_inverse_delta / 2.0), 0.0)


def betaij(t_i: float, t_j: float, c_g: float, sigma: float) -> float:
    return float(sigma * np.sqrt(2.0 * (2.0 * c_g + 2.0 * np.log(4.0 + np.log(t_i)) + 2.0 * np.log(4.0 + np.log(t_j))) * (1.0 / t_i + 1.0 / t_j)))


def min_dom(x_i: np.ndarray, x_j: np.ndarray, epsilon: float) -> float:
    return float(np.max(x_i + epsilon - x_j))


def min_non_dom(x_i: np.ndarray, x_j: np.ndarray, epsilon: float) -> float:
    return float(np.min(x_j - x_i + epsilon))


def pareto_mask(means: np.ndarray) -> np.ndarray:
    result = np.ones(len(means), dtype=bool)
    for i in range(len(means)):
        for j in range(len(means)):
            if i != j and np.all(means[i] <= means[j]) and np.any(means[i] < means[j]):
                result[i] = False
                break
    return result


@dataclass(frozen=True)
class OracleDecision:
    selected_arm: int
    counts: np.ndarray
    means: np.ndarray
    beta: np.ndarray | None
    pareto_arms: np.ndarray
    opt_arms: np.ndarray
    opt_comp_arms: np.ndarray
    b_t: int | None
    c_t: int | None
    z1_t: float | None
    z2_t: float | None
    z3_t: int | None
    z4_t: float | None
    reference_should_stop: bool
    fixed_budget_continuation: bool


class ReferenceFaithfulAPEOracle:
    """Line-for-line semantic oracle; fixed-budget continuation is explicit."""

    def __init__(self, num_arms: int, num_objectives: int, *, delta: float = 0.1, sigma: float = 1.0,
                 epsilon_1: float = 0.0, epsilon_2: float = 0.0, target_optimal_arms: int = 1 << 16) -> None:
        self.K = num_arms
        self.D = num_objectives
        self.delta = delta
        self.sigma = sigma
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        self.target_optimal_arms = target_optimal_arms
        self.c_g = cg(delta)
        self.counts = np.zeros(num_arms, dtype=int)
        self.means = np.zeros((num_arms, num_objectives), dtype=float)
        self.last_decision: OracleDecision | None = None

    def _beta(self) -> np.ndarray:
        return np.array([[betaij(self.counts[i], self.counts[j], self.c_g, self.sigma) for j in range(self.K)] for i in range(self.K)])

    def _h(self, i: int, beta: np.ndarray) -> float:
        return min(min_dom(self.means[i], self.means[j], self.epsilon_1) - beta[i, j] + INF * (i == j) for j in range(self.K))

    def _g(self, i: int, beta: np.ndarray, opt: np.ndarray) -> float:
        return max(min_non_dom(self.means[i], self.means[j], 0.0) - beta[i, j] + self.epsilon_2 * (self.epsilon_2 > 0.0 and j in opt) - INF * (i == j) for j in range(self.K))

    def choose_arm(self) -> int:
        warmup = np.flatnonzero(self.counts == 0)
        if len(warmup):
            selected = int(warmup[0])
            self.last_decision = OracleDecision(selected, self.counts.copy(), self.means.copy(), None,
                                                np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=int),
                                                None, None, None, None, None, None, False, False)
            return selected

        beta = self._beta()
        st = np.flatnonzero(pareto_mask(self.means))
        opt = np.array([i for i in range(self.K) if self._h(i, beta) > 0.0], dtype=int)
        opt_comp = np.array([i for i in range(self.K) if i not in opt], dtype=int)
        st_comp = np.array([i for i in range(self.K) if i not in st], dtype=int)
        z1 = min((self._h(i, beta) for i in st), default=INF)
        z2 = min((max(self._h(i, beta), self._g(i, beta, np.empty(0, dtype=int))) for i in st_comp), default=INF)
        z3 = len(opt)
        z4 = min(max(self._h(i, beta), self._g(i, beta, opt)) for i in range(self.K))
        reference_should_stop = not ((z1 < 0.0 or z2 < 0.0) and z3 < self.target_optimal_arms and z4 < 0.0)

        if not len(opt_comp):
            selected = int(np.argmin(self.counts))
            b_t = c_t = None
            continuation = True
        else:
            b_values = [min(min_dom(self.means[i], self.means[j], 0.0) + beta[j, i] + INF * (i == j) for j in range(self.K)) for i in opt_comp]
            b_t = int(opt_comp[int(np.argmax(b_values))])
            c_t = int(np.argmin([min_dom(self.means[b_t], self.means[i], 0.0) - beta[b_t, i] + INF * (i == b_t) for i in range(self.K)]))
            selected = c_t if self.counts[b_t] > self.counts[c_t] else b_t
            continuation = reference_should_stop

        self.last_decision = OracleDecision(selected, self.counts.copy(), self.means.copy(), beta, st, opt, opt_comp,
                                            b_t, c_t, z1, z2, z3, z4, reference_should_stop, continuation)
        return selected

    def learn(self, arm: int, reward: np.ndarray) -> None:
        self.means[arm] = (reward + self.counts[arm] * self.means[arm]) / (self.counts[arm] + 1)
        self.counts[arm] += 1
