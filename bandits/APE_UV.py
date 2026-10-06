"""Unknown-variance Adaptive Pareto Exploration (APE-UV).

APE's candidate/challenger sampling rule (``bandits/APE.py``) with the
known sub-Gaussian bound ``sigma`` replaced by variances estimated online.

For arm ``i``, objective ``d`` and ``n >= 2`` samples the confidence radius is

    r_{i,d}(n) = t_{n-1}^{-1}(1 - alpha_n / 2) * s_{i,d}(n) / sqrt(n),
    alpha_n    = 6 delta / (pi^2 K D n^2),

with ``s`` the sample standard deviation. The union over arms, objectives and
*every* sample count makes all intervals hold simultaneously with probability
at least ``1 - delta`` for i.i.d. Gaussian rewards, irrespective of the
adaptive sampling. The pairwise width for objective ``d`` is
``w_ij^d = r_{i,d} + r_{j,d}`` and replaces APE's scalar ``beta_ij``:

    L[i, j] = max_d (mu_i^d - mu_j^d - w_ij^d)   (lower bound on M(i, j))
    U[i, j] = max_d (mu_i^d - mu_j^d + w_ji^d)   (upper bound on M(i, j))

Selection is otherwise APE's: candidates are arms with ``min_j L[i, j] <= 0``,
``b_t = argmax_cand min_j U``, ``c_t = argmin_j L[b_t, j]``, and the less
sampled of the two is pulled. When every arm is certified, balanced sampling
is the same explicit fixed-budget extension as in ``APEAnytimeBandit``.

The test-only ``known_bound`` width mode sets ``w_ij^d = beta_ij(T_i, T_j)``.
Rounding is monotone, so ``max_d fl(x_d - b) == fl(max_d x_d - b)``, and that
mode reproduces ``APEAnytimeBandit`` bit for bit. When arm ``a`` is pulled
only row and column ``a`` of ``L``/``U`` change; they are recomputed with the
same elementwise expressions as a full rebuild, so the incrementally
maintained matrices are bitwise identical to recomputing them from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import t as student_t

from bandits.APE import _pareto_optimal_arms_mask, beta_ij, confidence_constant
from bandits.InterfaceMOMABPFI import BaseMOMABAlgorithm

WIDTH_MODES = ("unknown_variance", "known_bound")


def simultaneous_alpha(n: np.ndarray | int, num_arms: int, num_objectives: int, delta: float) -> np.ndarray:
    """Per-interval error level ``6 delta / (pi^2 K D n^2)`` (sums to delta over K, D, n >= 1)."""
    n = np.asarray(n, dtype=float)
    return 6.0 * delta / (np.pi**2 * num_arms * num_objectives * n**2)


class StudentQuantileTable:
    """Deterministic, lazily extended table of ``t_{n-1}^{-1}(1 - alpha_n / 2)``.

    ``isf`` evaluates the upper tail directly: for large ``n`` the level
    ``1 - alpha_n / 2`` rounds to exactly 1.0, where ``ppf`` would return inf.
    Values are elementwise, so the table does not depend on how it was grown.
    """

    def __init__(self, num_arms: int, num_objectives: int, delta: float) -> None:
        self.num_arms, self.num_objectives, self.delta = int(num_arms), int(num_objectives), float(delta)
        self._values = np.full(2, np.inf)

    def _compute(self, n: np.ndarray) -> np.ndarray:
        alpha = simultaneous_alpha(n, self.num_arms, self.num_objectives, self.delta)
        return student_t.isf(alpha / 2.0, n - 1)

    def ensure(self, n_max: int) -> None:
        if n_max < len(self._values):
            return
        new_length = max(int(n_max) + 1, 2 * len(self._values), 1024)
        n = np.arange(len(self._values), new_length, dtype=float)
        values = self._compute(n)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise FloatingPointError("non-finite Student-t quantile in APE-UV table")
        self._values = np.concatenate([self._values, values])

    def __getitem__(self, n):
        n_array = np.asarray(n)
        self.ensure(int(np.max(n_array)))
        return self._values[n_array]


_QUANTILE_TABLES: dict[tuple[int, int, float], StudentQuantileTable] = {}


def quantile_table(num_arms: int, num_objectives: int, delta: float) -> StudentQuantileTable:
    key = (int(num_arms), int(num_objectives), float(delta))
    if key not in _QUANTILE_TABLES:
        _QUANTILE_TABLES[key] = StudentQuantileTable(*key)
    return _QUANTILE_TABLES[key]


@dataclass(frozen=True)
class APEUVDecision:
    """Lightweight record of the latest selection (no matrix copies)."""

    selected_arm: int
    warmup: bool
    b_t: Optional[int]
    c_t: Optional[int]
    num_candidates: Optional[int]
    fixed_budget_continuation: bool


class APEUnknownVarianceBandit(BaseMOMABAlgorithm):
    """APE sampling with online per-arm/objective variance estimates.

    No noise information is accepted: there is no ``sigma`` argument and
    ``reset(env_stds)`` ignores its argument. ``num_warmup_pulls`` round-robin
    pulls per arm (default 4, matching TTPFTS) are counted within the budget.
    """

    def __init__(
        self,
        num_arms: int,
        num_objectives: int,
        *,
        delta: float = 0.1,
        num_warmup_pulls: int = 4,
        width_mode: str = "unknown_variance",
        known_sigma: Optional[float] = None,
    ) -> None:
        super().__init__(num_arms, num_objectives)
        if num_arms < 2:
            raise ValueError("APE-UV requires at least two arms")
        if width_mode not in WIDTH_MODES:
            raise ValueError(f"width_mode must be one of {WIDTH_MODES}")
        if width_mode == "unknown_variance":
            if known_sigma is not None:
                raise ValueError("the unknown-variance mode must not receive a noise level")
            if num_warmup_pulls < 2:
                raise ValueError("variance estimates need at least two warm-up pulls per arm")
        elif known_sigma is None or known_sigma <= 0.0:
            raise ValueError("known_bound mode requires a positive known_sigma")
        if num_warmup_pulls < 1:
            raise ValueError("num_warmup_pulls must be at least one")
        self.delta = float(delta)
        self.width_mode = width_mode
        self.known_sigma = None if known_sigma is None else float(known_sigma)
        self.num_warmup_pulls = int(num_warmup_pulls)
        self._cg = confidence_constant(delta)
        self._quantiles = (
            quantile_table(num_arms, num_objectives, delta) if width_mode == "unknown_variance" else None
        )
        self.reset(None)

    # ------------------------------------------------------------------ state
    def reset(self, env_stds) -> None:
        del env_stds
        self.pulls = np.zeros(self.num_arms, dtype=int)
        self.means = np.zeros((self.num_arms, self.num_objectives), dtype=float)
        self.m2 = np.zeros((self.num_arms, self.num_objectives), dtype=float)
        self.radii = np.full((self.num_arms, self.num_objectives), np.inf)
        self._L: Optional[np.ndarray] = None
        self._U: Optional[np.ndarray] = None
        self.last_decision: Optional[APEUVDecision] = None

    def learn(self, arm: int, reward: np.ndarray) -> None:
        reward = np.asarray(reward, dtype=float)
        if reward.shape != (self.num_objectives,):
            raise ValueError(f"reward must have shape ({self.num_objectives},), got {reward.shape}")
        n = self.pulls[arm]
        previous = self.means[arm].copy()
        self.means[arm] = (reward + n * self.means[arm]) / (n + 1)
        self.pulls[arm] += 1
        self.m2[arm] += (reward - previous) * (reward - self.means[arm])
        if self.width_mode == "unknown_variance":
            self.radii[arm] = self._radius(arm)
        if self._L is not None:
            self._update_index(arm)

    # ----------------------------------------------------------------- widths
    def _radius(self, arm: int) -> np.ndarray:
        n = int(self.pulls[arm])
        if n < 2:
            return np.full(self.num_objectives, np.inf)
        return (self._quantiles[n] * np.sqrt(self.m2[arm] / (n - 1))) / np.sqrt(n)

    def _all_radii(self) -> np.ndarray:
        n = self.pulls
        radii = np.full((self.num_arms, self.num_objectives), np.inf)
        valid = n >= 2
        if np.any(valid):
            q = self._quantiles[n[valid]]
            radii[valid] = (q[:, None] * np.sqrt(self.m2[valid] / (n[valid] - 1)[:, None])) / np.sqrt(n[valid])[:, None]
        return radii

    def _width_row(self, arm: int) -> np.ndarray:
        """``W[arm, j]`` for every ``j`` (broadcastable to ``(K, D)``)."""
        if self.width_mode == "unknown_variance":
            return self.radii[arm][None, :] + self.radii
        return beta_ij(self.pulls[arm], self.pulls, self._cg, self.known_sigma)[:, None]

    def _width_column(self, arm: int) -> np.ndarray:
        """``W[i, arm]`` for every ``i`` (broadcastable to ``(K, D)``)."""
        if self.width_mode == "unknown_variance":
            return self.radii + self.radii[arm][None, :]
        return beta_ij(self.pulls, self.pulls[arm], self._cg, self.known_sigma)[:, None]

    # --------------------------------------------------------- index matrices
    def full_index_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Recompute ``(L, U)`` from scratch (reference for the incremental path)."""
        diff = self.means[:, None, :] - self.means[None, :, :]
        if self.width_mode == "unknown_variance":
            radii = self._all_radii()
            widths = radii[:, None, :] + radii[None, :, :]
        else:
            widths = beta_ij(self.pulls[:, None], self.pulls[None, :], self._cg, self.known_sigma)[:, :, None]
        lower = np.max(diff - widths, axis=2)
        upper = np.max(diff + np.swapaxes(widths, 0, 1), axis=2)
        np.fill_diagonal(lower, np.inf)
        np.fill_diagonal(upper, np.inf)
        return lower, upper

    def _update_index(self, arm: int) -> None:
        row_width, column_width = self._width_row(arm), self._width_column(arm)
        row_diff = self.means[arm][None, :] - self.means
        column_diff = self.means - self.means[arm][None, :]
        self._L[arm, :] = np.max(row_diff - row_width, axis=1)
        self._U[arm, :] = np.max(row_diff + column_width, axis=1)
        self._L[:, arm] = np.max(column_diff - column_width, axis=1)
        self._U[:, arm] = np.max(column_diff + row_width, axis=1)
        self._L[arm, arm] = np.inf
        self._U[arm, arm] = np.inf

    def index_matrices(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """The incrementally maintained ``(L, U)``, or None before warm-up completes."""
        return None if self._L is None else (self._L, self._U)

    # -------------------------------------------------------------- selection
    def choose_arm(self) -> int:
        if int(self.pulls.min()) < self.num_warmup_pulls:
            selected = int(np.argmin(self.pulls))
            self.last_decision = APEUVDecision(selected, True, None, None, None, False)
            return selected
        if self._L is None:
            if self.width_mode == "unknown_variance":
                self.radii = self._all_radii()
            self._L, self._U = self.full_index_matrices()

        h_values = np.min(self._L, axis=1)
        candidates = np.flatnonzero(h_values <= 0.0)
        if not candidates.size:
            selected = int(np.argmin(self.pulls))
            self.last_decision = APEUVDecision(selected, False, None, None, 0, True)
            return selected
        b_values = np.min(self._U[candidates], axis=1)
        b_t = int(candidates[int(np.argmax(b_values))])
        c_t = int(np.argmin(self._L[b_t]))
        selected = c_t if self.pulls[b_t] > self.pulls[c_t] else b_t
        self.last_decision = APEUVDecision(selected, False, b_t, c_t, int(candidates.size), False)
        return selected

    def get_top_arms(self) -> np.ndarray:
        return np.flatnonzero(_pareto_optimal_arms_mask(self.means))
