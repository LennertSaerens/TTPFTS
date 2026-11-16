"""Pure, validity-aware posterior uncertainty summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from paretoset import paretoset
from scipy.special import log_ndtr

from environments.hardness import direct_gaps

@dataclass(frozen=True)
class UQResult:
    """UQ measures or an explicit scientific reason they are unavailable."""

    available: bool
    reason: str
    bhattacharyya: float | None
    symmetric_kl: float | None
    entropy: float | None
    first_front_size: int
    second_front_size: int


def _unavailable(reason: str, first_size: int = 0, second_size: int = 0) -> UQResult:
    return UQResult(False, reason, None, None, None, first_size, second_size)


def posterior_stds_from_tposterior_state(n: np.ndarray, m2: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    """Reconstruct actual posterior std(mu), preserving mathematically infinite values."""
    n = np.asarray(n, dtype=np.float64)
    m2 = np.asarray(m2, dtype=np.float64)
    if n.shape != m2.shape:
        raise ValueError("posterior n and M2 shapes differ")
    if np.any(n < 0) or np.any(m2 < 0) or not np.all(np.isfinite(m2)):
        raise ValueError("invalid TPosterior sufficient statistics")
    df = n + 2.0 * float(alpha) - 1.0
    if np.any(df <= 0):
        raise ValueError("posterior undefined: degrees of freedom are non-positive")
    scale = np.sqrt(m2 / (n * df))
    with np.errstate(divide="ignore", invalid="ignore"):
        variance_factor = np.where(df > 2.0, df / (df - 2.0), np.inf)
    return scale * np.sqrt(variance_factor)


def _fronts(means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = np.flatnonzero(paretoset(means, sense=["max"] * means.shape[1], distinct=False))
    remaining = np.setdiff1d(np.arange(means.shape[0]), first, assume_unique=True)
    if remaining.size == 0:
        return first, remaining
    second_mask = paretoset(means[remaining], sense=["max"] * means.shape[1], distinct=False)
    return first, remaining[np.flatnonzero(second_mask)]


def uncertainty_from_moments(means: np.ndarray, stds: np.ndarray) -> UQResult:
    """Compute the three existing moment-matched diagonal-Gaussian measures.

    The Student-t posterior is summarized by its posterior mean and actual
    marginal standard deviation. This preserves the former analysis's
    diagonal-Gaussian definitions while explicitly rejecting undefined states.
    """
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    if means.ndim != 2 or stds.shape != means.shape:
        raise ValueError("means and stds must be same-shaped two-dimensional arrays")
    if not np.all(np.isfinite(means)):
        return _unavailable("nonfinite_posterior_mean")
    if not np.all(np.isfinite(stds)) or np.any(stds <= 0):
        return _unavailable("nonfinite_or_nonpositive_posterior_std")
    first, second = _fronts(means)
    if second.size == 0:
        return _unavailable("no_second_estimated_front", int(first.size), 0)

    first_means, second_means = means[first], means[second]
    first_vars, second_vars = stds[first] ** 2, stds[second] ** 2
    total_bc = 0.0
    total_kl = 0.0
    for first_index in range(first.size):
        for second_index in range(second.size):
            var1 = first_vars[first_index]
            var2 = second_vars[second_index]
            var_avg = 0.5 * (var1 + var2)
            difference_squared = (first_means[first_index] - second_means[second_index]) ** 2
            db = 0.125 * np.sum(difference_squared / var_avg)
            db += 0.5 * (
                np.sum(np.log(var_avg))
                - 0.5 * (np.sum(np.log(var1)) + np.sum(np.log(var2)))
            )
            total_bc += float(np.exp(-db))
            kl12 = 0.5 * (
                np.sum(var1 / var2)
                + np.sum(difference_squared / var2)
                - means.shape[1]
                + np.sum(np.log(var2) - np.log(var1))
            )
            kl21 = 0.5 * (
                np.sum(var2 / var1)
                + np.sum(difference_squared / var1)
                - means.shape[1]
                + np.sum(np.log(var1) - np.log(var2))
            )
            total_kl += float(0.5 * (kl12 + kl21))
    pairs = first.size * second.size
    all_vars = np.vstack((first_vars, second_vars))
    entropy = 0.5 * means.shape[1] * (1.0 + np.log(2.0 * np.pi))
    entropy += 0.5 * np.mean(np.sum(np.log(all_vars), axis=1))
    return UQResult(
        True,
        "available",
        total_bc / pairs,
        total_kl / pairs,
        float(entropy),
        int(first.size),
        int(second.size),
    )


def uncertainty_from_tposterior_state(
    n: np.ndarray, means: np.ndarray, m2: np.ndarray, alpha: float = 0.0
) -> UQResult:
    """Compute UQ from serializable TPosterior Welford sufficient statistics."""
    try:
        stds = posterior_stds_from_tposterior_state(n, m2, alpha)
    except ValueError as exc:
        return _unavailable(str(exc))
    return uncertainty_from_moments(means, stds)


def true_pareto_boundary_pairs(true_means: np.ndarray, true_pareto_indices: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Fix each true Pareto arm's closest true dominated challenger.

    For true Pareto arm ``i`` and dominated arm ``j``, the direct hardness
    witness is ``max_c(mu_j[c] - mu_i[c])_+ + Delta_j``.  Keeping every
    minimizer produces a deterministic, ground-truth-conditioned boundary
    set. It cannot switch to an easier arm merely because a posterior estimate
    puts a dominated arm into the estimated Pareto front.
    """
    true_means = np.asarray(true_means, dtype=np.float64)
    pareto = np.asarray(true_pareto_indices, dtype=int)
    if true_means.ndim != 2:
        raise ValueError("true means must be two-dimensional")
    if pareto.ndim != 1 or pareto.size == 0:
        raise ValueError("true Pareto indices must be a nonempty vector")
    if len(np.unique(pareto)) != len(pareto) or np.any(pareto < 0) or np.any(pareto >= len(true_means)):
        raise ValueError("true Pareto indices are invalid")
    pareto_mask = np.zeros(len(true_means), dtype=bool)
    pareto_mask[pareto] = True
    dominated = np.flatnonzero(~pareto_mask)
    if dominated.size == 0:
        return ()
    gaps = direct_gaps(true_means, pareto_mask)
    pairs: list[tuple[int, int]] = []
    for optimal in pareto:
        scores = np.asarray([
            max(float(np.max(true_means[dominated_arm] - true_means[optimal])), 0.0) + gaps[dominated_arm]
            for dominated_arm in dominated
        ])
        minimum = float(scores.min())
        for dominated_arm in dominated[np.isclose(scores, minimum, rtol=0.0, atol=1e-12)]:
            pairs.append((int(optimal), int(dominated_arm)))
    return tuple(sorted(set(pairs)))


def truth_conditioned_uncertainty_from_moments(
    means: np.ndarray, stds: np.ndarray, boundary_pairs: tuple[tuple[int, int], ...]
) -> UQResult:
    """Compute UQ over a fixed ground-truth Pareto/dominated boundary."""
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    if means.ndim != 2 or stds.shape != means.shape:
        raise ValueError("means and stds must be same-shaped two-dimensional arrays")
    if not boundary_pairs:
        return _unavailable("no_true_dominated_boundary")
    if not np.all(np.isfinite(means)):
        return _unavailable("nonfinite_posterior_mean")
    if not np.all(np.isfinite(stds)) or np.any(stds <= 0):
        return _unavailable("nonfinite_or_nonpositive_posterior_std")

    total_bc = 0.0
    total_kl = 0.0
    boundary_arms = set()
    for optimal, dominated in boundary_pairs:
        if not (0 <= optimal < len(means) and 0 <= dominated < len(means)):
            raise ValueError("boundary pair index out of range")
        boundary_arms.update((optimal, dominated))
        var1 = stds[optimal] ** 2
        var2 = stds[dominated] ** 2
        var_avg = 0.5 * (var1 + var2)
        difference_squared = (means[optimal] - means[dominated]) ** 2
        db = 0.125 * np.sum(difference_squared / var_avg)
        db += 0.5 * (
            np.sum(np.log(var_avg))
            - 0.5 * (np.sum(np.log(var1)) + np.sum(np.log(var2)))
        )
        total_bc += float(np.exp(-db))
        kl12 = 0.5 * (
            np.sum(var1 / var2)
            + np.sum(difference_squared / var2)
            - means.shape[1]
            + np.sum(np.log(var2) - np.log(var1))
        )
        kl21 = 0.5 * (
            np.sum(var2 / var1)
            + np.sum(difference_squared / var1)
            - means.shape[1]
            + np.sum(np.log(var1) - np.log(var2))
        )
        total_kl += float(0.5 * (kl12 + kl21))
    boundary_arm_indices = sorted(boundary_arms)
    entropy = 0.5 * means.shape[1] * (1.0 + np.log(2.0 * np.pi))
    entropy += 0.5 * np.mean(np.sum(np.log(stds[boundary_arm_indices] ** 2), axis=1))
    return UQResult(
        True,
        "available",
        total_bc / len(boundary_pairs),
        total_kl / len(boundary_pairs),
        float(entropy),
        len({optimal for optimal, _ in boundary_pairs}),
        len({dominated for _, dominated in boundary_pairs}),
    )


def truth_conditioned_uncertainty_from_tposterior_state(
    n: np.ndarray,
    means: np.ndarray,
    m2: np.ndarray,
    boundary_pairs: tuple[tuple[int, int], ...],
    alpha: float = 0.0,
) -> UQResult:
    """Compute fixed-boundary UQ directly from serialized TPosterior state."""
    try:
        stds = posterior_stds_from_tposterior_state(n, m2, alpha)
    except ValueError as exc:
        return _unavailable(str(exc))
    return truth_conditioned_uncertainty_from_moments(means, stds, boundary_pairs)


# ---------------------------------------------------------------------------
# Robust Uncertainty Quantification Variants (Issue #2)
# ---------------------------------------------------------------------------

def _validate_moments(means: np.ndarray, stds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(means, dtype=np.float64)
    stds = np.asarray(stds, dtype=np.float64)
    if means.ndim != 2 or stds.shape != means.shape:
        raise ValueError("means and stds must be same-shaped two-dimensional arrays")
    return means, stds


def uq_directional_bc_from_moments(means: np.ndarray, stds: np.ndarray) -> float:
    """Variant 1 (U_BC^dir): Directional Chernoff-Bhattacharyya front-boundary overlap.

    Retains only the 1D pairwise Gaussian Bhattacharyya coefficient beta_c(u, v)
    via the coordinate-level Chernoff bound q_c(u, v) = 0.5 * beta_c(u, v)^2 in (0, 0.5],
    while fixing three structural failure modes of Eq. (10):
      1. Coordinate-tie masking & directional asymmetry:
         - Dominated arm j not in F1 escapes dominator i in F1 if any coordinate flips:
           u_escape(i, j) = 1 - prod_c (1 - q_c(i, j)), bottlenecked by min_{i in D1(j)} u_escape(i, j).
         - Within F1, unordered pair {a, b} can collapse into dominance iff one arm wins on all
           deficit coordinates (weight q_c) and holds leading coordinates (weight 1 - q_c).
      2. Empty second front & within-F1 ambiguity: checking unordered pairs within F1 covers
         diffuse or nearly tied all-Pareto states cleanly without falling back to 0.
      3. Dilution: noisy-OR aggregation across unordered F1 pairs and outside arms j not in F1
         prevents distant resolved pairs from diluting local boundary uncertainty.
    """
    means, stds = _validate_moments(means, stds)
    if not (np.all(np.isfinite(means)) and np.all(np.isfinite(stds)) and np.all(stds > 0)):
        return float("nan")
    k, _d = means.shape
    if k <= 1:
        return 0.0

    # Pairwise 1D Bhattacharyya coefficient tensor beta_c(u, v) and Chernoff risk q_c = 0.5 * beta_c^2
    vars_ = stds ** 2
    v_sum = vars_[:, None, :] + vars_[None, :, :]
    diff_sq = (means[:, None, :] - means[None, :, :]) ** 2
    s_prod = stds[:, None, :] * stds[None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        bc = np.sqrt((2.0 * s_prod) / v_sum) * np.exp(-0.25 * diff_sq / v_sum)
    q = np.clip(0.5 * (bc ** 2), 0.0, 0.5)

    first_mask = paretoset(means, sense=["max"] * means.shape[1], distinct=False)
    first = np.flatnonzero(first_mask)
    rest = np.flatnonzero(~first_mask)

    survival = 1.0
    # 1. Within-F1 unordered pair collapse overlap
    for idx_a in range(first.size):
        a = int(first[idx_a])
        for idx_b in range(idx_a + 1, first.size):
            b = int(first[idx_b])
            def_ab = means[a] <= means[b]
            p_ab = float(np.prod(np.where(def_ab, q[a, b], 1.0 - q[a, b])))
            def_ba = means[b] <= means[a]
            p_ba = float(np.prod(np.where(def_ba, q[b, a], 1.0 - q[b, a])))
            u_ab = min(1.0, p_ab + p_ba)
            survival *= max(0.0, 1.0 - u_ab)

    # 2. Outside-F1 challenger escape overlap against all empirical dominators in F1
    if rest.size > 0:
        f1_means = means[first][:, None, :]
        rest_means = means[rest][None, :, :]
        dom_f1_mask = np.all(f1_means >= rest_means, axis=-1) & np.any(f1_means > rest_means, axis=-1)
        # Escape probabilities for every pair (i in F1, j in rest): shape (|F1|, |rest|)
        esc_matrix = 1.0 - np.prod(1.0 - q[np.ix_(first, rest)], axis=-1)
        for idx_j, j in enumerate(rest):
            dom_idx = np.flatnonzero(dom_f1_mask[:, idx_j])
            if dom_idx.size > 0:
                best_esc = float(np.min(esc_matrix[dom_idx, idx_j]))
            else:
                # Defensive fallback if numeric tie left non-F1 arm dominated only by intermediate front
                all_dom = [
                    u for u in range(k)
                    if u != j and np.all(means[u] >= means[j]) and np.any(means[u] > means[j])
                ]
                if not all_dom:
                    return 1.0
                best_esc = min(1.0 - float(np.prod(1.0 - q[u, j])) for u in all_dom)
            survival *= max(0.0, 1.0 - min(1.0, best_esc))

    return float(np.clip(1.0 - survival, 0.0, 1.0))


def uq_dominance_risk_from_moments(
    means: np.ndarray,
    stds: np.ndarray,
    correlation_divisor_scale: float = 0.5,
) -> float:
    """Variant 2 (U_risk): Decision-based analytic Pareto-set instability risk.

    Analytically approximates the exact posterior disagreement probability
    q_t = Pr_{Theta ~ Pi_t}[Pareto(Theta) != F1_hat] in O(K^2 d) time using:
      1. Exact pairwise dominance probabilities under independent diagonal Gaussian posteriors:
         P_{u > v} = prod_{c=1}^d Phi((mu_{u,c} - mu_{v,c}) / sqrt(sigma_{u,c}^2 + sigma_{v,c}^2)).
      2. Equivalence theorem for Pareto(Theta) = F1_hat:
         No pair within F1 becomes comparable AND every arm outside F1 stays dominated by F1.
      3. Coordinate-aligned shared-arm variance correlation discounting with rank-order scaling,
         preventing double-counting when multiple challengers share a noisy dominator, when an
         undersampled challenger faces many well-sampled dominators, or when threats act on
         orthogonal vs overlapping objectives.
    """
    means, stds = _validate_moments(means, stds)
    if not (np.all(np.isfinite(means)) and np.all(np.isfinite(stds)) and np.all(stds > 0)):
        return float("nan")
    k, d = means.shape
    if k <= 1:
        return 0.0

    vars_ = stds ** 2
    v_sum = vars_[:, None, :] + vars_[None, :, :]
    diff = means[:, None, :] - means[None, :, :]
    z = diff / np.sqrt(v_sum)
    log_p_coord = log_ndtr(z)
    p_coord = np.exp(log_p_coord)
    p_dom = np.exp(np.sum(log_p_coord, axis=-1))
    np.fill_diagonal(p_dom, 0.0)

    first_mask = paretoset(means, sense=["max"] * d, distinct=False)
    first = np.flatnonzero(first_mask)
    rest = np.flatnonzero(~first_mask)

    def _coord_weights(dom_idx: int, sub_idx: int) -> np.ndarray:
        fail = np.maximum(0.0, 1.0 - p_coord[dom_idx, sub_idx])
        total = float(fail.sum())
        if total > 0.0:
            return fail / total
        return np.full(d, 1.0 / d)

    def _pair_corr(a: int, b: int, common: int, w_a: np.ndarray, w_b: np.ndarray) -> float:
        rho_c = vars_[common] / np.sqrt((vars_[a] + vars_[common]) * (vars_[b] + vars_[common]))
        return float(np.sum(np.minimum(w_a, w_b) * rho_c))

    def _combine_correlated_survival(
        probs: list[float],
        other_indices: list[int],
        common: int,
        weights_list: list[np.ndarray],
    ) -> float:
        surv = 1.0
        for m, (prob, idx_m) in enumerate(zip(probs, other_indices)):
            if prob <= 1e-15:
                continue
            if m == 0:
                eff = 1.0
            else:
                rhos = [
                    _pair_corr(other_indices[prev], idx_m, common, weights_list[prev], weights_list[m])
                    for prev in range(m)
                ]
                eff = max(0.0, 1.0 - max(rhos)) / (1.0 + correlation_divisor_scale * sum(rhos))
            surv *= max(0.0, 1.0 - prob) ** eff
        return surv

    # 1. Within-F1 unordered pair dominance stability
    within_survival = 1.0
    groups_f1: dict[int, list[tuple[float, int]]] = {}
    for idx_a in range(first.size):
        a = int(first[idx_a])
        for idx_b in range(idx_a + 1, first.size):
            b = int(first[idx_b])
            r_ab = min(1.0, float(p_dom[a, b] + p_dom[b, a]))
            if r_ab <= 1e-15:
                continue
            if p_dom[a, b] >= p_dom[b, a]:
                groups_f1.setdefault(a, []).append((r_ab, b))
            else:
                groups_f1.setdefault(b, []).append((r_ab, a))

    for dom_u, items in groups_f1.items():
        items.sort(key=lambda item: item[0], reverse=True)
        probs = [item[0] for item in items]
        targets = [item[1] for item in items]
        weights = [_coord_weights(dom_u, target) for target in targets]
        within_survival *= _combine_correlated_survival(probs, targets, dom_u, weights)

    # 2. Outside-F1 challengers: must stay dominated by F1 (grouped by primary dominator)
    outside_survival = 1.0
    if rest.size > 0:
        r_enter = np.zeros(rest.size, dtype=np.float64)
        primary_dom = np.zeros(rest.size, dtype=int)
        for idx_j, j in enumerate(rest):
            order = first[np.argsort(p_dom[first, j])[::-1]]
            primary_dom[idx_j] = int(order[0])
            pos = [(float(p_dom[int(i), int(j)]), int(i)) for i in order if p_dom[int(i), int(j)] > 1e-15]
            if not pos:
                r_enter[idx_j] = 1.0
            else:
                probs = [item[0] for item in pos]
                dom_ids = [item[1] for item in pos]
                weights = [_coord_weights(i_id, int(j)) for i_id in dom_ids]
                r_enter[idx_j] = _combine_correlated_survival(probs, dom_ids, int(j), weights)

        groups_out: dict[int, list[tuple[float, int]]] = {}
        for idx_j, j in enumerate(rest):
            if r_enter[idx_j] > 1e-15:
                groups_out.setdefault(int(primary_dom[idx_j]), []).append((float(r_enter[idx_j]), int(j)))

        for dom_i, items in groups_out.items():
            items.sort(key=lambda item: item[0], reverse=True)
            probs = [item[0] for item in items]
            chals = [item[1] for item in items]
            weights = [_coord_weights(dom_i, ch) for ch in chals]
            outside_survival *= _combine_correlated_survival(probs, chals, dom_i, weights)

    return float(np.clip(1.0 - within_survival * outside_survival, 0.0, 1.0))


def uq_mc_disagreement_from_moments(
    means: np.ndarray,
    stds: np.ndarray,
    n_draws: int = 512,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    batch_size: int = 32,
) -> tuple[float, float]:
    """Monte Carlo reference q_t = Pr_{Theta ~ Pi_t}[Pareto(Theta) != F1_hat] and MC SE."""
    means, stds = _validate_moments(means, stds)
    if not (np.all(np.isfinite(means)) and np.all(np.isfinite(stds)) and np.all(stds > 0)):
        return float("nan"), float("nan")
    if means.shape[0] <= 1:
        return 0.0, 0.0
    if rng is None:
        rng = np.random.default_rng(seed)
    first_hat = paretoset(means, sense=["max"] * means.shape[1], distinct=False)
    n_disagree = 0
    for start in range(0, n_draws, batch_size):
        bs = min(batch_size, n_draws - start)
        draws = means[None, :, :] + rng.normal(size=(bs, *means.shape)) * stds[None, :, :]
        left = draws[:, :, None, :]
        right = draws[:, None, :, :]
        dom = np.all(left >= right, axis=-1) & np.any(left > right, axis=-1)
        first_draws = ~np.any(dom, axis=1)
        n_disagree += int(np.sum(np.any(first_draws != first_hat[None, :], axis=1)))
    q = float(n_disagree / n_draws)
    se = float(np.sqrt(q * (1.0 - q) / n_draws))
    return q, se
