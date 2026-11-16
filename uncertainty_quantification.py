import numpy as np
from paretoset import paretoset
from numba import njit
from scipy.special import log_ndtr


# ---------------------------------------------------------------------------
# Pairwise distribution measures (all for diagonal-covariance Gaussians)
# ---------------------------------------------------------------------------

@njit
def bhattacharyya_coeff_gaussians(mu1, Sigma1, mu2, Sigma2):
    """
    Bhattacharyya coefficient between two multivariate Gaussians.
    mu1, mu2: (d,) arrays
    Sigma1, Sigma2: (d,d) covariance matrices (SPD).
    """
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    Sigma1 = np.asarray(Sigma1)
    Sigma2 = np.asarray(Sigma2)

    Sigma = 0.5 * (Sigma1 + Sigma2)

    diff = mu1 - mu2
    inv_Sigma = np.linalg.inv(Sigma)
    term1 = 0.125 * diff.T @ inv_Sigma @ diff

    det_Sigma = np.linalg.det(Sigma)
    det_Sigma1 = np.linalg.det(Sigma1)
    det_Sigma2 = np.linalg.det(Sigma2)

    term2 = 0.5 * np.log(det_Sigma / np.sqrt(det_Sigma1 * det_Sigma2))

    D_B = term1 + term2
    BC = np.exp(-D_B)
    return BC


@njit
def bhattacharyya_coeff_diag(mu1, var1, mu2, var2):
    """
    Optimized Bhattacharyya coefficient for diagonal covariance matrices.
    mu1, mu2: (d,) mean arrays
    var1, var2: (d,) variance arrays (diagonal elements)
    Avoids full matrix inverse/determinant — O(d) instead of O(d^3).
    """
    var_avg = 0.5 * (var1 + var2)
    diff = mu1 - mu2

    term1 = 0.125 * np.sum(diff ** 2 / var_avg)

    log_det_avg = np.sum(np.log(var_avg))
    log_det1 = np.sum(np.log(var1))
    log_det2 = np.sum(np.log(var2))
    term2 = 0.5 * (log_det_avg - 0.5 * (log_det1 + log_det2))

    D_B = term1 + term2
    return np.exp(-D_B)


@njit
def find_max_overlap_pair(F1_indices, F2_indices, means, vars):
    """
    Finds the pair of arms (i, j) with i in F1_indices and j in F2_indices
    that maximizes the Bhattacharyya coefficient (overlap).
    Returns (best_i, best_j).
    """
    max_bc = -1.0
    best_i = -1
    best_j = -1
    
    for i_idx in range(len(F1_indices)):
        i = F1_indices[i_idx]
        for j_idx in range(len(F2_indices)):
            j = F2_indices[j_idx]
            bc = bhattacharyya_coeff_diag(means[i], vars[i], means[j], vars[j])
            if bc > max_bc:
                max_bc = bc
                best_i = i
                best_j = j
                
    return best_i, best_j


@njit
def kl_divergence_diag(mu1, var1, mu2, var2):
    """
    Symmetric KL-divergence (Jensen divergence) between two diagonal Gaussians.
    Returns 0.5 * (KL(p||q) + KL(q||p)) where p = N(mu1, diag(var1)).
    """
    d = mu1.shape[0]
    diff = mu1 - mu2
    diff_sq = diff ** 2

    # KL(p||q) = 0.5 * [tr(Σ_q^{-1} Σ_p) + (μ_q - μ_p)^T Σ_q^{-1} (μ_q - μ_p) - d + ln(det Σ_q / det Σ_p)]
    kl_pq = 0.5 * (np.sum(var1 / var2) + np.sum(diff_sq / var2) - d + np.sum(np.log(var2) - np.log(var1)))
    kl_qp = 0.5 * (np.sum(var2 / var1) + np.sum(diff_sq / var1) - d + np.sum(np.log(var1) - np.log(var2)))

    return 0.5 * (kl_pq + kl_qp)


@njit
def gaussian_entropy_diag(var):
    """
    Differential entropy of a multivariate Gaussian with diagonal covariance.
    H = 0.5 * d * (1 + ln(2π)) + 0.5 * Σ_j ln(σ²_j)
    """
    d = var.shape[0]
    return 0.5 * d * (1.0 + np.log(2.0 * np.pi)) + 0.5 * np.sum(np.log(var))


@njit
def diag_cov_from_stds(stds):
    stds = np.asarray(stds)
    return np.diag(stds ** 2)


# ---------------------------------------------------------------------------
# Front extraction helper
# ---------------------------------------------------------------------------

def _extract_top_two_fronts(posterior_df):
    """
    Identify estimated Pareto front (front 1) and second front (front 2)
    from posterior means.  Returns dict with arrays or None if only one front.
    """
    means_array = np.vstack(posterior_df["means"].values)
    num_objectives = means_array.shape[1]
    sense = ["max"] * num_objectives

    pareto_idx = paretoset(means_array, sense=sense, distinct=False)
    opt_df = posterior_df.loc[pareto_idx]
    sub_df = posterior_df.loc[~pareto_idx]

    if sub_df.empty:
        return None

    sub_means = np.vstack(sub_df["means"].values)
    sub_pareto_idx = paretoset(sub_means, sense=sense, distinct=False)
    front2_df = sub_df.loc[sub_pareto_idx]

    return {
        "opt_means": np.vstack(opt_df["means"].values),
        "opt_stds": np.vstack(opt_df["stds"].values),
        "front2_means": np.vstack(front2_df["means"].values),
        "front2_stds": np.vstack(front2_df["stds"].values),
    }


# ---------------------------------------------------------------------------
# Top-level UQ functions
# ---------------------------------------------------------------------------

def uncertainty_quantification(posterior_df):
    """Compute average Bhattacharyya coefficient between front 1 and front 2 arms."""
    fronts = _extract_top_two_fronts(posterior_df)
    if fronts is None:
        return 0.0

    opt_vars = fronts["opt_stds"] ** 2
    sub_vars = fronts["front2_stds"] ** 2
    n_opt, n_sub = len(fronts["opt_means"]), len(fronts["front2_means"])
    total_comps = n_opt * n_sub
    if total_comps == 0:
        return 0.0

    total = 0.0
    for i in range(n_opt):
        for j in range(n_sub):
            total += bhattacharyya_coeff_diag(
                fronts["opt_means"][i], opt_vars[i],
                fronts["front2_means"][j], sub_vars[j],
            )
    return total / total_comps


def uncertainty_quantification_kl(posterior_df):
    """
    Compute average symmetric KL-divergence between front 1 and front 2 arms.
    Low KL → similar posteriors → high uncertainty about Pareto set.
    """
    fronts = _extract_top_two_fronts(posterior_df)
    if fronts is None:
        return 0.0

    opt_vars = fronts["opt_stds"] ** 2
    sub_vars = fronts["front2_stds"] ** 2
    n_opt, n_sub = len(fronts["opt_means"]), len(fronts["front2_means"])
    total_comps = n_opt * n_sub
    if total_comps == 0:
        return 0.0

    total = 0.0
    for i in range(n_opt):
        for j in range(n_sub):
            total += kl_divergence_diag(
                fronts["opt_means"][i], opt_vars[i],
                fronts["front2_means"][j], sub_vars[j],
            )
    return total / total_comps


def uncertainty_quantification_entropy(posterior_df):
    """
    Compute average differential entropy of front 1 ∪ front 2 arm posteriors.
    High entropy → wide posteriors → high uncertainty about arm values.
    """
    fronts = _extract_top_two_fronts(posterior_df)
    if fronts is None:
        return 0.0

    all_stds = np.vstack([fronts["opt_stds"], fronts["front2_stds"]])
    all_vars = all_stds ** 2
    n = len(all_vars)
    if n == 0:
        return 0.0

    total = 0.0
    for i in range(n):
        total += gaussian_entropy_diag(all_vars[i])
    return total / n


def uncertainty_quantification_all(posterior_df):
    """
    Compute all three UQ measures in a single pass (shared front decomposition).
    Returns (bhattacharyya, kl_divergence, entropy).
    """
    fronts = _extract_top_two_fronts(posterior_df)
    if fronts is None:
        return 0.0, 0.0, 0.0

    opt_vars = fronts["opt_stds"] ** 2
    sub_vars = fronts["front2_stds"] ** 2
    n_opt, n_sub = len(fronts["opt_means"]), len(fronts["front2_means"])
    total_comps = n_opt * n_sub

    # Bhattacharyya and KL (pairwise)
    total_bc = 0.0
    total_kl = 0.0
    if total_comps > 0:
        for i in range(n_opt):
            for j in range(n_sub):
                total_bc += bhattacharyya_coeff_diag(
                    fronts["opt_means"][i], opt_vars[i],
                    fronts["front2_means"][j], sub_vars[j],
                )
                total_kl += kl_divergence_diag(
                    fronts["opt_means"][i], opt_vars[i],
                    fronts["front2_means"][j], sub_vars[j],
                )
        total_bc /= total_comps
        total_kl /= total_comps

    # Entropy (pointwise over boundary arms)
    all_vars = np.vstack([opt_vars, sub_vars])
    total_entropy = 0.0
    for i in range(len(all_vars)):
        total_entropy += gaussian_entropy_diag(all_vars[i])
    total_entropy /= len(all_vars)

    return total_bc, total_kl, total_entropy


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


def uncertainty_quantification_v1(posterior_df):
    """Variant 1 (Directional Chernoff-Bhattacharyya overlap) from posterior DataFrame."""
    means = np.vstack(posterior_df["means"].values)
    stds = np.vstack(posterior_df["stds"].values)
    return uq_directional_bc_from_moments(means, stds)


def uncertainty_quantification_v2(posterior_df):
    """Variant 2 (Analytic dominance-reversal recommendation risk) from posterior DataFrame."""
    means = np.vstack(posterior_df["means"].values)
    stds = np.vstack(posterior_df["stds"].values)
    return uq_dominance_risk_from_moments(means, stds)


uncertainty_quantification_directional_bc = uncertainty_quantification_v1
uncertainty_quantification_dominance_risk = uncertainty_quantification_v2
