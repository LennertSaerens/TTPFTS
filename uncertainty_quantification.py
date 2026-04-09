import numpy as np
from paretoset import paretoset
from numba import njit


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
