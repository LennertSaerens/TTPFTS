import numpy as np
from paretoset import paretoset
from numba import njit


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

    # Mahalanobis term: sum of diff^2 / var_avg
    term1 = 0.125 * np.sum(diff ** 2 / var_avg)

    # Determinant term for diagonal matrices: product of elements
    # log(det(Sigma)) = sum(log(var_avg)), etc.
    log_det_avg = np.sum(np.log(var_avg))
    log_det1 = np.sum(np.log(var1))
    log_det2 = np.sum(np.log(var2))
    term2 = 0.5 * (log_det_avg - 0.5 * (log_det1 + log_det2))

    D_B = term1 + term2
    return np.exp(-D_B)


@njit
def diag_cov_from_stds(stds):
    stds = np.asarray(stds)
    return np.diag(stds ** 2)


def uncertainty_quantification(posterior_df):
    """Compute average Bhattacharyya coefficient between Pareto and second-front arms."""
    means_array = np.vstack(posterior_df["means"].values)
    num_objectives = means_array.shape[1]
    pareto_optimal_idx = paretoset(
        means_array,
        sense=["max"] * num_objectives,
        distinct=False
    )
    pareto_optimal_arms_df = posterior_df.loc[pareto_optimal_idx]

    pareto_suboptimal_arms = posterior_df.loc[~pareto_optimal_idx]
    if pareto_suboptimal_arms.empty:
        return 0

    pareto_suboptimal_means = np.vstack(pareto_suboptimal_arms["means"].values)
    optimal_suboptimal_pareto_idx = paretoset(
        pareto_suboptimal_means,
        sense=["max"] * num_objectives,
        distinct=False,
    )
    pareto_optimal_suboptimal_arms_df = pareto_suboptimal_arms.loc[optimal_suboptimal_pareto_idx]

    # Extract arrays for vectorized computation
    opt_means = np.vstack(pareto_optimal_arms_df["means"].values)
    opt_stds = np.vstack(pareto_optimal_arms_df["stds"].values)
    sub_means = np.vstack(pareto_optimal_suboptimal_arms_df["means"].values)
    sub_stds = np.vstack(pareto_optimal_suboptimal_arms_df["stds"].values)

    opt_vars = opt_stds ** 2
    sub_vars = sub_stds ** 2

    total_bc = 0.0
    total_comps = len(opt_means) * len(sub_means)

    if total_comps == 0:
        return 0

    for i in range(len(opt_means)):
        for j in range(len(sub_means)):
            total_bc += bhattacharyya_coeff_diag(
                opt_means[i], opt_vars[i], sub_means[j], sub_vars[j]
            )

    return total_bc / total_comps
