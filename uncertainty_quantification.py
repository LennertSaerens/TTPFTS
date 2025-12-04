import numpy as np
from paretoset import paretoset

def bhattacharyya_coeff_gaussians(mu1, Sigma1, mu2, Sigma2):
    """
    Bhattacharyya coefficient between two multivariate Gaussians.
    mu1, mu2: (d,) arrays
    Sigma1, Sigma2: (d,d) covariance matrices (SPD).
    Works for any dimension d.
    """
    mu1 = np.asarray(mu1)
    mu2 = np.asarray(mu2)
    Sigma1 = np.asarray(Sigma1)
    Sigma2 = np.asarray(Sigma2)

    Sigma = 0.5 * (Sigma1 + Sigma2)

    # Mahalanobis term
    diff = mu1 - mu2
    inv_Sigma = np.linalg.inv(Sigma)
    term1 = 0.125 * diff.T @ inv_Sigma @ diff

    # Determinant term
    det_Sigma  = np.linalg.det(Sigma)
    det_Sigma1 = np.linalg.det(Sigma1)
    det_Sigma2 = np.linalg.det(Sigma2)

    term2 = 0.5 * np.log(det_Sigma / np.sqrt(det_Sigma1 * det_Sigma2))

    D_B = term1 + term2
    BC = np.exp(-D_B)
    return BC


def diag_cov_from_stds(stds):
    stds = np.asarray(stds)
    return np.diag(stds**2)


def uncertainty_quantification(posterior_df):
    means_array = np.vstack(posterior_df["means"].values)
    pareto_optimal_idx = paretoset(
        means_array,
        sense=["max"] * means_array.shape[1],
        distinct=False
    )
    pareto_optimal_arms_df = posterior_df.loc[pareto_optimal_idx]

    pareto_suboptimal_arms = posterior_df.loc[~pareto_optimal_idx]
    if pareto_suboptimal_arms.empty:
        return 0
    pareto_suboptimal_means = np.vstack(pareto_suboptimal_arms["means"].values)
    optimal_suboptimal_pareto_idx = paretoset(
        pareto_suboptimal_means,
        sense=["max"] * means_array.shape[1],
        distinct=False,
    )
    pareto_optimal_suboptimal_arms_df = pareto_suboptimal_arms.loc[optimal_suboptimal_pareto_idx]

    # Sum of Bhattacharyya coefficients over all optimal / optimal-suboptimal pairs
    total_bc = 0.0
    for _, opt_row in pareto_optimal_arms_df.iterrows():
        mu_opt = np.array(opt_row["means"])
        std_opt = np.array(opt_row["stds"])
        Sigma_opt = diag_cov_from_stds(std_opt)

        for _, sub_row in pareto_optimal_suboptimal_arms_df.iterrows():
            mu_sub = np.array(sub_row["means"])
            std_sub = np.array(sub_row["stds"])
            Sigma_sub = diag_cov_from_stds(std_sub)

            bc = bhattacharyya_coeff_gaussians(mu_opt, Sigma_opt, mu_sub, Sigma_sub)
            total_bc += bc

    return total_bc
