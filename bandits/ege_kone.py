import numpy as np
import math


def M(xi, xj):
    r""" @func utils cf paper"""
    return np.max(xi - xj, -1)


def m(xi, xj):
    r""" @func utils cf paper"""
    return np.min(xj - xi, -1)


def is_non_dominated(Y: np.ndarray, eps=0.) -> np.ndarray:
    r"""Computes the non-dominated front.
  @ Copyright: this function is modified from boTorch utils
    Note: this assumes maximization.

    For small `n`, this method uses a highly parallel methodology
    that compares all pairs of points in Y. However, this is memory
    intensive and slow for large `n`. For large `n` (or if Y is larger
    than 5MB), this method will dispatch to a loop-based approach
    that is faster and has a lower memory footprint.

    Args:
        Y: A `(batch_shape) x n x m`-dim tensor of outcomes.
        deduplicate: A boolean indicating whether to only return
            unique points on the pareto frontier.

    Returns:
        A `(batch_shape) x n`-dim boolean tensor indicating whether
        each point is non-dominated.
    """
    #n = Y.shape[-2]
    Y1 = np.expand_dims(Y, -3)
    Y2 = np.expand_dims(Y, -2)
    # eps from context
    dominates = (Y1 >= Y2 + eps).all(axis=-1) & (Y1 > Y2 + eps).any(axis=-1)
    nd_mask = ~(dominates.any(axis=-1))
    return nd_mask


def EGE_SR(T, K, D, environment):
    r"""
    Implements EGE SR
    :param T: Budget of the algorithm
    :param K: Number of arms
    :param D: Number of objectives
    :param environment: The multi-objective bandit environment
    :return: The estimated Pareto optimal arms
    """
    arms = np.arange(K)
    inf = (1 << 31) * 1.
    # implementing the SR scheme [cf Audibert et al 2010]
    log_K = 1 / 2 + np.sum(1 / np.arange(2, K + 1))
    n_ks = np.ceil([0, *(1 / log_K) * (T - K) / (K + 1 - np.arange(1, K))]).astype(int)
    total = np.zeros((K, D))
    active = np.ones(K, bool)
    means = np.empty((K, D), float)
    Nc = np.zeros(K, dtype=int)
    accepts = []
    rejects = []
    for r in range(1, K):
        num_pulls = n_ks[r] - n_ks[r - 1]
        if num_pulls > 0:
            for a in arms[active]:
                total[a] += environment.sample([a] * num_pulls).sum(0)
                Nc[a] += num_pulls
            means[active] = total[active] / Nc[active, None]
        active_idx = arms[active]
        Ik = np.eye(active.sum())
        index_of = {v: k for k, v in enumerate(active_idx)}
        g_i = lambda i: max(m(means[i], means[active]) - inf * Ik[index_of[i]])
        f_i = lambda i: min(min(M(means[i], means[active]) + inf * Ik[index_of[i]]),
                            min([max(M(means[j], means[i]), 0) + max(g_i(j), 0) for j in active_idx] + inf * Ik[
                                index_of[i]]))
        rk, dk, ak = [None] * 3
        indices = [-np.inf, -np.inf]
        dk = active_idx[np.argmax([g_i(i) for i in active_idx])]
        indices[0] = g_i(dk)
        ak = active_idx[np.argmax([f_i(i) for i in active_idx])]
        indices[1] = f_i(ak)
        # Implements the tie-breaking rule
        if indices[0] >= indices[1]:
            # remove an arm and classify as sub-optimal
            rejects += [dk]
            rk = dk
        else:
            #accept an arm as optimal
            accepts += [ak]
            rk = ak
        active[rk] = False
    accepts += [*arms[active]]
    return accepts


# Commented out original EGE-SH implementation for reference. Now using optimized EGE_SH below.
# def EGE_SH(T, K, D, environment):
#     r"""
#     Implements EGE-SH
#     :param T: Budget of the algorithm
#     :param K: Number of arms
#     :param D: Number of objectives
#     :param environment: The multi-objective bandit environment
#     :return: The estimated Pareto optimal arms
#     """
#     arms = np.arange(K)
#     inf = (1 << 31) * 1.
#     # defines a small constant to implement the tie Breaking rule
#     # the gaps of empirically sub-optimal arms are increased by c_0
#     c_0 = 1e-7
#     total = np.zeros((K, D))
#     active = np.ones(K, bool)
#     means = np.empty((K, D), float)
#     Nc = np.zeros(K, dtype=int)
#     accepts = []
#     rejects = []
#     # implementing the SH scheme
#     ceil_log2_K = math.ceil(np.log2(K))
#     for k in range(ceil_log2_K):
#         num_pulls = math.floor(T / (sum(active) * ceil_log2_K))
#         if num_pulls > 0:
#             for a in arms[active]:
#                 total[a] += environment.sample([a] * num_pulls).sum(0)
#                 Nc[a] += num_pulls
#             means[active] = total[active] / Nc[active, None]
#         active_idx = arms[active]
#         Sk_star_mask = is_non_dominated(means[active_idx])
#         Ik = np.eye(active.sum())
#         index_of = {v: k for k, v in enumerate(active_idx)}
#         g_i = lambda i: max(m(means[i], means[active]) - inf * Ik[index_of[i]])
#         f_i = lambda i: min(min(M(means[i], means[active]) + inf * Ik[index_of[i]]),
#                             min([max(M(means[j], means[i]), 0) + max(g_i(j), 0) for j in active_idx] + inf * Ik[
#                                 index_of[i]]))
#         # compute empirical gaps
#         Delta_i = lambda i: f_i(i) if (Sk_star_mask[index_of[i]]) else (c_0 + g_i(i))
#         num_arms_to_keep = math.ceil(len(active_idx) / 2)
#         # sort arms by their gaps
#         sorted_arms = np.argsort([-Delta_i(i) for i in active_idx])
#         arms_to_dismiss = active_idx[sorted_arms[:-num_arms_to_keep]]
#         # classify the dismissed arms
#         for a in arms_to_dismiss:
#             active[a] = False
#             if Sk_star_mask[index_of[a]]:
#                 accepts += [a]
#             else:
#                 rejects += [a]
#     assert sum(active) == 1, "There should not be more than one active arm remaining"
#     accepts += [*arms[active]]
#     return accepts


def EGE_SH(T, K, D, environment):
    """
    Vectorized and optimized implementation of EGE-SH.
    Preserves original algorithm semantics while reducing Python-level loops
    and repeated calls to environment.sample.
    """
    arms = np.arange(K)
    inf = (1 << 31) * 1.
    c_0 = 1e-7
    total = np.zeros((K, D))
    active = np.ones(K, bool)
    means = np.empty((K, D), float)
    Nc = np.zeros(K, dtype=int)
    accepts = []
    rejects = []

    ceil_log2_K = math.ceil(np.log2(K))
    for k in range(ceil_log2_K):
        s = active.sum()
        if s == 0:
            break

        num_pulls = math.floor(T / (s * ceil_log2_K))

        active_idx = arms[active]
        # vectorized sampling: pull each active arm num_pulls times in one call
        if num_pulls > 0:
            pulls = np.repeat(active_idx, num_pulls)
            rewards = environment.sample(pulls)  # shape (s * num_pulls, D)
            # reshape and sum per arm
            rewards = rewards.reshape(s, num_pulls, D).sum(axis=1)
            total[active_idx] += rewards
            Nc[active_idx] += num_pulls
            means[active] = total[active_idx] / Nc[active_idx, None]

        # compute empirical Pareto mask for the active subset
        means_active = means[active_idx]  # shape (s, D)
        Sk_star_mask = is_non_dominated(means_active)

        # compute pairwise M and m matrices via broadcasting (vectorized)
        # M_matrix[i, j] = max(means_active[i] - means_active[j]) across objectives
        # m_matrix[i, j] = min(means_active[j] - means_active[i]) across objectives
        M_matrix = M(means_active[:, None, :], means_active[None, :, :])
        m_matrix = m(means_active[:, None, :], means_active[None, :, :])

        # compute g vector: g[i] = max_j!=i m_matrix[i, j]
        # set diagonal to -inf to exclude self
        if s > 1:
            m_masked = m_matrix.copy()
            np.fill_diagonal(m_masked, -inf)
            g = np.max(m_masked, axis=1)
        else:
            # if only one active, g is -inf (no other arms to compare)
            g = np.array([-inf], dtype=float)

        # term1[i] = min_j!=i M_matrix[i, j]
        if s > 1:
            M_masked = M_matrix.copy()
            np.fill_diagonal(M_masked, inf)
            term1 = np.min(M_masked, axis=1)
        else:
            term1 = np.array([inf], dtype=float)

        # term2[i] = min_j!=i ( max(M[j,i],0) + max(g[j],0) )
        # Build A where A[j, i] = max(M[j,i],0) + max(g[j],0), then exclude diagonal
        if s > 1:
            A = np.maximum(M_matrix, 0.0) + np.maximum(g, 0.0)[:, None]  # shape (s, s) transposed to align columns as i
            np.fill_diagonal(A, inf)
            term2 = np.min(A, axis=0)
        else:
            term2 = np.array([inf], dtype=float)

        f = np.minimum(term1, term2)

        # Delta: if in Sk_star then f else c_0 + g
        Delta = np.where(Sk_star_mask, f, c_0 + g)

        # determine how many to keep
        num_arms_to_keep = math.ceil(s / 2)
        # sort arms by descending Delta (ties handled by numpy's stable argsort)
        sorted_idx_within_active = np.argsort(-Delta)
        # arms to dismiss are the lower-ranked ones
        dismiss_indices_within_active = sorted_idx_within_active[:-num_arms_to_keep]
        arms_to_dismiss = active_idx[dismiss_indices_within_active]

        # classify dismissed arms
        # build a mapping from arm -> index within active for mask lookup
        pos = np.full(K, -1, dtype=int)
        pos[active_idx] = np.arange(s)
        for a in arms_to_dismiss:
            active[a] = False
            # index to check Sk_star_mask
            idx = pos[a]
            if Sk_star_mask[idx]:
                accepts.append(a)
            else:
                rejects.append(a)

    # after rounds, exactly one active arm should remain
    assert sum(active) == 1, "There should not be more than one active arm remaining"
    accepts += [*arms[active]]
    return accepts
