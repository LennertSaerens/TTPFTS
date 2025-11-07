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


def EGE_SH(T, K, D, environment):
    r"""
    Implements EGE-SH
    :param T: Budget of the algorithm
    :param K: Number of arms
    :param D: Number of objectives
    :param environment: The multi-objective bandit environment
    :return: The estimated Pareto optimal arms
    """
    arms = np.arange(K)
    inf = (1 << 31) * 1.
    # defines a small constant to implement the tie Breaking rule
    # the gaps of empirically sub-optimal arms are increased by c_0
    c_0 = 1e-7
    total = np.zeros((K, D))
    active = np.ones(K, bool)
    means = np.empty((K, D), float)
    Nc = np.zeros(K, dtype=int)
    accepts = []
    rejects = []
    # implementing the SH scheme
    ceil_log2_K = math.ceil(np.log2(K))
    for k in range(ceil_log2_K):
        num_pulls = math.floor(T / (sum(active) * ceil_log2_K))
        if num_pulls > 0:
            for a in arms[active]:
                total[a] += environment.sample([a] * num_pulls).sum(0)
                Nc[a] += num_pulls
            means[active] = total[active] / Nc[active, None]
        active_idx = arms[active]
        Sk_star_mask = is_non_dominated(means[active_idx])
        Sk_star = active_idx[Sk_star_mask]
        Sk_star_comp = active_idx[~Sk_star_mask]
        Ik = np.eye(active.sum())
        index_of = {v: k for k, v in enumerate(active_idx)}
        g_i = lambda i: max(m(means[i], means[active]) - inf * Ik[index_of[i]])
        f_i = lambda i: min(min(M(means[i], means[active]) + inf * Ik[index_of[i]]),
                            min([max(M(means[j], means[i]), 0) + max(g_i(j), 0) for j in active_idx] + inf * Ik[
                                index_of[i]]))
        # compute empirical gaps
        Delta_i = lambda i: f_i(i) if (Sk_star_mask[index_of[i]]) else (c_0 + g_i(i))
        num_arms_to_keep = math.ceil(len(active_idx) / 2)
        # sort arms by their gaps
        sorted_arms = np.argsort([-Delta_i(i) for i in active_idx])
        arms_to_dismiss = active_idx[sorted_arms[:-num_arms_to_keep]]
        # classify the dismissed arms
        for a in arms_to_dismiss:
            active[a] = False
            if Sk_star_mask[index_of[a]]:
                accepts += [a]
            else:
                rejects += [a]
    assert sum(active) == 1, "There should not be more than one active arm remaining"
    accepts += [*arms[active]]
    return accepts
