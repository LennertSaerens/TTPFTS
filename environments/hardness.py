"""
MOMAB Pareto-identification hardness (problem complexity) computation.

Implements the direct definitions of the per-arm suboptimality/optimality
gaps Delta_a and the problem complexity H(nu), together with the equivalent,
Pareto-set-independent formulation from Lemma 2.1.2, both taken from:

    "/Users/lennertsaerens/Desktop/PhD/TTPFTS/Bounds Workspace clean/
     Ground Truth and Knowledge/TTPFTS MOMAB PSI.md"

Definitions (mu in R^{K x D}, all objectives maximized)
---------------------------------------------------------------------------
m(i, j) := min_c [mu_j^c - mu_i^c]
M(i, j) := max_c [mu_i^c - mu_j^c]

Direct definition (case split on membership in the Pareto set S*):
  - i not in S*:  Delta_i = max_{j in S*} m(i, j)
  - i in S*:      Delta_i = min(delta_i^+, delta_i^-), where
        delta_i^+ = min_{j in S*, j != i} min[M(i, j), M(j, i)]
        delta_i^- = min_{j not in S*} [ (M(j, i))_+ + Delta_j ]
    (delta_i^- requires the suboptimal arms' gaps, which only depend on the
    direct suboptimal-arm formula above, so there is no circularity.)

Lemma 2.1.2 (equivalent, Pareto-set-independent, single formula for all k):
    Delta_k^star  = max_{j in [K]} m(k, j)                      (>= 0, since j=k gives 0)
    delta_k^star  = min_{j != k} [ M(k, j) ^ ( (M(j, k))_+ + (Delta_j^star)_+ ) ]
    Delta_k = max(Delta_k^star, delta_k^star)

Problem complexity: H(nu) = sum_a 1 / Delta_a^2.

This module cross-checks both formulations arm-by-arm and explicitly
reports (rather than silently regularizing) any arm with Delta_a == 0,
which makes H(nu) infinite (a violated hardness precondition, see the
"Learning model" section of the source document: "we assume every arm is
separated from the Pareto decision boundary by a strictly positive gap").
"""

from dataclasses import dataclass

import numpy as np
from paretoset import paretoset


def compute_pareto_mask(arms: np.ndarray) -> np.ndarray:
    """Independent Pareto-mask computation (max-max, ties kept as optimal).

    This intentionally re-implements the O(K^2 D) pairwise-domination check
    directly (rather than reusing BaseEnvironment._compute_pareto_indices /
    the `paretoset` library) so that it constitutes an *independent* check
    of the environment's cached Pareto set, not merely a re-import of the
    same code path.
    """
    K = arms.shape[0]
    dominated = np.zeros(K, dtype=bool)
    for i in range(K):
        if dominated[i]:
            continue
        for j in range(K):
            if i == j:
                continue
            # j dominates i iff j >= i in every coordinate and j > i in at least one.
            if np.all(arms[j] >= arms[i]) and np.any(arms[j] > arms[i]):
                dominated[i] = True
                break
    return ~dominated


def _m(arms: np.ndarray, i: int, j: int) -> float:
    """m(i, j) = min_c [mu_j^c - mu_i^c]."""
    return float(np.min(arms[j] - arms[i]))


def _M(arms: np.ndarray, i: int, j: int) -> float:
    """M(i, j) = max_c [mu_i^c - mu_j^c]."""
    return float(np.max(arms[i] - arms[j]))


def direct_gaps(arms: np.ndarray, pareto_mask: np.ndarray) -> np.ndarray:
    """Per-arm gaps Delta_a via the direct, Pareto-set-dependent definition."""
    K = arms.shape[0]
    pareto_indices = np.where(pareto_mask)[0]
    suboptimal_indices = np.where(~pareto_mask)[0]

    gaps = np.full(K, np.nan, dtype=np.float64)

    # Suboptimal arms first: Delta_i = max_{j in S*} m(i, j). Well-defined
    # independently of the optimal arms' gaps.
    for i in suboptimal_indices:
        gaps[i] = max(_m(arms, i, j) for j in pareto_indices)

    # Optimal arms: Delta_i = min(delta_i^+, delta_i^-); delta_i^- only needs
    # the (already computed) suboptimal Delta_j.
    for i in pareto_indices:
        if len(pareto_indices) > 1:
            delta_plus = min(
                min(_M(arms, i, j), _M(arms, j, i))
                for j in pareto_indices if j != i
            )
        else:
            delta_plus = np.inf
        if len(suboptimal_indices) > 0:
            delta_minus = min(
                max(_M(arms, j, i), 0.0) + gaps[j]
                for j in suboptimal_indices
            )
        else:
            delta_minus = np.inf
        gaps[i] = min(delta_plus, delta_minus)

    return gaps


def lemma_gaps(arms: np.ndarray) -> np.ndarray:
    """Per-arm gaps Delta_a via the unified Lemma 2.1.2 formula (no Pareto-set case split)."""
    K = arms.shape[0]
    delta_star = np.array([
        max(_m(arms, k, j) for j in range(K))  # includes j=k -> m(k,k)=0
        for k in range(K)
    ])
    gaps = np.empty(K, dtype=np.float64)
    for k in range(K):
        if K > 1:
            small_delta = min(
                min(_M(arms, k, j), max(_M(arms, j, k), 0.0) + max(delta_star[j], 0.0))
                for j in range(K) if j != k
            )
        else:
            small_delta = np.inf
        gaps[k] = max(delta_star[k], small_delta)
    return gaps


@dataclass
class HardnessResult:
    env_name: str
    num_arms: int
    pareto_indices_direct: np.ndarray
    pareto_indices_independent: np.ndarray
    pareto_sets_match: bool
    gaps_direct: np.ndarray
    gaps_lemma: np.ndarray
    max_abs_gap_mismatch: float
    gaps_match: bool
    zero_gap_arms: np.ndarray
    hardness: float  # np.inf if any zero-gap arm


def compute_hardness(env_name: str, arms: np.ndarray,
                     reference_pareto_indices: np.ndarray,
                     gap_atol: float = 1e-9) -> HardnessResult:
    """Full armwise cross-check + H(nu) for one environment instance.

    :param reference_pareto_indices: the Pareto indices the environment
        itself reports (e.g. BaseEnvironment.pareto_indices), used only to
        cross-check against this module's independent Pareto computation.
    """
    independent_mask = compute_pareto_mask(arms)
    independent_indices = np.where(independent_mask)[0]
    reference_set = set(np.asarray(reference_pareto_indices).tolist())
    independent_set = set(independent_indices.tolist())
    pareto_match = reference_set == independent_set

    gaps_direct = direct_gaps(arms, independent_mask)
    gaps_lemma = lemma_gaps(arms)

    mismatch = np.abs(gaps_direct - gaps_lemma)
    max_mismatch = float(np.max(mismatch))
    gaps_match = bool(np.all(mismatch <= gap_atol))

    zero_gap_arms = np.where(np.isclose(gaps_direct, 0.0, atol=gap_atol))[0]
    if len(zero_gap_arms) > 0:
        hardness = float("inf")
    else:
        hardness = float(np.sum(1.0 / gaps_direct ** 2))

    return HardnessResult(
        env_name=env_name,
        num_arms=arms.shape[0],
        pareto_indices_direct=np.where(independent_mask)[0],
        pareto_indices_independent=independent_indices,
        pareto_sets_match=pareto_match,
        gaps_direct=gaps_direct,
        gaps_lemma=gaps_lemma,
        max_abs_gap_mismatch=max_mismatch,
        gaps_match=gaps_match,
        zero_gap_arms=zero_gap_arms,
        hardness=hardness,
    )


def objective_value_ties(arms: np.ndarray) -> list[tuple[int, int, int]]:
    """Return every ``(objective, arm_i, arm_j)`` triple with an exact tie.

    An exact tie between two arms in a single objective means one arm can only
    ever be *weakly* dominated along that coordinate. Whenever the tying arm is
    the sole witness of a suboptimal arm's improvement, the corresponding gap
    Delta collapses to exactly 0 and H(nu) diverges (see ``compute_hardness``).
    Ties are therefore reported as a structural degeneracy of the instance,
    independently of whether they happen to zero a gap in this particular
    configuration.
    """
    arms = np.asarray(arms, dtype=np.float64)
    ties: list[tuple[int, int, int]] = []
    for d in range(arms.shape[1]):
        column = arms[:, d]
        order = np.argsort(column, kind="mergesort")
        sorted_column = column[order]
        for pos in np.where(np.diff(sorted_column) == 0.0)[0]:
            ties.append((d, int(order[pos]), int(order[pos + 1])))
    return ties


def minimum_objective_separation(arms: np.ndarray) -> float:
    """Smallest gap between two distinct arms' values within one objective."""
    arms = np.asarray(arms, dtype=np.float64)
    separations = []
    for d in range(arms.shape[1]):
        diffs = np.diff(np.sort(arms[:, d]))
        separations.append(float(diffs.min()) if diffs.size else np.inf)
    return min(separations)


def assert_non_degenerate(env_name: str, arms: np.ndarray,
                          reference_pareto_indices: np.ndarray,
                          gap_atol: float = 1e-9,
                          require_tie_free: bool = False) -> HardnessResult:
    """Assert an instance has finite hardness and a consistent Pareto set.

    Exact objective-value ties are only *fatal* when they actually zero a gap;
    an instance may tie two arms in one objective and still separate every arm
    from the Pareto decision boundary (EgeExp7's lattice construction does).
    Pass ``require_tie_free=True`` for instances whose design forbids ties
    outright. Raises ``ValueError`` listing every offending arm pair, so a
    degenerate instance can never silently reach an experiment.
    """
    ties = objective_value_ties(arms)
    result = compute_hardness(env_name, arms, reference_pareto_indices, gap_atol=gap_atol)
    problems = []
    if ties and require_tie_free:
        problems.append(f"exact objective-value ties (objective, arm_i, arm_j): {ties}")
    if len(result.zero_gap_arms) > 0:
        if ties:
            problems.append(f"exact objective-value ties (objective, arm_i, arm_j): {ties}")
        problems.append(f"zero-gap arms {result.zero_gap_arms.tolist()} => H(nu) = inf")
    if not result.pareto_sets_match:
        problems.append("environment Pareto set disagrees with the independent computation")
    if not result.gaps_match:
        problems.append(
            f"direct vs. Lemma 2.1.2 gaps disagree (max |diff| = {result.max_abs_gap_mismatch})")
    if problems:
        raise ValueError(f"{env_name} is a degenerate instance: " + "; ".join(problems))
    return result
