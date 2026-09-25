"""Tidy, self-describing result schema.

Every row fully describes one (environment, algorithm, repetition,
arm_pull) observation, and stores enough information (``recommendation`` and
``true_pareto``) that the three metric columns are recomputable from the row
itself, independent of any live environment/bandit state.
"""

from __future__ import annotations

import json
from typing import Iterable, Sequence

SCHEMA_COLUMNS = [
    "environment",
    "instance_hash",
    "algorithm",
    "posterior_model",
    "repetition",
    "arm_pull",
    "max_budget",
    "evaluation_grid",
    "rho",
    "master_seed",
    "bernoulli",
    "jaccard",
    "misidentification",
    "recommendation",
    "true_pareto",
]


def encode_int_set(values: Iterable[int]) -> str:
    """Stably JSON-encode a set/sequence of ints as a sorted, deduplicated list.

    Sorting + deduplicating before encoding guarantees byte-identical output
    for the same underlying set regardless of insertion/iteration order
    (Python ``set`` iteration order is not guaranteed stable across
    processes/seeds), which matters for exact-match comparisons in tests and
    for reproducible merged files.
    """
    return json.dumps(sorted({int(v) for v in values}), separators=(",", ":"))


def decode_int_set(encoded: str) -> set[int]:
    return set(json.loads(encoded))


def build_row(*, environment: str, instance_hash: str, algorithm: str, posterior_model: str,
             repetition: int, arm_pull: int, max_budget: int, evaluation_grid: Sequence[int],
             rho, master_seed, recommendation: Iterable[int], true_pareto: Iterable[int],
             bernoulli=None, jaccard=None, misidentification=None) -> dict:
    """Build one tidy result row, computing metrics from recommendation/true_pareto
    when they are not supplied directly (they always agree with the environment's
    own metric methods; see ``recompute_metrics``)."""
    rec_set = {int(v) for v in recommendation}
    pareto_set = {int(v) for v in true_pareto}
    if bernoulli is None or jaccard is None or misidentification is None:
        computed = recompute_metrics(rec_set, pareto_set, num_arms=_infer_num_arms(rec_set, pareto_set, max_budget))
        bernoulli = computed["bernoulli"] if bernoulli is None else bernoulli
        jaccard = computed["jaccard"] if jaccard is None else jaccard
        misidentification = computed["misidentification"] if misidentification is None else misidentification
    return {
        "environment": environment,
        "instance_hash": instance_hash,
        "algorithm": algorithm,
        "posterior_model": posterior_model,
        "repetition": int(repetition),
        "arm_pull": int(arm_pull),
        "max_budget": int(max_budget),
        "evaluation_grid": encode_int_set(evaluation_grid),
        "rho": "" if rho is None else rho,
        "master_seed": "" if master_seed is None else int(master_seed),
        "bernoulli": bernoulli,
        "jaccard": jaccard,
        "misidentification": misidentification,
        "recommendation": encode_int_set(rec_set),
        "true_pareto": encode_int_set(pareto_set),
    }


def _infer_num_arms(rec_set, pareto_set, max_budget):
    # Only used as a last resort inside build_row when num_arms isn't known
    # directly; real call sites always pass metrics explicitly instead.
    return max(max(rec_set, default=-1), max(pareto_set, default=-1)) + 1


def recompute_metrics(recommendation: Iterable[int], true_pareto: Iterable[int], num_arms: int) -> dict:
    """Recompute the three metrics purely from stored sets, mirroring
    ``BaseEnvironment.bernoulli_metric``/``jaccard_metric``/``mis_id_metric``
    exactly (duplicated here rather than imported so results stay
    recomputable from a CSV row alone, without reconstructing any live
    environment object)."""
    rec_set = {int(v) for v in recommendation}
    pareto_set = {int(v) for v in true_pareto}
    bernoulli = int(rec_set == pareto_set)
    union = rec_set | pareto_set
    jaccard = (len(rec_set & pareto_set) / len(union)) if union else 1.0
    symmetric_diff = rec_set.symmetric_difference(pareto_set)
    misidentification = len(symmetric_diff) / num_arms if num_arms else 0.0
    return {"bernoulli": bernoulli, "jaccard": jaccard, "misidentification": misidentification}


def validate_row_metrics(row: dict, num_arms: int, atol: float = 1e-9) -> list[str]:
    """Return a list of human-readable mismatches between a row's stored
    metrics and metrics recomputed from its own recommendation/true_pareto."""
    rec_set = decode_int_set(row["recommendation"])
    pareto_set = decode_int_set(row["true_pareto"])
    recomputed = recompute_metrics(rec_set, pareto_set, num_arms)
    problems = []
    for key in ("bernoulli", "jaccard", "misidentification"):
        stored = float(row[key])
        expected = float(recomputed[key])
        if abs(stored - expected) > atol:
            problems.append(f"{key}: stored={stored} recomputed={expected}")
    return problems
