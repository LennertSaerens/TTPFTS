"""Regression tests for the paired heteroscedastic candidate ensemble."""

from __future__ import annotations

import copy
import sys
import traceback

import numpy as np

from environments.distributions import make_distributions
from experiments.heteroscedastic.config import load_candidate_config
from experiments.heteroscedastic.instances import (
    FrozenHeteroscedasticEnvironment,
    generate_candidate_pair,
    generate_candidates,
    load_frozen_candidates,
)


FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def run(name: str, fn) -> None:
    print(f"\n--- {name} ---")
    try:
        fn()
    except Exception:
        FAILURES.append(name)
        print(f"[FAIL] {name} raised:")
        traceback.print_exc()


def _small_config() -> dict:
    config = copy.deepcopy(load_candidate_config())
    config["candidate_instances"] = 3
    return config


def test_gaussian_std_shapes() -> None:
    means = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    matrix = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    for name, stds, expected in (
        ("scalar", 0.3, np.full((2, 2), 0.3)),
        ("objective vector", np.array([0.2, 0.4]), np.array([[0.2, 0.4], [0.2, 0.4]])),
        ("matrix", matrix, matrix),
    ):
        distributions = make_distributions(means, gaussian_std=stds)
        observed = np.asarray([distribution.std for distribution in distributions])
        check(f"Gaussian factory accepts {name} standard deviations", np.array_equal(observed, expected))
    try:
        make_distributions(means, gaussian_std=np.ones((3, 2)))
    except ValueError:
        check("Gaussian factory rejects invalid std shape", True)
    else:
        check("Gaussian factory rejects invalid std shape", False)
    try:
        make_distributions(means, gaussian_std=np.array([[0.0, 0.1], [0.1, 0.1]]))
    except ValueError:
        check("Gaussian factory rejects non-positive std", True)
    else:
        check("Gaussian factory rejects non-positive std", False)


def test_pair_is_deterministic_and_matched() -> None:
    config = _small_config()
    heterogeneous, control = generate_candidate_pair(0, config)
    repeat_heterogeneous, repeat_control = generate_candidate_pair(0, config)
    check("candidate generation is deterministic", np.array_equal(heterogeneous.means, repeat_heterogeneous.means)
          and np.array_equal(heterogeneous.stds, repeat_heterogeneous.stds))
    check("paired conditions have identical means", np.array_equal(heterogeneous.means, control.means))
    check("paired conditions have identical Pareto set", np.array_equal(
        heterogeneous.pareto_indices, control.pareto_indices
    ))
    check("paired conditions have identical hardness and budget",
          heterogeneous.hardness.hardness == control.hardness.hardness
          and heterogeneous.budget == control.budget)
    check("heterogeneous stds use only configured discrete values",
          set(np.unique(heterogeneous.stds)).issubset({0.1, 0.2, 0.3, 0.4, 0.5}))
    check("control std is homogeneous", np.unique(control.stds).size == 1)
    check("control RMS exactly matches heterogeneous RMS",
          np.array_equal(np.sqrt(np.mean(heterogeneous.stds**2)), control.stds[0, 0]))
    check("candidate grid is exactly zero plus ten positive points",
          len(heterogeneous.grid) == 11 and heterogeneous.grid[0] == 0
          and heterogeneous.grid[-1] == heterogeneous.budget)
    check("candidate hardness is finite and positive", np.isfinite(heterogeneous.hardness.hardness)
          and heterogeneous.hardness.hardness > 0)
    check("candidate direct and lemma gaps agree", heterogeneous.hardness.gaps_match)

    environment = FrozenHeteroscedasticEnvironment(heterogeneous.means, heterogeneous.stds)
    before_means, before_stds = environment.arms.copy(), environment.stds.copy()
    environment.reset()
    check("frozen environment reset does not redraw means or stds",
          np.array_equal(before_means, environment.arms) and np.array_equal(before_stds, environment.stds))


def test_candidate_prefix_invariance() -> None:
    config = _small_config()
    small = generate_candidates(config)
    expanded = copy.deepcopy(config)
    expanded["candidate_instances"] = 5
    larger = generate_candidates(expanded)
    for index in range(3):
        small_pair = [candidate for candidate in small if candidate.instance_id == index]
        large_pair = [candidate for candidate in larger if candidate.instance_id == index]
        check(
            f"candidate {index} is invariant when later instances are added",
            all(
                np.array_equal(left.means, right.means) and np.array_equal(left.stds, right.stds)
                and left.budget == right.budget
                for left, right in zip(small_pair, large_pair)
            ),
        )


def test_frozen_candidate_reload() -> None:
    candidates = load_frozen_candidates()
    ids = {candidate.instance_id for candidate in candidates}
    check("frozen artifact loads all 100 paired candidates", len(candidates) == 200 and len(ids) == 100)
    check(
        "frozen artifact retains both conditions for every candidate",
        all(
            {candidate.condition for candidate in candidates if candidate.instance_id == instance_id}
            == {"heterogeneous", "homoscedastic_rms"}
            for instance_id in ids
        ),
    )


if __name__ == "__main__":
    run("Gaussian matrix standard-deviation support", test_gaussian_std_shapes)
    run("Deterministic paired candidate generation", test_pair_is_deterministic_and_matched)
    run("Candidate prefix invariance", test_candidate_prefix_invariance)
    run("Frozen candidate artifact reload", test_frozen_candidate_reload)
    print("\n" + "=" * 80)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("ALL TESTS PASSED")
