#!/usr/bin/env python3
"""Correctness gate for the unknown-variance APE variant (APE-UV)."""

from __future__ import annotations

import traceback

import numpy as np
from scipy.stats import t as student_t

from bandits.APE import APEAnytimeBandit
from bandits.APE_UV import (
    APEUnknownVarianceBandit,
    StudentQuantileTable,
    simultaneous_alpha,
)
from momab_main import ENVIRONMENTS, _construct_environment, _make_bandit, _run_repetition_anytime
from runner.seeding import seed_rngs


FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def execute(name: str, fn) -> None:
    print(f"\n=== {name} ===")
    try:
        fn()
    except Exception as error:  # noqa: BLE001 - report every failure
        traceback.print_exc()
        check(f"{name} raised no exception", False, f"{type(error).__name__}: {error}")


def _environment(name: str):
    return _construct_environment(ENVIRONMENTS[name][0], "gaussian")


def test_known_bound_mode_reproduces_ape() -> None:
    for name, steps in (("EgeExp2", 2500), ("EgeExp7", 2500), ("EgeExp4", 1500), ("EgeExp8", 2500), ("EgeExp3", 1000)):
        environment = _environment(name)
        K, D = environment.num_arms, environment.num_objectives
        ape = APEAnytimeBandit(K, D, sigma=0.5, delta=0.1)
        variant = APEUnknownVarianceBandit(K, D, delta=0.1, num_warmup_pulls=1,
                                           width_mode="known_bound", known_sigma=0.5)
        seed_rngs(0, "ape-uv-equivalence", name)
        identical = True
        for _ in range(steps):
            arm = ape.choose_arm()
            if variant.choose_arm() != arm:
                identical = False
                break
            reward = environment.pull_arm(arm)
            ape.learn(arm, reward)
            variant.learn(arm, reward)
        check(f"{name}: known-bound APE-UV selects exactly APE's arms for {steps} pulls", identical)
        check(f"{name}: identical empirical means and recommendation",
              np.array_equal(ape.means, variant.means) and np.array_equal(ape.get_top_arms(), variant.get_top_arms()))


def test_incremental_matrices_are_exact() -> None:
    for name in ("EgeExp7", "EgeExp4", "EgeExp3"):
        environment = _environment(name)
        bandit = APEUnknownVarianceBandit(environment.num_arms, environment.num_objectives)
        seed_rngs(0, "ape-uv-incremental", name)
        exact = []
        for step in range(4 * environment.num_arms + 2500):
            arm = bandit.choose_arm()
            bandit.learn(arm, environment.pull_arm(arm))
            if bandit.index_matrices() is not None and step % 499 == 0:
                lower, upper = bandit.index_matrices()
                full_lower, full_upper = bandit.full_index_matrices()
                exact.append(np.array_equal(lower, full_lower) and np.array_equal(upper, full_upper))
        check(f"{name}: incrementally maintained L/U equal a full rebuild bitwise", bool(exact) and all(exact))


def test_no_noise_information() -> None:
    environment = _environment("EgeExp7")
    K, D = environment.num_arms, environment.num_objectives
    rejected = False
    try:
        APEUnknownVarianceBandit(K, D, known_sigma=0.5)
    except ValueError:
        rejected = True
    check("unknown-variance mode refuses a noise level", rejected)
    bandit = _make_bandit("APE_UV", environment, p=0.5)
    check("runner constructs unknown-variance APE-UV with a 4-pull warm-up",
          isinstance(bandit, APEUnknownVarianceBandit) and bandit.width_mode == "unknown_variance"
          and bandit.num_warmup_pulls == 4 and bandit.known_sigma is None)
    trajectories = []
    for stds in (environment.stds, np.full_like(environment.stds, 123.0), None):
        seed_rngs(0, "ape-uv-leak", 1)
        bandit = APEUnknownVarianceBandit(K, D)
        bandit.reset(stds)
        arms = []
        for _ in range(1500):
            arm = bandit.choose_arm()
            arms.append(arm)
            bandit.learn(arm, environment.pull_arm(arm))
        trajectories.append(arms)
    check("trajectory is invariant to the env_stds passed to reset",
          trajectories[0] == trajectories[1] == trajectories[2])


def test_quantile_table() -> None:
    K, D, delta = 22, 2, 0.1
    grown = StudentQuantileTable(K, D, delta)
    for n_max in (5, 1100, 5000, 700_000):
        grown.ensure(n_max)
    single = StudentQuantileTable(K, D, delta)
    single.ensure(700_000)
    n = np.array([2, 3, 4, 10, 100, 1000, 10_000, 100_000, 700_000])
    expected = np.array([student_t.isf(simultaneous_alpha(k, K, D, delta) / 2.0, k - 1) for k in n])
    check("table equals scalar scipy Student-t upper quantiles", np.array_equal(grown[n], expected))
    check("table is independent of how it was grown", np.array_equal(grown[np.arange(2, 700_001)], single[np.arange(2, 700_001)]))
    values = single[np.arange(2, 700_001)]
    radius_factor = values / np.sqrt(np.arange(2, 700_001))
    check("quantiles are finite and positive, and the radius factor q_n/sqrt(n) decreases with n",
          bool(np.all(np.isfinite(values)) and np.all(values > 0) and np.all(np.diff(radius_factor) < 0)))
    tiny = simultaneous_alpha(2_000_000, 200, 2, delta) / 2.0
    check("isf stays finite where 1 - alpha/2 rounds to 1 and ppf would return inf",
          1.0 - tiny == 1.0 and np.isinf(student_t.ppf(1.0 - tiny, 1_999_999))
          and np.isfinite(student_t.isf(tiny, 1_999_999)))


def test_warmup_and_variance_estimates() -> None:
    environment = _environment("EgeExp2")
    K, D = environment.num_arms, environment.num_objectives
    bandit = APEUnknownVarianceBandit(K, D)
    seed_rngs(0, "ape-uv-warmup")
    observed: dict[int, list[np.ndarray]] = {arm: [] for arm in range(K)}
    arms = []
    for _ in range(4 * K + 600):
        arm = bandit.choose_arm()
        reward = environment.pull_arm(arm)
        observed[arm].append(reward)
        arms.append(arm)
        bandit.learn(arm, reward)
    check("first 4K pulls are a round-robin warm-up", arms[: 4 * K] == list(range(K)) * 4)
    variance = np.array([np.var(np.array(observed[arm]), axis=0, ddof=1) for arm in range(K)])
    means = np.array([np.mean(np.array(observed[arm]), axis=0) for arm in range(K)])
    check("Welford variances equal batch sample variances",
          np.allclose(bandit.m2 / (bandit.pulls[:, None] - 1), variance, rtol=1e-10, atol=1e-14))
    check("running means equal batch means", np.allclose(bandit.means, means, rtol=1e-12, atol=1e-14))


def test_simultaneous_coverage() -> None:
    rng = np.random.default_rng(20261006)
    K, D, delta, horizon, runs = 3, 2, 0.1, 400, 3000
    true_means = np.array([[0.2, 0.7], [0.5, 0.5], [0.9, 0.1]])
    sigmas = np.array([[0.1, 0.5], [0.3, 0.3], [0.5, 0.2]])
    table = StudentQuantileTable(K, D, delta)
    n = np.arange(1, horizon + 1, dtype=float)
    failures = 0
    for _ in range(runs):
        samples = true_means[:, :, None] + sigmas[:, :, None] * rng.standard_normal((K, D, horizon))
        csum, csq = np.cumsum(samples, axis=2), np.cumsum(samples**2, axis=2)
        mean = csum / n
        var = np.maximum(csq - n * mean**2, 0.0)[:, :, 1:] / (n[1:] - 1)
        radius = table[np.arange(2, horizon + 1)] * np.sqrt(var) / np.sqrt(n[1:])
        failures += bool(np.any(np.abs(mean[:, :, 1:] - true_means[:, :, None]) > radius))
    rate = failures / runs
    print(f"    simultaneous miss rate over all arms, objectives and n<= {horizon}: {rate:.4f}")
    check("Monte Carlo simultaneous miss rate is at most delta", rate <= delta)


def test_fixed_budget_continuation() -> None:
    # Dominated arms always keep h <= 0, so only an all-Pareto instance can be
    # fully certified and reach the balanced-sampling extension.
    class AllOptimal:
        num_arms, num_objectives = 3, 2
        stds = np.full((3, 2), 0.01)
        _means = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])

        def pull_arm(self, arm):
            return np.random.normal(self._means[arm], 0.01)

    environment = AllOptimal()
    bandit = APEUnknownVarianceBandit(3, 2)
    seed_rngs(0, "ape-uv-separated")
    continued = []
    for _ in range(3000):
        arm = bandit.choose_arm()
        if bandit.last_decision.fixed_budget_continuation:
            continued.append(arm == int(np.argmin(bandit.pulls)))
        bandit.learn(arm, environment.pull_arm(arm))
    check("certified instances fall back to balanced sampling", bool(continued) and all(continued))
    check("recommendation is the empirical Pareto set", list(bandit.get_top_arms()) == [0, 1, 2])


def test_runner_determinism() -> None:
    environment = _environment("EgeExp2")
    grid = [0, 100, 500, 2544]
    rows = [
        _run_repetition_anytime(environment, "EgeExp2", "gaussian", "APE_UV",
                                _make_bandit("APE_UV", environment, p=0.5), 2544, grid, 0, 3)
        for _ in range(2)
    ]
    check("shared anytime runner is deterministic for APE_UV", rows[0] == rows[1])
    check("rows carry no posterior model or rho",
          all(row["posterior_model"] == "" and row["rho"] == "" for row in rows[0]))


if __name__ == "__main__":
    execute("Known-bound mode reproduces APE bitwise", test_known_bound_mode_reproduces_ape)
    execute("Incremental index matrices are exact", test_incremental_matrices_are_exact)
    execute("No noise information enters APE-UV", test_no_noise_information)
    execute("Student-t quantile table", test_quantile_table)
    execute("Warm-up and online variance estimates", test_warmup_and_variance_estimates)
    execute("Simultaneous interval coverage", test_simultaneous_coverage)
    execute("Fixed-budget continuation", test_fixed_budget_continuation)
    execute("Runner determinism", test_runner_determinism)
    print("\n" + "=" * 80)
    if FAILURES:
        print("FAILED:", ", ".join(FAILURES))
        raise SystemExit(1)
    print("ALL TESTS PASSED")
