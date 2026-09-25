#!/usr/bin/env python3
"""Standalone deterministic Gate 1 for the fixed-budget APE port.

Run with ``python test_ape_equivalence.py``.  It performs two independent
checks: formula parity against the unmodified C++ helper implementation, then
full shared-reward-tape traces against ``ape_reference_oracle.py``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ape_reference_oracle import ReferenceFaithfulAPEOracle, betaij, cg, min_dom, min_non_dom, pareto_mask
from bandits.APE import APEAnytimeBandit, confidence_constant

ROOT = Path(__file__).resolve().parent
CPP_ROOT = ROOT / "8606_Adaptive_Algorithms_for_R_Supplementary Material/bandit_lib_c++"


def assert_close(actual, expected, label: str) -> None:
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, err_msg=label)


def _cpp_helper_parity() -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("Gate 1 requires a C++ compiler to check the supplied helper implementation")
    source = r'''
#include <iomanip>
#include <iostream>
#include "utils.hpp"
int main() {
  std::vector<std::vector<double>> means{{1., 1.}, {1., 1.}, {.5, 1.5}};
  std::vector<double> zeros(3, 0.);
  std::cout << std::setprecision(17)
            << Cg(.1) << " " << betaij(3, 7, Cg(.1), 1.) << " "
            << minimum_quantity_dom(means[0], means[2], 0.) << " "
            << minimum_quantity_non_dom(means[0], means[2], 0.) << "\n";
  auto mask = pareto_optimal_arms_mask(means, zeros, 0.);
  for (bool value : mask) std::cout << value << " ";
  std::cout << "\n";
  std::vector<size_t> st{0, 1, 2};
  std::vector<size_t> st_comp{};
  std::vector<size_t> opt{};
  std::vector<std::vector<double>> beta(3, std::vector<double>(3));
  std::vector<size_t> counts{3, 7, 5};
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      beta[i][j] = betaij(counts[i], counts[j], Cg(.1), 1.);
  std::cout << get_h(0, means, beta, 0.) << " "
            << get_g(2, means, beta, opt, 0.) << " "
            << get_z1t(means, st, beta, 0.) << " "
            << get_z2t(means, st_comp, beta, 0.) << " "
            << get_z_tilde(means, beta, opt, 0., 0.) << "\n";
}
'''
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        harness = directory / "ape_helper_parity.cxx"
        executable = directory / "ape_helper_parity"
        harness.write_text(source)
        subprocess.run([
            compiler, "-std=c++17", "-I", str(CPP_ROOT / "src/cpp"), str(harness),
            str(CPP_ROOT / "src/cpp/utils.cxx"), "-o", str(executable),
        ], check=True, capture_output=True, text=True)
        output = subprocess.run([str(executable)], check=True, capture_output=True, text=True).stdout.splitlines()
    values = np.fromstring(output[0], sep=" ")
    expected = np.array([cg(.1), betaij(3, 7, cg(.1), 1.), min_dom(np.array([1., 1.]), np.array([.5, 1.5]), 0.), min_non_dom(np.array([1., 1.]), np.array([.5, 1.5]), 0.)])
    assert_close(values, expected, "C++ helper formula parity")
    means = np.array([[1., 1.], [1., 1.], [.5, 1.5]])
    np.testing.assert_array_equal(np.fromstring(output[1], sep=" ").astype(bool), pareto_mask(means))
    oracle = ReferenceFaithfulAPEOracle(3, 2, delta=.1, sigma=1.)
    oracle.means[:] = means
    oracle.counts[:] = [3, 7, 5]
    native_helpers = np.fromstring(output[2], sep=" ")
    oracle_beta = oracle._beta()
    expected_helpers = np.array([
        oracle._h(0, oracle_beta),
        oracle._g(2, oracle_beta, np.empty(0, dtype=int)),
        min(oracle._h(i, oracle_beta) for i in range(3)),
        1e12,
        min(max(oracle._h(i, oracle_beta), oracle._g(i, oracle_beta, np.empty(0, dtype=int))) for i in range(3)),
    ])
    assert_close(native_helpers, expected_helpers, "C++ stopping helper parity")
    assert_close(confidence_constant(.1), values[0], "production Cg parity")


def _assert_decision_equal(actual, expected, scenario: str, step: int) -> None:
    assert actual.selected_arm == expected.selected_arm, (scenario, step, "selected arm")
    assert actual.b_t == expected.b_t, (scenario, step, "b_t")
    assert actual.c_t == expected.c_t, (scenario, step, "c_t")
    assert actual.reference_should_stop == expected.reference_should_stop, (scenario, step, "stop")
    assert actual.fixed_budget_continuation == expected.fixed_budget_continuation, (scenario, step, "continuation")
    for field in ("counts", "means", "pareto_arms", "opt_arms", "opt_comp_arms"):
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), rtol=0, atol=0, err_msg=f"{scenario} step={step}: {field}")
    if actual.beta is None:
        assert expected.beta is None
        return
    assert_close(actual.beta, expected.beta, f"{scenario} step={step}: beta")
    for field in ("z1_t", "z2_t", "z4_t"):
        assert_close(getattr(actual, field), getattr(expected, field), f"{scenario} step={step}: {field}")
    assert actual.z3_t == expected.z3_t


def _trace(scenario: str, tapes: np.ndarray, steps: int, *, expect_reference_sampling: bool = True, **kwargs) -> None:
    candidate = APEAnytimeBandit(tapes.shape[0], tapes.shape[2], **kwargs)
    oracle = ReferenceFaithfulAPEOracle(tapes.shape[0], tapes.shape[2], **kwargs)
    candidate.reset(np.ones(tapes.shape[2]))
    tape_offsets = np.zeros(tapes.shape[0], dtype=int)
    saw_reference_sampling = False
    for step in range(steps):
        arm = candidate.choose_arm()
        oracle_arm = oracle.choose_arm()
        assert arm == oracle_arm, (scenario, step, arm, oracle_arm)
        _assert_decision_equal(candidate.last_decision, oracle.last_decision, scenario, step)
        if candidate.last_decision.b_t is not None:
            saw_reference_sampling = True
        reward = tapes[arm, tape_offsets[arm]]
        tape_offsets[arm] += 1
        candidate.learn(arm, reward)
        oracle.learn(arm, reward)
        np.testing.assert_allclose(candidate.means, oracle.means, rtol=0, atol=0, err_msg=f"{scenario} step={step}: post-reward means")
    if expect_reference_sampling:
        assert saw_reference_sampling, f"{scenario}: trace never exercised the reference sampling branch"


def _shared_tape_traces() -> None:
    rng = np.random.default_rng(731)
    _trace("mixed-pareto", rng.normal(size=(4, 24, 2)), 20, delta=.1, sigma=1., epsilon_1=.15, epsilon_2=.1)
    _trace("exact-ties", np.broadcast_to(np.array([[[1., 1.]]]), (3, 20, 2)).copy(), 14, delta=.1, sigma=1.)
    # The first two pulls certify arm 0 under the C++ stopping rule. The
    # reference b_t/c_t selector remains defined, so fixed-budget execution
    # keeps using it after recording the diagnostic stop.
    separated = np.array([[[10., 10.]] * 12, [[0., 0.]] * 12], dtype=float)
    candidate = APEAnytimeBandit(2, 2, delta=.1, sigma=1.)
    candidate.reset(np.ones(2))
    for expected_arm in (0, 1):
        assert candidate.choose_arm() == expected_arm
        candidate.learn(expected_arm, separated[expected_arm, 0])
    candidate.choose_arm()
    assert candidate.last_decision.reference_should_stop
    assert candidate.last_decision.fixed_budget_continuation
    assert candidate.last_decision.b_t is not None
    assert candidate.last_decision.c_t is not None
    assert candidate.last_decision.selected_arm == 1
    _trace("confidence-stop-continuation", separated, 10, delta=.1, sigma=1.)

    # With every arm certified Pareto-optimal, the original get_bt selector has
    # an empty candidate set and therefore no post-stop definition. Balanced
    # sampling is tested and labelled as the fixed-budget-only extension.
    tradeoff = np.array([[[10., 0.]] * 8, [[0., 10.]] * 8], dtype=float)
    candidate = APEAnytimeBandit(2, 2, delta=.1, sigma=1.)
    candidate.reset(np.ones(2))
    for expected_arm in (0, 1):
        assert candidate.choose_arm() == expected_arm
        candidate.learn(expected_arm, tradeoff[expected_arm, 0])
    candidate.choose_arm()
    assert candidate.last_decision.reference_should_stop
    assert candidate.last_decision.fixed_budget_continuation
    assert not candidate.last_decision.opt_comp_arms.size
    assert candidate.last_decision.b_t is None
    assert candidate.last_decision.c_t is None
    _trace("empty-candidate-continuation", tradeoff, 8, delta=.1, sigma=1.,
           expect_reference_sampling=False)


def main() -> None:
    _cpp_helper_parity()
    _shared_tape_traces()
    print("APE Gate 1 passed: C++ helper formulas and all deterministic shared-tape traces agree.")


if __name__ == "__main__":
    main()
