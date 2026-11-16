#!/usr/bin/env python3
"""Unit and regression tests for Variant 1 (directional BC) and Variant 2 (dominance risk)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.uq.analytic_checks import run_checks
from experiments.uq.metrics import (
    uq_directional_bc_from_moments,
    uq_dominance_risk_from_moments,
    uq_mc_disagreement_from_moments,
)
from uncertainty_quantification import (
    uncertainty_quantification_v1,
    uncertainty_quantification_v2,
)


def test_dataframe_wrappers_and_edge_cases() -> None:
    s = 0.05
    df = pd.DataFrame({
        "arm": [0, 1],
        "means": [[10.0 * s, 0.01 * s], [0.0, 0.0]],
        "stds": [[s, s], [s, s]],
    })
    v1 = uncertainty_quantification_v1(df)
    v2 = uncertainty_quantification_v2(df)
    assert np.isclose(v1, 0.5, atol=1e-3), f"unexpected v1={v1}"
    assert abs(v2 - 0.49718) < 1e-4, f"unexpected v2={v2}"

    # Single-arm edge case
    assert uq_directional_bc_from_moments(np.array([[1.0, 2.0]]), np.array([[0.1, 0.1]])) == 0.0
    assert uq_dominance_risk_from_moments(np.array([[1.0, 2.0]]), np.array([[0.1, 0.1]])) == 0.0

    # Non-finite input returns NaN safely
    assert np.isnan(uq_directional_bc_from_moments(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[np.inf, 0.1], [0.1, 0.1]])))
    assert np.isnan(uq_dominance_risk_from_moments(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[np.inf, 0.1], [0.1, 0.1]])))

    # Chunked MC reproducibility check
    q1, se1 = uq_mc_disagreement_from_moments(
        np.vstack([df["means"].tolist()]), np.vstack([df["stds"].tolist()]),
        n_draws=256, seed=99, batch_size=32,
    )
    q2, se2 = uq_mc_disagreement_from_moments(
        np.vstack([df["means"].tolist()]), np.vstack([df["stds"].tolist()]),
        n_draws=256, seed=99, batch_size=256,
    )
    assert q1 == q2 and se1 == se2
    print("[PASS] DataFrame wrappers, single-arm, non-finite, and MC chunking invariance checks passed.")


def main() -> int:
    test_dataframe_wrappers_and_edge_cases()
    run_checks()
    print("ALL UQ VARIANT TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
