"""Endpoint-inclusive evaluation grids.

A single evaluation grid is computed per environment/budget and re-used for
*every* algorithm, so that dense vs. sparse ``--step`` values, or an explicit
list of evaluation budgets, never make one algorithm's logged points a subset
of another's for the same environment.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union


def parse_explicit_budgets(spec: Union[str, Sequence[int], None]) -> Optional[list[int]]:
    """Parse a comma-separated string (or pass through a sequence) of budgets."""
    if spec is None:
        return None
    if isinstance(spec, str):
        stripped = spec.strip()
        if not stripped:
            return None
        return [int(v.strip()) for v in stripped.split(",") if v.strip() != ""]
    return [int(v) for v in spec]


def build_normalized_grid(max_budget: int, num_intervals: int) -> list[int]:
    """Build a grid of ``num_intervals + 1`` points spread over ``[0, B]``.

    Point ``k`` is ``round(k * max_budget / num_intervals)``, so the grid
    always contains ``0`` and ``max_budget`` exactly and every environment is
    evaluated at the same *normalized* budgets even when the raw maxima
    differ. Duplicates (possible when ``num_intervals > max_budget``) are
    collapsed, so the result may be shorter than ``num_intervals + 1``.
    """
    if max_budget < 0:
        raise ValueError(f"max_budget must be non-negative, got {max_budget}")
    if num_intervals < 1:
        raise ValueError(f"num_intervals must be a positive integer, got {num_intervals}")
    values = {int(round(k * max_budget / num_intervals)) for k in range(num_intervals + 1)}
    values.add(0)
    values.add(max_budget)
    return sorted(values)


def build_evaluation_grid(max_budget: int, step: int = 1,
                          explicit_budgets: Optional[Iterable[int]] = None,
                          num_intervals: Optional[int] = None) -> list[int]:
    """Build the endpoint-inclusive evaluation grid for a single environment.

    :param max_budget: The fixed pull budget B (must be >= 0).
    :param step: Stride between logged points, in arm pulls. ``0`` and
        ``max_budget`` are always included even when ``max_budget`` is not a
        multiple of ``step``.
    :param explicit_budgets: Optional explicit set of evaluation budgets
        (overrides ``step`` entirely when given). Values outside
        ``[0, max_budget]`` are rejected rather than silently clipped, since
        silently dropping a requested point would break the "same grid for
        every algorithm" contract.
    :param num_intervals: Optional normalized-grid size, overriding ``step``
        (but not ``explicit_budgets``). See :func:`build_normalized_grid`.
    :return: A sorted list of unique integers, always containing 0 and
        ``max_budget``.
    """
    if max_budget < 0:
        raise ValueError(f"max_budget must be non-negative, got {max_budget}")

    if explicit_budgets is not None:
        if num_intervals is not None:
            raise ValueError("explicit_budgets and num_intervals are mutually exclusive")
        values = {int(v) for v in explicit_budgets}
        out_of_range = [v for v in values if v < 0 or v > max_budget]
        if out_of_range:
            raise ValueError(
                f"Explicit evaluation budgets {sorted(out_of_range)} fall outside "
                f"[0, {max_budget}]"
            )
    elif num_intervals is not None:
        return build_normalized_grid(max_budget, num_intervals)
    else:
        if step < 1:
            raise ValueError(f"step must be a positive integer, got {step}")
        values = set(range(0, max_budget + 1, step))

    values.add(0)
    values.add(max_budget)
    return sorted(values)
