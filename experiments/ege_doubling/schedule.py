"""Pure doubling-trick schedule arithmetic (no I/O, no randomness)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Stage:
    """One completed doubling stage.

    ``budget`` is the nominal fixed budget given to the stage's fresh EGE run,
    ``start`` is the cumulative pull at which that run completes (``e_s``), and
    ``end`` is the last cumulative pull, inclusive, at which its recommendation
    is still the most recent completed one.
    """

    index: int
    budget: int
    start: int
    end: int


def doubling_schedule(num_arms: int, max_budget: int) -> list[Stage]:
    """Return every stage that completes within ``max_budget`` pulls.

    Stage ``s`` has budget ``b_s = 2**(s-1) * K`` and completes at
    ``e_s = (2**s - 1) * K``. A stage whose endpoint exceeds ``max_budget`` is
    incomplete and therefore never contributes a recommendation.
    """
    num_arms, max_budget = int(num_arms), int(max_budget)
    if num_arms < 2:
        raise ValueError(f"EGE requires at least two arms, got {num_arms}")
    if max_budget < num_arms:
        raise ValueError(f"max_budget={max_budget} is smaller than the first stage endpoint K={num_arms}")
    endpoints: list[tuple[int, int]] = []
    index = 1
    while (2**index - 1) * num_arms <= max_budget:
        endpoints.append((2 ** (index - 1) * num_arms, (2**index - 1) * num_arms))
        index += 1
    stages = []
    for position, (budget, start) in enumerate(endpoints):
        end = endpoints[position + 1][1] - 1 if position + 1 < len(endpoints) else max_budget
        stages.append(Stage(index=position + 1, budget=budget, start=start, end=end))
    return stages


def stage_at(schedule: Sequence[Stage], arm_pull: int) -> Stage | None:
    """Return the most recently completed stage at ``arm_pull`` (None before stage 1)."""
    arm_pull = int(arm_pull)
    if not schedule or arm_pull < schedule[0].start:
        return None
    if arm_pull > schedule[-1].end:
        raise ValueError(f"arm_pull={arm_pull} exceeds the schedule's final budget {schedule[-1].end}")
    current = schedule[0]
    for stage in schedule:
        if stage.start <= arm_pull:
            current = stage
        else:
            break
    return current


def evaluation_points(schedule: Sequence[Stage], grid: Iterable[int], max_budget: int) -> list[int]:
    """Exact step-function evaluation points for one doubling trajectory.

    The union of every stage endpoint ``e_s``, the pull just before every later
    endpoint (``e_s - 1``, so both sides of each jump are recorded), every
    shared grid point at or after ``K``, and ``max_budget``. Nothing earlier
    than the first endpoint is included: no stage has completed there.
    """
    if not schedule:
        raise ValueError("empty doubling schedule")
    first = schedule[0].start
    points = {stage.start for stage in schedule}
    points.update(stage.start - 1 for stage in schedule[1:])
    points.update(int(value) for value in grid if first <= int(value) <= int(max_budget))
    points.add(int(max_budget))
    return sorted(points)
