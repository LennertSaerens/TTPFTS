"""Measure warmed steady-state per-pull costs for final anytime algorithms.

Run from the repository root:

    PYTHONPATH=. ./.venv/bin/python experiments/reports/measure_anytime_per_pull.py

The script intentionally warms the numba-backed ``paretoset`` path and then
warms each bandit past its own initial round-robin phase before timing. This
avoids assigning one-off JIT compilation or short-window posterior warm-up
effects to an algorithm/environment pair.
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import numpy as np
from paretoset import paretoset

from momab_main import ENVIRONMENTS, _construct_environment, _make_bandit
from runner.seeding import seed_rngs


ALGORITHMS = ["Uniform", "APE", "TTPFTS_KNOWN_VAR", "TTPFTS_UNKNOWN_VAR"]
ENVIRONMENTS_FINAL = [f"EgeExp{i}" for i in range(1, 9)]
TIMED_PULLS = {
    "Uniform": 50_000,
    "APE": 1_000,
    "TTPFTS_KNOWN_VAR": 5_000,
    "TTPFTS_UNKNOWN_VAR": 5_000,
}
REPEATS = 5


def run_pulls(env, bandit, pulls: int) -> None:
    for _ in range(pulls):
        arm = bandit.choose_arm()
        reward = env.pull_arm(arm)
        bandit.learn(arm, reward)


def state_warmup_pulls(algorithm: str, num_arms: int) -> int:
    if algorithm.startswith("TTPFTS_"):
        return 4 * num_arms + 100
    if algorithm == "APE":
        return num_arms + 100
    return 100


def warm_numba() -> None:
    # Compile the numba-backed paretoset path before any timed region.
    rng = np.random.default_rng(0)
    for k, d in [(10, 2), (60, 2), (50, 10), (200, 2)]:
        paretoset(rng.normal(size=(k, d)), sense=["max"] * d, distinct=False)


def measure_once(env_name: str, algorithm: str, repeat: int) -> float:
    env_cls, _ = ENVIRONMENTS[env_name]
    env = _construct_environment(env_cls, "gaussian")
    seed_rngs(0, env_name, algorithm, "anytime-cost", repeat)
    env.reset()
    bandit = _make_bandit(algorithm, env, p=0.5)
    if bandit is None:
        raise ValueError(f"Unknown algorithm {algorithm}")
    bandit.reset(env.stds)

    run_pulls(env, bandit, state_warmup_pulls(algorithm, env.num_arms))
    start = time.perf_counter()
    run_pulls(env, bandit, TIMED_PULLS[algorithm])
    elapsed = time.perf_counter() - start
    return elapsed * 1_000_000 / TIMED_PULLS[algorithm]


def main() -> None:
    warm_numba()
    rows = []
    detail_rows = []
    for env_name in ENVIRONMENTS_FINAL:
        env_cls, _ = ENVIRONMENTS[env_name]
        env = _construct_environment(env_cls, "gaussian")
        row = {"env": env_name, "K": env.num_arms, "D": env.num_objectives}
        for algorithm in ALGORITHMS:
            values = [measure_once(env_name, algorithm, repeat) for repeat in range(REPEATS)]
            median = statistics.median(values)
            row[algorithm] = f"{median:.3f}"
            for repeat, value in enumerate(values):
                detail_rows.append({
                    "env": env_name,
                    "K": env.num_arms,
                    "D": env.num_objectives,
                    "algorithm": algorithm,
                    "repeat": repeat,
                    "timed_pulls": TIMED_PULLS[algorithm],
                    "state_warmup_pulls": state_warmup_pulls(algorithm, env.num_arms),
                    "microseconds_per_pull": f"{value:.6f}",
                })
        rows.append(row)
        print(row, flush=True)

    out = Path("experiments/reports/anytime_per_pull_costs.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["env", "K", "D", *ALGORITHMS])
        writer.writeheader()
        writer.writerows(rows)

    detail_out = Path("experiments/reports/anytime_per_pull_measurements.csv")
    with detail_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)


if __name__ == "__main__":
    main()
