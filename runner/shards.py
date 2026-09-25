"""Atomic per-(environment, algorithm, repetition) result shards.

Each repetition of each algorithm in each environment is written to its own
shard file. A shard is only ever written by atomic replace (write to a
temp file in the same directory, then ``os.replace``), so a reader never
observes a partially-written shard: it either sees the previous complete
shard (if any) or the new complete one, never a half-written file. This
removes the need for any cross-process locking to avoid corruption; the
worst case of two workers redoing the same (deterministic) shard
concurrently is two equivalent atomic replacements, not a corrupted file.

A repetition is considered *complete* only if its shard exists, parses, has
exactly the expected schema columns, and contains exactly one row per
budget in the expected evaluation grid (no more, no fewer, no duplicates).
Anything else (missing file, truncated/corrupt CSV, wrong columns, a grid
that doesn't match exactly) is treated as incomplete and is recomputed from
scratch and overwritten atomically -- this is what "resume" means here,
replacing the old unsafe shared-append `_completed_experiments` semantics.
"""

from __future__ import annotations

import csv
import os
import tempfile
from typing import Iterable, Optional

from runner.schema import SCHEMA_COLUMNS, validate_row_metrics


def sanitize_component(name: str) -> str:
    """Make a string safe to use as a single path component."""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(name))


def shard_path(run_dir: str, environment: str, algorithm: str, repetition: int) -> str:
    return os.path.join(
        run_dir, "shards",
        sanitize_component(environment),
        sanitize_component(algorithm),
        f"rep{int(repetition):05d}.csv",
    )


def write_shard_atomic(path: str, rows: list[dict]) -> None:
    """Atomically write a complete shard (temp file + os.replace)."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-shard-", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SCHEMA_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def read_shard(path: str) -> list[dict]:
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def shard_is_complete(path: str, expected_grid: Iterable[int]) -> bool:
    """Whether an existing shard is well-formed and covers exactly the
    expected grid, with no duplicate/missing/extra budgets."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        rows = read_shard(path)
    except (OSError, csv.Error):
        return False
    if not rows:
        return False
    if set(rows[0].keys()) != set(SCHEMA_COLUMNS):
        return False
    try:
        arm_pulls = [int(r["arm_pull"]) for r in rows]
    except (KeyError, ValueError, TypeError):
        return False
    if len(arm_pulls) != len(set(arm_pulls)):
        return False  # duplicates
    return sorted(arm_pulls) == sorted(int(v) for v in expected_grid)


def needs_recompute(path: str, expected_grid: Iterable[int]) -> bool:
    return not shard_is_complete(path, expected_grid)


def discover_shards(run_dir: str, environment: Optional[str] = None) -> list[str]:
    """List every shard CSV path under a run directory (optionally scoped to
    one environment), in deterministic sorted order."""
    root = os.path.join(run_dir, "shards")
    if environment is not None:
        root = os.path.join(root, sanitize_component(environment))
    if not os.path.isdir(root):
        return []
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".csv") and not name.startswith(".tmp-"):
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def merge_shards(run_dir: str, environment: str, output_path: str,
                 expected_grids: Optional[dict] = None,
                 validate_metrics: bool = True) -> int:
    """Deterministically merge every *complete* shard for one environment
    into a single tidy CSV.

    :param expected_grids: optional ``{algorithm: grid}`` map; when given,
        shards whose algorithm has a known expected grid are re-validated
        for completeness before merging (skips stale/partial shards rather
        than silently including them).
    :param validate_metrics: when True, every merged row's stored metrics
        are cross-checked against metrics recomputed from its own
        ``recommendation``/``true_pareto`` columns; a mismatch raises.
    :return: number of rows written.
    """
    all_rows: list[dict] = []
    for path in discover_shards(run_dir, environment):
        rows = read_shard(path)
        if not rows:
            continue
        algorithm = rows[0]["algorithm"]
        if expected_grids is not None and algorithm in expected_grids:
            if not shard_is_complete(path, expected_grids[algorithm]):
                continue
        all_rows.extend(rows)

    all_rows.sort(key=lambda r: (r["algorithm"], int(r["repetition"]), int(r["arm_pull"])))

    if validate_metrics:
        for row in all_rows:
            num_arms = len(_decode_or_empty(row["true_pareto"])) or len(_decode_or_empty(row["recommendation"]))
            # num_arms proper value isn't stored directly; misidentification
            # was computed with the real num_arms, so re-derive it instead of
            # guessing: symmetric_diff / misidentification = num_arms.
            problems = validate_row_metrics(row, _num_arms_from_row(row))
            if problems:
                raise ValueError(
                    f"Shard row failed metric recomputation for "
                    f"{row['environment']}/{row['algorithm']}/rep{row['repetition']}/"
                    f"t={row['arm_pull']}: {problems}"
                )

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory or ".", prefix=".tmp-merge-", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SCHEMA_COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, output_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return len(all_rows)


def _decode_or_empty(encoded: str) -> set:
    import json
    try:
        return set(json.loads(encoded))
    except (TypeError, ValueError):
        return set()


def _num_arms_from_row(row: dict) -> int:
    """Recover num_arms from a row's own misidentification value.

    misidentification = |symmetric_diff| / num_arms, and both
    recommendation/true_pareto are known, so whenever the symmetric
    difference is non-empty num_arms is exactly recoverable; when it is
    empty (a perfect recommendation) misidentification is 0 regardless of
    num_arms, so the check degenerates to 0 == 0 and any positive num_arms
    works.
    """
    rec_set = _decode_or_empty(row["recommendation"])
    pareto_set = _decode_or_empty(row["true_pareto"])
    symmetric_diff = rec_set.symmetric_difference(pareto_set)
    misid = float(row["misidentification"])
    if symmetric_diff and misid > 0:
        recovered = len(symmetric_diff) / misid
        return int(round(recovered))
    # Fall back to the largest arm index seen + 1 (still exercises bernoulli/jaccard).
    return max(rec_set | pareto_set, default=-1) + 1
