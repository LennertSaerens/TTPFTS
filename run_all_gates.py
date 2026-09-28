#!/usr/bin/env python3
"""Run the complete MOMAB validation/gate suite from one entry point."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOCAL_TMP = ROOT / ".gate_tmp"


@dataclass(frozen=True)
class Gate:
    name: str
    command: tuple[str, ...]
    description: str
    slow: bool = False


GATES: tuple[Gate, ...] = (
    Gate("ape", ("test_ape_equivalence.py",),
         "Gate 1: APE reference equivalence (includes C++ helper parity)"),
    Gate("posteriors", ("validate_posteriors.py",),
         "Gate 2: TTPFTS known/unknown-variance posterior validation", slow=True),
    Gate("uq-replay", ("test_uq_replay.py",),
         "UQ replay: posterior-state validity and trajectory-correlation semantics"),
    Gate("reward-report", ("-m", "environments.validate_reward_environment_gate"),
         "Gate 2/3: regenerate reward/environment/hardness reports"),
    Gate("environment", ("test_environment_reward_hardness_gate.py",),
         "Gate 2/3: reward covariance, frozen instances, Pareto/hardness checks"),
    Gate("runner-grid-schema", ("test_runner_grid_seeding_schema.py",),
         "Phase 5: grid, seeding, instance hash, schema checks"),
    Gate("runner-shards", ("test_runner_shards_integrity.py",),
         "Phase 5: atomic shard and deterministic merge checks"),
    Gate("runner-pipeline", ("test_runner_pipeline_invariance.py",),
         "Phase 5: end-to-end runner invariance and resume checks", slow=True),
    Gate("legacy-fixes", ("test_legacy_fixes_coverage.py",),
         "Retired verify_fixes.py coverage: five original bug-fix behaviours"),
)


def _python() -> str:
    venv_python = ROOT / ".venv" / "bin" / "python"
    return str(venv_python if venv_python.exists() else Path(sys.executable))


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    LOCAL_TMP.mkdir(exist_ok=True)
    # Keep all subprocess temporary files inside the repository.  This avoids
    # relying on caller shell state and keeps C++ build/test scratch local.
    env["TMPDIR"] = str(LOCAL_TMP)
    env["TEMP"] = str(LOCAL_TMP)
    env["TMP"] = str(LOCAL_TMP)
    return env


def _selected_gates(only: str | None, fast: bool) -> list[Gate]:
    gates = list(GATES)
    if only:
        aliases = {g.name: g for g in gates}
        matches = [g for g in gates if g.name == only]
        if not matches:
            prefix_matches = [g for g in gates if g.name.startswith(only)]
            if len(prefix_matches) == 1:
                matches = prefix_matches
        if not matches:
            available = ", ".join(aliases)
            raise SystemExit(f"Unknown gate {only!r}. Available: {available}")
        gates = matches
    if fast:
        gates = [g for g in gates if not g.slow]
    return gates


def _run_gate(gate: Gate, env: dict[str, str]) -> tuple[str, bool, float]:
    cmd = (_python(), *gate.command)
    print(f"\n=== {gate.name}: {gate.description} ===", flush=True)
    print("+ " + " ".join(cmd), flush=True)
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT, env=env, text=True)
    elapsed = time.perf_counter() - start
    return gate.name, result.returncode == 0, elapsed


def _print_summary(rows: list[tuple[str, bool, float]], skipped: list[Gate]) -> None:
    print("\n" + "=" * 72)
    print("Gate summary")
    print("=" * 72)
    print(f"{'gate':<24} {'status':<8} {'seconds':>10}")
    print("-" * 72)
    for name, ok, elapsed in rows:
        print(f"{name:<24} {('PASS' if ok else 'FAIL'):<8} {elapsed:>10.2f}")
    for gate in skipped:
        print(f"{gate.name:<24} {'SKIPPED':<8} {'--':>10}")
    print("-" * 72)
    if rows:
        slowest = sorted(rows, key=lambda row: row[2], reverse=True)[:3]
        slow_text = ", ".join(f"{name} ({elapsed:.2f}s)" for name, _, elapsed in slowest)
        print(f"Slowest measured gates: {slow_text}")
    if skipped:
        print("Fast mode skipped: " + ", ".join(g.name for g in skipped))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="Skip gates marked slow (currently posteriors and runner-pipeline).")
    parser.add_argument("--only", metavar="NAME",
                        help="Run one gate by exact name or unique prefix.")
    parser.add_argument("--list", action="store_true", help="List gate names and exit.")
    args = parser.parse_args(argv)

    if args.list:
        for gate in GATES:
            marker = " [slow]" if gate.slow else ""
            print(f"{gate.name:<24}{marker}  {gate.description}")
        return 0

    selected = _selected_gates(args.only, args.fast)
    selected_names = {g.name for g in selected}
    skipped = [g for g in GATES if g.name not in selected_names and (args.fast and g.slow)]

    env = _env()
    rows = [_run_gate(gate, env) for gate in selected]
    _print_summary(rows, skipped)

    try:
        if LOCAL_TMP.exists() and not any(LOCAL_TMP.iterdir()):
            LOCAL_TMP.rmdir()
    except OSError:
        pass

    return 0 if all(ok for _, ok, _ in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
