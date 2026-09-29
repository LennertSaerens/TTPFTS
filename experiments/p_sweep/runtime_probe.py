#!/usr/bin/env python3
"""Measure local TTPFTS p-sweep runtime before launching the full sweep."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from .config import default_output_root, load_config
from .run import run, run_repetition


def probe(
    *,
    environments: list[str],
    p_values: list[float],
    repetition: int,
    concurrency_probe: bool,
    config: dict,
) -> dict[str, object]:
    """Measure representative exact-budget trajectories and optional eight-way load."""
    measurements = []
    with tempfile.TemporaryDirectory(prefix="mots-p-sweep-probe-") as temporary:
        root = Path(temporary)
        for environment in environments:
            for p in p_values:
                start = time.perf_counter()
                run_repetition(environment, p, repetition, root, config)
                measurements.append(
                    {
                        "environment": environment,
                        "p": p,
                        "repetition": repetition,
                        "wall_seconds": time.perf_counter() - start,
                    }
                )
        concurrency = None
        if concurrency_probe:
            from .config import expected_environments

            all_environments = [item["name"] for item in expected_environments()]
            concurrent_root = root / "concurrent"
            start = time.perf_counter()
            run(all_environments, [0.5], [repetition], concurrent_root, config, n_jobs=8)
            concurrency = {
                "environment_count": len(all_environments),
                "p": 0.5,
                "repetition": repetition,
                "workers": 8,
                "wall_seconds": time.perf_counter() - start,
            }

    by_environment_p = {
        f"{record['environment']}/p={record['p']:.1f}": record["wall_seconds"]
        for record in measurements
    }
    # Each representative measurement is one exact trajectory. Scaling by 100
    # repetitions estimates core-seconds; dividing by eight is only a balanced
    # wall-time estimate and is reported separately from observed contention.
    estimated_core_seconds = sum(float(record["wall_seconds"]) * 100 for record in measurements)
    return {
        "schema_version": "hardness-normalized-ttpfts-p-sweep-runtime-probe-v1",
        "host_cpu_count": os.cpu_count(),
        "measurements": measurements,
        "seconds_by_environment_p": by_environment_p,
        "representative_estimated_core_seconds_for_100_repetitions": estimated_core_seconds,
        "representative_balanced_eight_worker_seconds": estimated_core_seconds / 8,
        "concurrency_probe": concurrency,
        "note": (
            "This probe measures only the listed environment/p combinations. "
            "Do not treat its scaled estimate as a substitute for full-run accounting."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environments", default="EgeExp2,EgeExp3")
    parser.add_argument("--p-values", default="0.0,0.5,1.0")
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--concurrency-probe", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_config()
    result = probe(
        environments=args.environments.split(","),
        p_values=[float(value) for value in args.p_values.split(",")],
        repetition=args.repetition,
        concurrency_probe=args.concurrency_probe,
        config=config,
    )
    output = args.output or default_output_root(config) / "runtime_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
