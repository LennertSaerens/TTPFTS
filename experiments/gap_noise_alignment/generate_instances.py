#!/usr/bin/env python3
"""Freeze the paired custom environments and print their immutable identity."""

from __future__ import annotations

from .config import config_hash, load_config
from .instances import freeze_conditions, load_conditions


def main() -> None:
    config = load_config()
    frozen = freeze_conditions(config)
    reloaded = load_conditions()
    if [item.instance_hash for item in frozen] != [item.instance_hash for item in reloaded]:
        raise RuntimeError("reload verification changed a frozen condition identity")
    print(f"config_hash={config_hash(config)}")
    for item in frozen:
        print(
            f"{item.condition}: pareto={item.pareto_indices.tolist()} "
            f"H={item.hardness.hardness:.12f} B={item.budget} hash={item.instance_hash}"
        )


if __name__ == "__main__":
    main()
