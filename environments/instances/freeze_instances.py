"""
Deterministic, version-controlled derivation of the frozen EgeExp1/EgeExp4/EgeExp5
arm-mean instances used for the paper's final experiments.

Rationale
---------
EgeExp1, EgeExp4 and EgeExp5 originally redrew part (or all) of their arm means
from ``np.random`` every time ``reset()`` was called, so every one of the 100
trajectories per algorithm faced a *different* random problem instance. That
is scientifically undesirable for the "final" paper runs: it silently
convolves environment-instance variance with algorithm variance, and it makes
the reported hardness H(nu) and Pareto set undefined (they depend on the
realized arm means).

Per user decision, exactly ONE problem instance is frozen per environment,
generated from a single preregistered master seed (0), with NO seed-shopping:
this script is run exactly once, with the RNG streams below, and the output is
accepted and committed as-is -- it is never re-run with a different seed to
"improve" the resulting instance. Re-running this script must reproduce
byte-identical CSVs (verified by the checksums this script also (re)writes).

Usage
-----
    python -m environments.instances.freeze_instances

This (re)writes:
    environments/instances/EgeExp1_arms.csv
    environments/instances/EgeExp4_arms.csv
    environments/instances/EgeExp5_arms.csv
    environments/instances/CHECKSUMS.sha256
    environments/instances/manifest.json
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

MASTER_SEED = 0
INSTANCES_DIR = Path(__file__).resolve().parent

# Fixed, documented spawn order -- determines which independent RNG substream
# each environment receives from the single master SeedSequence. This order
# must never change once frozen (it is part of the preregistered derivation).
ENV_SPAWN_ORDER = ["EgeExp1", "EgeExp4", "EgeExp5"]


def _make_child_generators() -> dict:
    """Spawn one independent RNG stream per environment from MASTER_SEED.

    Using SeedSequence.spawn gives each environment a statistically
    independent, reproducible substream without any manual seed arithmetic.
    """
    ss = np.random.SeedSequence(MASTER_SEED)
    children = ss.spawn(len(ENV_SPAWN_ORDER))
    return {
        name: np.random.Generator(np.random.PCG64(child))
        for name, child in zip(ENV_SPAWN_ORDER, children)
    }


def _generate_egeexp1_arms(rng: np.random.Generator) -> np.ndarray:
    """Reproduces environments/EgeExp1.py:generate_arms() with an explicit RNG."""
    arms1 = [(x ** 2, 1 / (4 * x ** 2)) for x in np.linspace(0.55, 0.95, 10)]
    arms2 = []
    while len(arms2) < 50:
        x = rng.uniform(0.1, 0.8)
        y = rng.uniform(0.1, 0.8)
        if x * y <= 0.2:
            arms2.append((x, y))
    return np.array(arms1 + arms2)


def _generate_egeexp4_arms(rng: np.random.Generator) -> np.ndarray:
    """Reproduces environments/EgeExp4.py:generate_arms() with an explicit RNG."""
    arms_low = rng.uniform(0.2, 0.45, size=(30, 10))
    arms_high = rng.uniform(0.55, 0.75, size=(20, 10))
    return np.vstack([arms_low, arms_high])


def _generate_egeexp5_arms(rng: np.random.Generator) -> np.ndarray:
    """Reproduces environments/EgeExp5.py:generate_arms() with an explicit RNG."""
    arms_low = rng.uniform(0.2, 0.4, size=(10, 2))
    arms_high = rng.uniform(0.5, 0.7, size=(10, 2))
    return np.vstack([arms_low, arms_high])


_GENERATORS = {
    "EgeExp1": _generate_egeexp1_arms,
    "EgeExp4": _generate_egeexp4_arms,
    "EgeExp5": _generate_egeexp5_arms,
}

_PARETO_INDICES = {
    # EgeExp1/EgeExp3 hardcode the first block of arms as Pareto-optimal by
    # construction (see the respective generate_arms() docstrings).
    "EgeExp1": list(range(10)),
    # EgeExp4/EgeExp5 do not hardcode a Pareto set; it is computed from the
    # frozen means at freeze time and recorded here for the manifest only
    # (the environments still recompute it from the loaded means at runtime).
}


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_csv(path: Path, arms: np.ndarray) -> None:
    num_objectives = arms.shape[1]
    header = [f"obj_{c}" for c in range(num_objectives)]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in arms:
            writer.writerow([f"{v:.17g}" for v in row])


def freeze_all(write: bool = True) -> dict:
    """Derive all frozen instances and (optionally) write them + checksums/manifest.

    Returns a dict {env_name: arms_ndarray} for programmatic use (e.g. by the
    hardness/Pareto validation scripts), regardless of `write`.
    """
    from paretoset import paretoset  # local import: only needed at freeze time

    generators = _make_child_generators()
    arms_by_env = {name: _GENERATORS[name](generators[name]) for name in ENV_SPAWN_ORDER}

    if not write:
        return arms_by_env

    manifest = {
        "master_seed": MASTER_SEED,
        "spawn_order": ENV_SPAWN_ORDER,
        "note": (
            "Frozen once from MASTER_SEED via SeedSequence(MASTER_SEED).spawn(3) "
            "in spawn_order, without seed-shopping. Re-running freeze_instances.py "
            "must reproduce byte-identical CSVs (see CHECKSUMS.sha256)."
        ),
        "environments": {},
    }

    for name in ENV_SPAWN_ORDER:
        arms = arms_by_env[name]
        csv_path = INSTANCES_DIR / f"{name}_arms.csv"
        _write_csv(csv_path, arms)
        pareto_mask = paretoset(arms, sense=["max"] * arms.shape[1], distinct=False)
        pareto_indices = np.where(pareto_mask)[0].tolist()
        if name in _PARETO_INDICES:
            assert pareto_indices == _PARETO_INDICES[name], (
                f"{name}: computed Pareto indices {pareto_indices} do not match "
                f"the hardcoded set {_PARETO_INDICES[name]} used by the original "
                f"environment; freezing aborted."
            )
        manifest["environments"][name] = {
            "num_arms": int(arms.shape[0]),
            "num_objectives": int(arms.shape[1]),
            "csv_file": csv_path.name,
            "pareto_indices": pareto_indices,
        }

    checksum_lines = []
    for name in ENV_SPAWN_ORDER:
        csv_path = INSTANCES_DIR / f"{name}_arms.csv"
        digest = _sha256_of_file(csv_path)
        manifest["environments"][name]["sha256"] = digest
        checksum_lines.append(f"{digest}  {csv_path.name}")

    (INSTANCES_DIR / "CHECKSUMS.sha256").write_text("\n".join(checksum_lines) + "\n")
    (INSTANCES_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    return arms_by_env


if __name__ == "__main__":
    result = freeze_all(write=True)
    for env_name, arms in result.items():
        print(f"{env_name}: froze {arms.shape[0]} arms x {arms.shape[1]} objectives")
    print(f"Wrote CSVs, CHECKSUMS.sha256 and manifest.json to {INSTANCES_DIR}")
