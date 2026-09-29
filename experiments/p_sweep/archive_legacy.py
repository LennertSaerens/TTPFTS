#!/usr/bin/env python3
"""Fingerprint, without moving, the superseded flat-5,000 known-variance p sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results"
ARCHIVED_FIGURE = ROOT / "archive_prepatch" / "figures" / "sweep_experiment.png"
SLURM_SCRIPT = ROOT / "hpc" / "sweep_ttpfts_p.slurm"
OUTPUT = Path(__file__).resolve().parent / "legacy_flat_5000_manifest.json"
CELL_PATTERN = re.compile(r"^EgeExp[1-8]_TTPFTS_GKV_p(?:0\.[0-9]|1\.0)_5000_100\.csv$")
COMBINED_PATTERN = re.compile(r"^EgeExp[1-8]_sweep_combined\.csv$")


def sha256(path: Path) -> str:
    """Stream a file digest so the multi-gigabyte legacy corpus is never copied."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def build_manifest(results_root: Path = RESULTS_ROOT) -> dict[str, object]:
    """Build a strict inventory of the 88 cells, eight combined files, and evidence."""
    cells = sorted(path for path in results_root.iterdir() if CELL_PATTERN.fullmatch(path.name))
    combined = sorted(path for path in results_root.iterdir() if COMBINED_PATTERN.fullmatch(path.name))
    if len(cells) != 88:
        raise ValueError(f"expected 88 legacy p-sweep cells, found {len(cells)}")
    if len(combined) != 8:
        raise ValueError(f"expected eight legacy combined files, found {len(combined)}")
    supporting = [path for path in (ARCHIVED_FIGURE, SLURM_SCRIPT) if path.is_file()]
    if len(supporting) != 2:
        raise FileNotFoundError("missing archived legacy figure or legacy SLURM script")
    files = [_entry(path) for path in [*cells, *combined, *supporting]]
    return {
        "schema_version": "legacy-known-variance-flat-5000-p-sweep-v1",
        "status": "superseded_preserved_not_for_current_analysis",
        "design": {
            "algorithm": "TTPFTS_GKV",
            "posterior": "NormalPosterior with known true reward variance",
            "max_budget": 5000,
            "checkpoints": "every arm pull",
            "num_runs": 100,
            "p_values": [round(index / 10, 1) for index in range(11)],
        },
        "cell_file_count": len(cells),
        "combined_file_count": len(combined),
        "files": files,
        "total_bytes": sum(int(entry["bytes"]) for entry in files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = build_manifest(args.results_root)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}: {len(manifest['files'])} checksummed files")


if __name__ == "__main__":
    main()
