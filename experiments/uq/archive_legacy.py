#!/usr/bin/env python3
"""Create a checksummed manifest for the immutable legacy 5,000-pull UQ corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = ROOT / "results_corr"
OUTPUT = Path(__file__).resolve().parent / "legacy_5000_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(root: Path) -> dict:
    if not root.is_dir():
        raise FileNotFoundError(root)
    entries = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != ".DS_Store"):
        entries.append(
            {
                "path": path.relative_to(root.parent).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "schema_version": "legacy-uq-archive-v1",
        "design": {
            "max_budget": 5000,
            "status": "superseded_by_hardness_normalized_experiment",
            "note": "Preserved legacy data. It is not evidence for the current hardness-budget design.",
        },
        "root": root.name,
        "files": entries,
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, default=LEGACY_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = build_manifest(args.legacy_root)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}: {manifest['file_count']} files")


if __name__ == "__main__":
    main()
