"""Hardened experiment-running utilities for momab_main.py (Phase 5).

Sub-modules:
  - grid: endpoint-inclusive evaluation grids shared across algorithms.
  - seeding: deterministic RNG seeding from (master seed, tags...).
  - instance: deterministic hashing of a frozen environment instance.
  - schema: the tidy self-describing result schema + recomputable metrics.
  - shards: atomic per-(environment, algorithm, repetition) result shards,
    completeness validation, and deterministic merging.
  - config: run-level configuration / provenance manifest.
"""
