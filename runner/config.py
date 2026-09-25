"""Run-level configuration / provenance manifest.

Per-row schema columns hold everything needed to recompute a single row's
metrics; the manifest holds everything needed to reproduce the *run itself*
(which CLI arguments were used, when, against which environments/algorithms),
so it is written once per run directory rather than repeated on every row.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class RunManifest:
    master_seed: Optional[int]
    dist: str
    environments: list[str]
    algorithms: list[str]
    num_runs: int
    step: int
    explicit_budgets: Optional[list[int]]
    eval_points: Optional[int]
    budgets: dict  # {environment: max_budget}
    evaluation_grids: dict  # {environment: [budgets...]}
    n_jobs: int
    rho: float
    results_dir: str
    run_dir: str
    created_at: float = field(default_factory=time.time)
    cli_args: Optional[list[str]] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self, path: str) -> None:
        """Atomically write the manifest as sidecar JSON."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory or ".", prefix=".tmp-manifest-", suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.to_dict(), f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    @classmethod
    def read(cls, path: str) -> "RunManifest":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
