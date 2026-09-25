"""Loader and CLI helpers for the preregistered final MOMAB experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).with_name("final_config.json")


def _canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_final_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def config_hash(config: dict[str, Any] | None = None) -> str:
    data = load_final_config() if config is None else config
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def environments(config: dict[str, Any] | None = None) -> list[str]:
    data = load_final_config() if config is None else config
    return [env["name"] for env in data["environments"]]


def budgets(config: dict[str, Any] | None = None) -> list[int]:
    data = load_final_config() if config is None else config
    return [int(env["budget"]) for env in data["environments"]]


def algorithms(config: dict[str, Any] | None = None) -> list[str]:
    data = load_final_config() if config is None else config
    return list(data["algorithms"])


def build_momab_args(
    *,
    config: dict[str, Any] | None = None,
    results_dir: str,
    run_dir: str | None = None,
    n_jobs: int = 1,
    num_runs: int | None = None,
    budgets_override: list[int] | None = None,
    environments_override: list[str] | None = None,
    algorithms_override: list[str] | None = None,
) -> list[str]:
    """Build a ``momab_main.py`` argument vector from the frozen config."""
    data = load_final_config() if config is None else config
    envs = environments_override or environments(data)
    env_budget_map = {env["name"]: int(env["budget"]) for env in data["environments"]}
    selected_budgets = budgets_override or [env_budget_map[e] for e in envs]
    algs = algorithms_override or algorithms(data)
    args = [
        "--algorithms", ",".join(algs),
        "--environments", ",".join(envs),
        "--budgets", ",".join(str(b) for b in selected_budgets),
        "--num_runs", str(num_runs if num_runs is not None else data["num_runs"]),
        "--eval_points", str(data["eval_points"]),
        "--dist", data["reward_model"]["distribution"],
        "--seed", str(data["master_seed"]),
        "--n_jobs", str(n_jobs),
        "--results_dir", results_dir,
    ]
    if run_dir is not None:
        args.extend(["--run_dir", run_dir])
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--hash", action="store_true", help="Print the canonical config hash.")
    parser.add_argument("--print-momab-args", action="store_true")
    parser.add_argument("--results-dir", default="results_preregistered_final")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--environments", default=None, help="Optional comma-separated subset.")
    parser.add_argument("--algorithms", default=None, help="Optional comma-separated subset.")
    parser.add_argument("--budgets", default=None, help="Optional comma-separated budget override.")
    args = parser.parse_args()

    data = load_final_config(args.config)
    if args.hash:
        print(config_hash(data))
    if args.print_momab_args:
        envs = args.environments.split(",") if args.environments else None
        algs = args.algorithms.split(",") if args.algorithms else None
        budget_values = [int(v) for v in args.budgets.split(",")] if args.budgets else None
        print(" ".join(build_momab_args(
            config=data,
            results_dir=args.results_dir,
            run_dir=args.run_dir,
            n_jobs=args.n_jobs,
            num_runs=args.num_runs,
            budgets_override=budget_values,
            environments_override=envs,
            algorithms_override=algs,
        )))


if __name__ == "__main__":
    main()
