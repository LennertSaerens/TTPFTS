import argparse
import csv
import os
import random
import re
import zlib

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from bandits.Posteriors import TPosterior, NormalPosterior, BetaBernoulliPosterior, GammaExponentialPosterior, GammaPoissonPosterior
from bandits.TTPFTS import TTPFTSBandit, CPFTSBandit, RPFTSBandit, OverlapTTPFTSBandit
from bandits.UCB import PUCB1Bandit
from bandits.Uniform import UniformBandit
from bandits.APE import APEAnytimeBandit
from bandits.ege_kone import EGE_SH, EGE_SR
from environments.CovBoost import CovBoost
from environments.EgeExp1 import EgeExp1
from environments.EgeExp2 import EgeExp2
from environments.EgeExp3 import EgeExp3
from environments.EgeExp4 import EgeExp4
from environments.EgeExp5 import EgeExp5
from environments.EgeExp6 import EgeExp6
from environments.EgeExp7 import EgeExp7
from environments.EgeExp8 import EgeExp8
from environments.N50VS import N50VS
from environments.PoissonExp import PoissonExp

from runner.config import RunManifest
from runner.grid import build_evaluation_grid, parse_explicit_budgets
from runner.instance import compute_instance_hash
from runner.schema import SCHEMA_COLUMNS, build_row
from runner.seeding import seed_rngs
from runner.shards import (
    discover_shards,
    merge_shards,
    needs_recompute,
    read_shard,
    shard_is_complete,
    shard_path,
    write_shard_atomic,
)

# ---------------------------------------------------------------------------
# Legacy (pre-Phase-5) CSV schema, kept only for the backward-compatible
# helpers below (``run_anytime_experiment`` / ``run_EGE_experiment`` /
# ``_completed_experiments``), which remain importable for existing callers
# such as benchmark_ape.py. The Phase 5 CLI / sharded engine further down
# never uses this schema or these helpers.
# ---------------------------------------------------------------------------
CSV_HEADER = ["algorithm", "experiment", "timestep", "bernoulli", "jaccard", "misidentification"]

ENVIRONMENTS = {
    "EgeExp1": (EgeExp1, 5000),
    "EgeExp2": (EgeExp2, 5000),
    "EgeExp3": (EgeExp3, 5000),
    "EgeExp4": (EgeExp4, 5000),
    "EgeExp5": (EgeExp5, 5000),
    "EgeExp6": (EgeExp6, 5000),
    "EgeExp7": (EgeExp7, 5000),
    "EgeExp8": (EgeExp8, 5000),
    "N50VS": (N50VS, 5000),
    "CovBoost": (CovBoost, 5000),
    "PoissonExp": (PoissonExp, 5000),
}

EGE_ALGORITHMS = {
    "EGE_SR": EGE_SR,
    "EGE_SH": EGE_SH,
}

# Canonical (unambiguous) anytime algorithm keys, plus explicit legacy
# aliases retained for backward compatibility with old result filenames /
# scripts (see ``_make_bandit`` and ``ALGORITHM_ALIASES`` below). Legacy
# aliases intentionally keep their *original* hyperparameters (e.g.
# TTPFTS_UNI's alpha=-1/2 improper prior), which differ from the new
# unambiguous TTPFTS_UNKNOWN_VAR (alpha=0 reference/Jeffreys prior) -- they
# are not silently redefinitions of one another.
ANYTIME_ALGORITHMS = [
    "Uniform", "PUCB1", "APE",
    "TTPFTS_KNOWN_VAR", "TTPFTS_UNKNOWN_VAR",
    "TTPFTS_GKV", "OverlapTTPFTS_GKV", "TTPFTS_UNI",
    "CPFTS_UNI", "CPFTS_GKV", "RPFTS_UNI", "RPFTS_GKV",
    "TTPFTS_BETA", "TTPFTS_GAMMA", "TTPFTS_POISSON",
]

# Display names may carry a "_p<value>" suffix injected by the CLI so that
# different p values land in different result files.
_P_SUFFIX = re.compile(r"_p\d+(?:\.\d+)?$")


def _base_algorithm_name(display_name):
    """Strip the CLI-injected '_p<value>' suffix to recover the algorithm key."""
    return _P_SUFFIX.sub("", display_name)


def _make_bandit(base_name, environment, p):
    """Construct an anytime bandit from its algorithm key, or None if unknown.

    ``TTPFTS_KNOWN_VAR``/``TTPFTS_UNKNOWN_VAR`` are the unambiguous Phase 5
    algorithms: matched 4-pulls-per-arm warmup and a fixed rho/p=0.5,
    differing only in whether the reward variance is treated as known
    (``NormalPosterior`` seeded with the environment's true std) or unknown
    (``TPosterior`` with the alpha=0 reference/Jeffreys prior).
    """
    num_arms, num_objectives = environment.num_arms, environment.num_objectives
    if base_name == "Uniform":
        return UniformBandit(num_arms, num_objectives)
    if base_name == "PUCB1":
        return PUCB1Bandit(num_arms, num_objectives)
    if base_name == "APE":
        sigma = float(np.max(np.asarray(environment.stds, dtype=float)))
        return APEAnytimeBandit(num_arms, num_objectives, delta=0.1, sigma=sigma,
                                epsilon_1=0.0, epsilon_2=0.0)
    if base_name == "TTPFTS_KNOWN_VAR":
        return TTPFTSBandit(NormalPosterior(num_arms, num_objectives, environment.stds),
                            p=0.5, num_warmup_pulls=4)
    if base_name == "TTPFTS_UNKNOWN_VAR":
        return TTPFTSBandit(TPosterior(num_arms, num_objectives, alpha=0.0),
                            p=0.5, num_warmup_pulls=4)
    # --- legacy aliases: original hyperparameters, kept for backward compat ---
    if base_name == "TTPFTS_GKV":
        return TTPFTSBandit(NormalPosterior(num_arms, num_objectives, environment.stds), p=p)
    if base_name == "OverlapTTPFTS_GKV":
        return OverlapTTPFTSBandit(NormalPosterior(num_arms, num_objectives, environment.stds), p=p)
    if base_name == "TTPFTS_UNI":
        return TTPFTSBandit(TPosterior(num_arms, num_objectives, alpha=-1 / 2), num_warmup_pulls=4)
    if base_name == "CPFTS_UNI":
        return CPFTSBandit(TPosterior(num_arms, num_objectives, alpha=-1 / 2), num_warmup_pulls=4)
    if base_name == "CPFTS_GKV":
        return CPFTSBandit(NormalPosterior(num_arms, num_objectives, environment.stds))
    if base_name == "RPFTS_UNI":
        return RPFTSBandit(TPosterior(num_arms, num_objectives, alpha=-1 / 2), num_warmup_pulls=4)
    if base_name == "RPFTS_GKV":
        return RPFTSBandit(NormalPosterior(num_arms, num_objectives, environment.stds))
    if base_name == "TTPFTS_BETA":
        return TTPFTSBandit(BetaBernoulliPosterior(num_arms, num_objectives), num_warmup_pulls=0)
    if base_name == "TTPFTS_GAMMA":
        return TTPFTSBandit(GammaExponentialPosterior(num_arms, num_objectives), num_warmup_pulls=1)
    if base_name == "TTPFTS_POISSON":
        return TTPFTSBandit(GammaPoissonPosterior(num_arms, num_objectives), num_warmup_pulls=1)
    return None


def _bandit_metadata(bandit):
    """Return (posterior_model, rho) describing a constructed bandit for the schema."""
    posterior = getattr(bandit, "posterior", None)
    posterior_model = type(posterior).__name__ if posterior is not None else None
    rho = getattr(bandit, "p", None)
    if rho is None:
        rho = getattr(bandit, "rho", None)
    return posterior_model, rho


def _construct_environment(env_cls, dist):
    try:
        return env_cls(dist=dist)
    except TypeError:
        # N50VS and PoissonExp don't accept a dist parameter
        return env_cls()


# ---------------------------------------------------------------------------
# Phase 5 sharded engine: exact-budget pulls, endpoint-inclusive grids shared
# across every algorithm in an environment, per-(env, algorithm, repetition
# [, budget]) deterministic seeding, and atomic per-shard persistence.
# ---------------------------------------------------------------------------

def _run_repetition_anytime(environment, env_name, dist, algorithm_name, bandit,
                            max_budget, grid, master_seed, repetition):
    """Run one repetition of an anytime algorithm for exactly ``max_budget`` pulls.

    Logs the prior recommendation at arm_pull=0 (before any pull), pulls and
    updates the bandit once per t=1..max_budget (exactly max_budget pulls,
    regardless of how sparse ``grid`` is), and logs a row for every t in
    ``grid`` (which always includes 0 and max_budget).
    """
    env_tag = f"{env_name}|{dist}"
    # Environment instance seed is algorithm-independent (common random
    # numbers): every algorithm sees the same frozen arms for this repetition.
    seed_rngs(master_seed, env_tag, "env", repetition)
    environment.reset()
    instance_hash = compute_instance_hash(environment)
    true_pareto = list(environment.get_top_arms())

    seed_rngs(master_seed, env_tag, algorithm_name, repetition)
    bandit.reset(environment.stds)
    posterior_model, rho = _bandit_metadata(bandit)

    grid_set = set(grid)
    rows = []

    def _log(arm_pull):
        rec = list(bandit.get_top_arms())
        rows.append(build_row(
            environment=env_name, instance_hash=instance_hash, algorithm=algorithm_name,
            posterior_model=posterior_model or "", repetition=repetition, arm_pull=arm_pull,
            max_budget=max_budget, evaluation_grid=grid, rho=rho, master_seed=master_seed,
            recommendation=rec, true_pareto=true_pareto,
            bernoulli=environment.bernoulli_metric(rec),
            jaccard=environment.jaccard_metric(rec),
            misidentification=environment.mis_id_metric(rec),
        ))

    if 0 in grid_set:
        _log(0)
    for t in range(1, max_budget + 1):
        arm = bandit.choose_arm()
        reward = environment.pull_arm(arm)
        bandit.learn(arm, reward)
        if t in grid_set:
            _log(t)
    return rows


def _run_repetition_ege(environment, env_name, dist, algorithm_name, ege_func,
                        max_budget, grid, master_seed, repetition):
    """Run one repetition of a fixed-budget EGE algorithm across the whole grid.

    Each evaluated budget seeds independently from
    ``(master_seed, env_tag, algorithm, repetition, budget)`` so that a
    shared budget produces byte-identical rows regardless of grid density,
    execution order, or resume.
    """
    env_tag = f"{env_name}|{dist}"
    seed_rngs(master_seed, env_tag, "env", repetition)
    environment.reset()
    instance_hash = compute_instance_hash(environment)
    true_pareto = list(environment.get_top_arms())

    rows = []
    for budget in grid:
        seed_rngs(master_seed, env_tag, algorithm_name, repetition, "budget", budget)
        recommended_arms = list(ege_func(
            T=budget, K=environment.num_arms, D=environment.num_objectives,
            environment=environment,
        ))
        rows.append(build_row(
            environment=env_name, instance_hash=instance_hash, algorithm=algorithm_name,
            posterior_model="", repetition=repetition, arm_pull=budget, max_budget=max_budget,
            evaluation_grid=grid, rho=None, master_seed=master_seed,
            recommendation=recommended_arms, true_pareto=true_pareto,
            bernoulli=environment.bernoulli_metric(recommended_arms),
            jaccard=environment.jaccard_metric(recommended_arms),
            misidentification=environment.mis_id_metric(recommended_arms),
        ))
    return rows


def run_algorithm_sharded(run_dir, environment, env_name, dist, algorithm_name, base_name,
                          max_budget, grid, num_runs, master_seed, n_jobs=1, p=0.5,
                          write=True, is_ege=False, ege_func=None):
    """Compute (and, unless ``write`` is False, atomically persist) every
    repetition of one algorithm in one environment, skipping repetitions
    whose shard is already complete for the current grid."""

    def _compute(repetition):
        if is_ege:
            return _run_repetition_ege(environment, env_name, dist, algorithm_name, ege_func,
                                       max_budget, grid, master_seed, repetition)
        bandit = _make_bandit(base_name, environment, p)
        return _run_repetition_anytime(environment, env_name, dist, algorithm_name, bandit,
                                       max_budget, grid, master_seed, repetition)

    if write:
        pending = [r for r in range(num_runs)
                  if needs_recompute(shard_path(run_dir, env_name, algorithm_name, r), grid)]
    else:
        pending = list(range(num_runs))

    if not pending:
        print(f"  {algorithm_name}: all {num_runs} repetitions already complete, skipping.")
        return []

    if n_jobs == 1:
        results = [(r, _compute(r))
                  for r in tqdm(pending, desc=f"Running {algorithm_name}", unit="repetition")]
    else:
        computed = Parallel(n_jobs=n_jobs)(delayed(_compute)(r) for r in pending)
        results = list(zip(pending, computed))

    if write:
        for repetition, rows in results:
            write_shard_atomic(shard_path(run_dir, env_name, algorithm_name, repetition), rows)

    return results


# ---------------------------------------------------------------------------
# Legacy (pre-Phase-5) API: unsafe shared-append CSV with in-band resume via
# ``_completed_experiments``. Kept only for existing callers (e.g.
# benchmark_ape.py) that import these names directly; the Phase 5 CLI below
# never calls them.
# ---------------------------------------------------------------------------

def _seed_globals(base_seed, tag, experiment_id):
    """Seed numpy and stdlib RNGs deterministically from (seed, tag, run id)."""
    if base_seed is None:
        return
    seed_seq = np.random.SeedSequence([int(base_seed), zlib.crc32(tag.encode()), int(experiment_id)])
    np_seed, py_seed = seed_seq.generate_state(2, dtype=np.uint32)
    np.random.seed(int(np_seed))
    random.seed(int(py_seed))


def _seeded_reset(environment, base_seed, run_tag, experiment_id, algorithm_name):
    _seed_globals(base_seed, f"{run_tag}|env", experiment_id)
    environment.reset()
    _seed_globals(base_seed, f"{run_tag}|{algorithm_name}", experiment_id)


def _completed_experiments(results_file):
    """Return set of (algorithm, experiment_id) pairs already in the results file."""
    completed = set()
    if not os.path.exists(results_file):
        return completed
    with open(results_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and header[0] == "algorithm":
            pass  # skip header
        else:
            # No header — rewind
            f.seek(0)
            reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                completed.add((row[0], int(row[1])))
    return completed


def _run_single_ege_experiment(experiment_id, environment, ege_func, max_budget, step,
                               seed=None, run_tag=""):
    """Run a single EGE experiment and return buffered results (legacy schema)."""
    _seeded_reset(environment, seed, run_tag, experiment_id, ege_func.__name__)
    results = []
    for budget in range(0, max_budget + 1, step):
        recommended_arms = ege_func(
            T=budget,
            K=environment.num_arms,
            D=environment.num_objectives,
            environment=environment
        )
        results.append([
            ege_func.__name__,
            experiment_id,
            budget,
            environment.bernoulli_metric(recommended_arms),
            environment.jaccard_metric(recommended_arms),
            environment.mis_id_metric(recommended_arms),
        ])
    return results


def run_EGE_experiment(num_runs, max_budget, environment, ege_func, results_file=None,
                       write=True, step=1, n_jobs=1, seed=None, run_tag=""):
    """Legacy (pre-Phase-5) EGE runner. Prefer ``run_algorithm_sharded``."""
    completed = _completed_experiments(results_file) if results_file and write else set()
    pending_ids = [i for i in range(num_runs)
                   if (ege_func.__name__, i) not in completed]

    if not pending_ids:
        print(f"  {ege_func.__name__}: all {num_runs} runs already completed, skipping.")
        return

    if n_jobs == 1:
        all_results = []
        for exp_id in tqdm(pending_ids, desc=f"Running {ege_func.__name__}", unit="experiment"):
            all_results.extend(
                _run_single_ege_experiment(exp_id, environment, ege_func, max_budget, step,
                                           seed=seed, run_tag=run_tag)
            )
    else:
        batched = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_ege_experiment)(exp_id, environment, ege_func, max_budget, step,
                                               seed, run_tag)
            for exp_id in pending_ids
        )
        all_results = [row for batch in batched for row in batch]

    if results_file is not None and write:
        _write_results(results_file, all_results)


def _run_single_anytime_experiment(experiment_id, environment, algorithm_name, bandit, max_budget,
                                   step, seed=None, run_tag=""):
    """Run a single anytime experiment and return buffered results (legacy schema).

    One arm is pulled at every timestep; ``step`` only controls how often the
    recommendation is evaluated and logged.
    """
    _seeded_reset(environment, seed, run_tag, experiment_id, algorithm_name)
    bandit.reset(environment.stds)
    results = []
    for t_step in range(0, max_budget + 1):
        arm = bandit.choose_arm()
        reward = environment.pull_arm(arm)
        bandit.learn(arm, reward)
        if t_step % step:
            continue
        recommended_arms = bandit.get_top_arms()
        results.append([
            algorithm_name,
            experiment_id,
            t_step,
            environment.bernoulli_metric(recommended_arms),
            environment.jaccard_metric(recommended_arms),
            environment.mis_id_metric(recommended_arms),
        ])
    return results


def run_anytime_experiment(num_runs, max_budget, environment, algorithms, results_file=None,
                           write=True, step=1, p=0.5, seed=None, run_tag=""):
    """Legacy (pre-Phase-5) anytime runner. Prefer ``run_algorithm_sharded``."""
    alg_objs = {}
    for alg in algorithms:
        bandit = _make_bandit(_base_algorithm_name(alg), environment, p)
        if bandit is None:
            raise ValueError(
                f"Unknown anytime algorithm {alg!r}. Available: {', '.join(ANYTIME_ALGORITHMS)}"
            )
        alg_objs[alg] = bandit

    completed = _completed_experiments(results_file) if results_file and write else set()

    for algorithm_name, bandit in alg_objs.items():
        pending_ids = [i for i in range(num_runs)
                       if (algorithm_name, i) not in completed]

        if not pending_ids:
            print(f"  {algorithm_name}: all {num_runs} runs already completed, skipping.")
            continue

        # Anytime algorithms are stateful so parallel execution requires separate bandit copies
        all_results = []
        for exp_id in tqdm(pending_ids, desc=f"Running {algorithm_name}", unit="experiment"):
            all_results.extend(
                _run_single_anytime_experiment(exp_id, environment, algorithm_name, bandit,
                                               max_budget, step, seed=seed, run_tag=run_tag)
            )

        if results_file is not None and write:
            _write_results(results_file, all_results)


def _write_results(results_file, rows):
    """Write results to CSV, adding header if file is new (legacy schema)."""
    file_exists = os.path.exists(results_file) and os.path.getsize(results_file) > 0
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOMAB Benchmarking Framework")
    parser.add_argument('--algorithms', required=True,
                        help='Comma-separated list: EGE_SR,EGE_SH,Uniform,PUCB1,'
                             'TTPFTS_KNOWN_VAR,TTPFTS_UNKNOWN_VAR,TTPFTS_GKV,TTPFTS_UNI,'
                             'TTPFTS_BETA,TTPFTS_GAMMA,TTPFTS_POISSON')
    parser.add_argument('--environments', required=True,
                        help='Comma-separated environments: EgeExp1,EgeExp2,CovBoost...')
    parser.add_argument('--budgets', required=False, default=None,
                        help='Comma-separated budgets, overrides default for environment.')
    parser.add_argument('--num_runs', type=int, default=100)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--run_dir', default=None,
                        help='Directory holding atomic per-repetition shards + manifest '
                             '(default: <results_dir>/run). Reusing the same --run_dir across '
                             'invocations resumes incomplete/missing repetitions only.')
    parser.add_argument('--step', type=int, default=1,
                        help='Stride (in arm pulls) between evaluated/logged budgets; 0 and the '
                             'max budget are always included even if not a multiple of step.')
    parser.add_argument('--eval_points', type=int, default=None,
                        help='Number of evaluation intervals per environment, overriding --step. '
                             'Produces eval_points+1 points at round(k*B/eval_points), so every '
                             'environment is evaluated at the same normalized budgets k/eval_points '
                             'even when the raw maxima differ.')
    parser.add_argument('--eval_budgets', default=None,
                        help='Optional comma-separated explicit evaluation budgets, overriding '
                             '--step entirely. Must lie within [0, max_budget] for every '
                             'requested environment.')
    parser.add_argument('--no_write', action='store_true')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs per algorithm (default: 1)')
    parser.add_argument('--p', type=float, default=0.5,
                        help='p parameter for legacy TTPFTS_GKV/OverlapTTPFTS_GKV variants '
                             '(default: 0.5); TTPFTS_KNOWN_VAR/TTPFTS_UNKNOWN_VAR always use 0.5.')
    parser.add_argument('--dist', type=str, default='gaussian',
                        choices=['gaussian', 'bernoulli', 'exponential', 'poisson'],
                        help='Reward distribution family (default: gaussian)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Master RNG seed; every repetition (and, for EGE, every evaluated '
                             'budget) is seeded independently from '
                             '(seed, environment, dist, algorithm, repetition[, budget]) '
                             '(default: 0)')
    args = parser.parse_args()

    requested = [a.strip() for a in args.algorithms.split(',')]
    unknown_algs = [a for a in requested
                    if a not in EGE_ALGORITHMS and a not in ANYTIME_ALGORITHMS]
    if unknown_algs:
        parser.error(
            f"Unknown algorithm(s): {', '.join(unknown_algs)}. "
            f"Available: {', '.join(list(EGE_ALGORITHMS) + ANYTIME_ALGORITHMS)}"
        )

    environments = [e.strip() for e in args.environments.split(',')]
    unknown_envs = [e for e in environments if e not in ENVIRONMENTS]
    if unknown_envs:
        parser.error(
            f"Unknown environment(s): {', '.join(unknown_envs)}. "
            f"Available: {', '.join(ENVIRONMENTS)}"
        )

    # Resolve one budget per environment.
    if not args.budgets:
        budget_list = [ENVIRONMENTS[e][1] for e in environments]
    else:
        budgets = [int(b.strip()) for b in args.budgets.split(',')]
        if len(budgets) == 1:
            budget_list = budgets * len(environments)
        elif len(budgets) == len(environments):
            budget_list = budgets
        else:
            parser.error(
                f"--budgets has {len(budgets)} value(s) but --environments has "
                f"{len(environments)}; pass one budget, one per environment, or none."
            )

    try:
        explicit_budgets = parse_explicit_budgets(args.eval_budgets)
    except ValueError as exc:
        parser.error(str(exc))
    if explicit_budgets is not None and args.eval_points is not None:
        parser.error("--eval_budgets and --eval_points are mutually exclusive.")
    if args.eval_points is not None and args.eval_points < 1:
        parser.error("--eval_points must be a positive integer.")

    algorithms = list(requested)
    # Inject p value into algorithm name for legacy TTPFTS_GKV to separate results
    for idx, alg in enumerate(algorithms):
        if alg == "TTPFTS_GKV":
            algorithms[idx] = f"TTPFTS_GKV_p{args.p}"
        elif alg == "OverlapTTPFTS_GKV":
            algorithms[idx] = f"OverlapTTPFTS_GKV_p{args.p}"

    os.makedirs(args.results_dir, exist_ok=True)
    run_dir = args.run_dir or os.path.join(args.results_dir, "run")
    os.makedirs(run_dir, exist_ok=True)

    evaluation_grids = {}
    for i, env_name in enumerate(environments):
        max_budget = budget_list[i]
        try:
            evaluation_grids[env_name] = build_evaluation_grid(
                max_budget, step=args.step, explicit_budgets=explicit_budgets,
                num_intervals=args.eval_points)
        except ValueError as exc:
            parser.error(f"{env_name}: {exc}")

    manifest = RunManifest(
        master_seed=args.seed,
        dist=args.dist,
        environments=environments,
        algorithms=algorithms,
        num_runs=args.num_runs,
        step=args.step,
        explicit_budgets=explicit_budgets,
        eval_points=args.eval_points,
        budgets=dict(zip(environments, budget_list)),
        evaluation_grids=evaluation_grids,
        n_jobs=args.n_jobs,
        rho=args.p,
        results_dir=args.results_dir,
        run_dir=run_dir,
    )
    if not args.no_write:
        manifest.write(os.path.join(run_dir, "manifest.json"))

    for i, env_name in enumerate(environments):
        env_cls, _ = ENVIRONMENTS[env_name]
        environment = _construct_environment(env_cls, args.dist)
        max_budget = budget_list[i]
        grid = evaluation_grids[env_name]

        print(f"== {env_name} (dist={args.dist}, budget={max_budget}, "
              f"grid={len(grid)} points) ==")

        for ege_name, ege_func in EGE_ALGORITHMS.items():
            if ege_name in algorithms:
                run_algorithm_sharded(run_dir, environment, env_name, args.dist, ege_name,
                                      ege_name, max_budget, grid, args.num_runs, args.seed,
                                      n_jobs=args.n_jobs, write=not args.no_write,
                                      is_ege=True, ege_func=ege_func)

        anytime_algs = [alg for alg in algorithms
                        if _base_algorithm_name(alg) in ANYTIME_ALGORITHMS]
        for alg in anytime_algs:
            run_algorithm_sharded(run_dir, environment, env_name, args.dist, alg,
                                  _base_algorithm_name(alg), max_budget, grid, args.num_runs,
                                  args.seed, n_jobs=args.n_jobs, p=args.p,
                                  write=not args.no_write)

        if not args.no_write:
            output_path = os.path.join(args.results_dir, f"{env_name}.csv")
            expected_grids = {alg: grid for alg in algorithms}
            n_rows = merge_shards(run_dir, env_name, output_path,
                                  expected_grids=expected_grids, validate_metrics=True)
            print(f"  merged {n_rows} rows -> {output_path}")
