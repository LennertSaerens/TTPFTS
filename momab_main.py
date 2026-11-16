import argparse
import csv
import os
import random
import zlib

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from bandits.APE import APEAnytimeBandit
from bandits.APE_UV import APEUnknownVarianceBandit
from bandits.Posteriors import TPosterior, NormalPosterior, BetaBernoulliPosterior, GammaExponentialPosterior, GammaPoissonPosterior
from bandits.TTPFTS import TTPFTSBandit, CPFTSBandit, RPFTSBandit, OverlapTTPFTSBandit
from bandits.UCB import PUCB1Bandit
from bandits.Uniform import UniformBandit
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

CSV_HEADER = ["algorithm", "experiment", "timestep", "bernoulli", "jaccard", "misidentification"]

ENVIRONMENTS = {
    "EgeExp1": (EgeExp1, 15218),
    "EgeExp2": (EgeExp2, 2544),
    "EgeExp3": (EgeExp3, 685036),
    "EgeExp4": (EgeExp4, 15977),
    "EgeExp5": (EgeExp5, 197373),
    "EgeExp6": (EgeExp6, 51966),
    "EgeExp7": (EgeExp7, 7000),
    "EgeExp8": (EgeExp8, 237958),
    "N50VS": (N50VS, 5000),
    "CovBoost": (CovBoost, 5000),
    "PoissonExp": (PoissonExp, 5000),
}

EGE_ALGORITHMS = {
    "EGE_SR": EGE_SR,
    "EGE_SH": EGE_SH,
}

ANYTIME_ALGORITHMS = [
    "Uniform", "PUCB1", "APE", "APE_UV",
    "TTPFTS_KNOWN_VAR", "TTPFTS_UNKNOWN_VAR",
    "TTPFTS_GKV", "OverlapTTPFTS_GKV", "TTPFTS_UNI",
    "CPFTS_UNI", "CPFTS_GKV", "RPFTS_UNI", "RPFTS_GKV",
    "TTPFTS_BETA", "TTPFTS_GAMMA", "TTPFTS_POISSON",
]


def _seed_rngs(master_seed, *parts):
    """Derive independent deterministic numpy/stdlib seeds when master_seed is set."""
    if master_seed is None:
        return None
    entropy = [int(master_seed) & 0xFFFFFFFF]
    for part in parts:
        entropy.append(zlib.crc32(str(part).encode("utf-8")) & 0xFFFFFFFF)
    seed_seq = np.random.SeedSequence(entropy)
    np_seed, py_seed = seed_seq.generate_state(2, dtype=np.uint32)
    np.random.seed(int(np_seed))
    random.seed(int(py_seed))
    return int(np_seed)


def _eval_points(max_budget, step):
    """Return sorted endpoint-inclusive timestep checkpoints in [0, max_budget]."""
    points = list(range(0, max_budget + 1, step))
    if not points or points[-1] != max_budget:
        points.append(max_budget)
    return points


def _make_bandit(alg, environment, p=0.5):
    """Construct an anytime bandit instance from its CLI key."""
    num_arms, num_objectives = environment.num_arms, environment.num_objectives
    if alg == "Uniform":
        return UniformBandit(num_arms, num_objectives)
    if alg == "PUCB1":
        return PUCB1Bandit(num_arms, num_objectives)
    if alg == "APE":
        sigma = float(np.max(np.asarray(environment.stds, dtype=float)))
        return APEAnytimeBandit(
            num_arms, num_objectives, delta=0.1, sigma=sigma, epsilon_1=0.0, epsilon_2=0.0
        )
    if alg == "APE_UV":
        return APEUnknownVarianceBandit(num_arms, num_objectives, delta=0.1, num_warmup_pulls=4)
    if alg == "TTPFTS_KNOWN_VAR":
        return TTPFTSBandit(
            NormalPosterior(num_arms, num_objectives, environment.stds),
            p=p, num_warmup_pulls=4,
        )
    if alg == "TTPFTS_UNKNOWN_VAR":
        return TTPFTSBandit(
            TPosterior(num_arms, num_objectives, alpha=0.0),
            p=p, num_warmup_pulls=4,
        )
    if alg == "TTPFTS_GKV":
        return TTPFTSBandit(
            NormalPosterior(num_arms, num_objectives, environment.stds), p=p
        )
    if alg == "OverlapTTPFTS_GKV":
        return OverlapTTPFTSBandit(
            NormalPosterior(num_arms, num_objectives, environment.stds), p=p
        )
    if alg == "TTPFTS_UNI":
        return TTPFTSBandit(
            TPosterior(num_arms, num_objectives, alpha=-1 / 2), num_warmup_pulls=4
        )
    if alg == "CPFTS_UNI":
        return CPFTSBandit(
            TPosterior(num_arms, num_objectives, alpha=-1 / 2), num_warmup_pulls=4
        )
    if alg == "CPFTS_GKV":
        return CPFTSBandit(
            NormalPosterior(num_arms, num_objectives, environment.stds)
        )
    if alg == "RPFTS_UNI":
        return RPFTSBandit(
            TPosterior(num_arms, num_objectives, alpha=-1 / 2), num_warmup_pulls=4
        )
    if alg == "RPFTS_GKV":
        return RPFTSBandit(
            NormalPosterior(num_arms, num_objectives, environment.stds)
        )
    if alg == "TTPFTS_BETA":
        return TTPFTSBandit(
            BetaBernoulliPosterior(num_arms, num_objectives), num_warmup_pulls=0
        )
    if alg == "TTPFTS_GAMMA":
        return TTPFTSBandit(
            GammaExponentialPosterior(num_arms, num_objectives), num_warmup_pulls=1
        )
    if alg == "TTPFTS_POISSON":
        return TTPFTSBandit(
            GammaPoissonPosterior(num_arms, num_objectives), num_warmup_pulls=1
        )
    raise ValueError(f"Unknown anytime algorithm: {alg}")


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
                               env_name="", seed=None):
    """Run a single EGE experiment and return buffered results."""
    _seed_rngs(seed, env_name, "env", experiment_id)
    environment.reset()
    results = []
    for budget in _eval_points(max_budget, step):
        _seed_rngs(seed, env_name, ege_func.__name__, experiment_id, "budget", budget)
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
                       write=True, step=1, n_jobs=1, seed=None, env_name=""):
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
                                           env_name=env_name, seed=seed)
            )
    else:
        batched = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_ege_experiment)(
                exp_id, environment, ege_func, max_budget, step, env_name=env_name, seed=seed
            )
            for exp_id in pending_ids
        )
        all_results = [row for batch in batched for row in batch]

    if results_file is not None and write:
        _write_results(results_file, all_results)


def _run_single_anytime_experiment(experiment_id, environment, algorithm_name, bandit,
                                   max_budget, step, env_name="", seed=None):
    """Run one anytime experiment for exactly ``max_budget`` reward pulls."""
    _seed_rngs(seed, env_name, "env", experiment_id)
    environment.reset()
    _seed_rngs(seed, env_name, algorithm_name, experiment_id)
    bandit.reset(environment.stds)

    eval_set = set(_eval_points(max_budget, step))
    results = []

    def _record(t_step):
        recommended_arms = bandit.get_top_arms()
        results.append([
            algorithm_name,
            experiment_id,
            t_step,
            environment.bernoulli_metric(recommended_arms),
            environment.jaccard_metric(recommended_arms),
            environment.mis_id_metric(recommended_arms),
        ])

    if 0 in eval_set:
        _record(0)
    for t_step in range(1, max_budget + 1):
        arm = bandit.choose_arm()
        reward = environment.pull_arm(arm)
        bandit.learn(arm, reward)
        if t_step in eval_set:
            _record(t_step)
    return results


def run_anytime_experiment(num_runs, max_budget, environment, algorithms, results_file=None,
                           write=True, step=1, p=0.5, seed=None, env_name=""):
    alg_objs = {alg: _make_bandit(alg, environment, p=p) for alg in algorithms}

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
                _run_single_anytime_experiment(
                    exp_id, environment, algorithm_name, bandit, max_budget, step,
                    env_name=env_name, seed=seed,
                )
            )

        if results_file is not None and write:
            _write_results(results_file, all_results)


def _write_results(results_file, rows):
    """Write results to CSV, adding header if file is new."""
    file_exists = os.path.exists(results_file) and os.path.getsize(results_file) > 0
    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MOMAB Benchmarking Framework")
    parser.add_argument(
        '--algorithms', required=True,
        help='Comma-separated list: ' + ",".join([*EGE_ALGORITHMS.keys(), *ANYTIME_ALGORITHMS]),
    )
    parser.add_argument('--environments', required=True,
                        help='Comma-separated environments: EgeExp1,EgeExp2,CovBoost...')
    parser.add_argument('--budgets', required=False, default=None,
                        help='Comma-separated budgets, overrides default for environment.')
    parser.add_argument('--num_runs', type=int, default=100)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--no_write', action='store_true')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs for EGE experiments (default: 1)')
    parser.add_argument('--dist', type=str, default='gaussian',
                        choices=['gaussian', 'bernoulli', 'exponential', 'poisson'],
                        help='Reward distribution family (default: gaussian)')
    parser.add_argument('--p', type=float, default=0.5,
                        help='Top-front selection probability p for TTPFTS variants (default: 0.5)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional master RNG seed for reproducible runs')
    args = parser.parse_args()

    if args.step < 1:
        parser.error("--step must be a positive integer.")

    algorithms = [a.strip() for a in args.algorithms.split(',')]
    valid_algorithms = set(EGE_ALGORITHMS) | set(ANYTIME_ALGORITHMS)
    unknown = [a for a in algorithms if a not in valid_algorithms]
    if unknown:
        parser.error(f"Unknown algorithms: {unknown}")

    algorithm_names = "_".join(algorithms)
    environments = [e.strip() for e in args.environments.split(',')]
    budgets = None
    if args.budgets:
        budgets = [int(b.strip()) for b in args.budgets.split(',')]

    os.makedirs(args.results_dir, exist_ok=True)

    for i, env_name in enumerate(environments):
        if env_name not in ENVIRONMENTS:
            print(f"Warning: Environment {env_name} unknown.")
            continue
        env_cls, default_budget = ENVIRONMENTS[env_name]
        try:
            environment = env_cls(dist=args.dist)
        except TypeError:
            # N50VS and PoissonExp don't accept dist parameter
            environment = env_cls()
        budget_list = budgets if budgets else [default_budget] * len(environments)
        max_budget = budget_list[i]
        results_file = os.path.join(
            args.results_dir,
            f"{env_name}_{algorithm_names}_{max_budget}_{args.num_runs}.csv"
        )
        for ege_name, ege_func in EGE_ALGORITHMS.items():
            if ege_name in algorithms:
                run_EGE_experiment(args.num_runs, max_budget, environment, ege_func,
                                   results_file=results_file, write=not args.no_write,
                                   step=args.step, n_jobs=args.n_jobs,
                                   seed=args.seed, env_name=f"{env_name}|{args.dist}")
        anytime_algs = [alg for alg in algorithms if alg in ANYTIME_ALGORITHMS]
        if anytime_algs:
            run_anytime_experiment(args.num_runs, max_budget, environment, anytime_algs,
                                   results_file=results_file, write=not args.no_write,
                                   step=args.step, p=args.p,
                                   seed=args.seed, env_name=f"{env_name}|{args.dist}")
