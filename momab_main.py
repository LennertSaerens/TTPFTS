import argparse
import csv
import os

from joblib import Parallel, delayed
from tqdm import tqdm

from bandits.Posteriors import TPosterior, NormalPosterior, BetaBernoulliPosterior, GammaExponentialPosterior
from bandits.TTPFTS import TTPFTSBandit, CPFTSBandit, RPFTSBandit
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
}


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


def _run_single_ege_experiment(experiment_id, environment, ege_func, max_budget, step):
    """Run a single EGE experiment and return buffered results."""
    environment.reset()
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
                       write=True, step=1, n_jobs=1):
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
                _run_single_ege_experiment(exp_id, environment, ege_func, max_budget, step)
            )
    else:
        batched = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_ege_experiment)(exp_id, environment, ege_func, max_budget, step)
            for exp_id in pending_ids
        )
        all_results = [row for batch in batched for row in batch]

    if results_file is not None and write:
        _write_results(results_file, all_results)


def _run_single_anytime_experiment(experiment_id, environment, algorithm_name, bandit, max_budget, step):
    """Run a single anytime experiment and return buffered results."""
    environment.reset()
    bandit.reset(environment.stds)
    results = []
    for t_step in range(0, max_budget + 1, step):
        arm = bandit.choose_arm()
        reward = environment.pull_arm(arm)
        bandit.learn(arm, reward)
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
                           write=True, step=1):
    alg_objs = {}
    for alg in algorithms:
        if alg == "Uniform":
            alg_objs[alg] = UniformBandit(environment.num_arms, environment.num_objectives)
        elif alg == "PUCB1":
            alg_objs[alg] = PUCB1Bandit(environment.num_arms, environment.num_objectives)
        elif alg == "TTPFTS_GKV":
            alg_objs[alg] = TTPFTSBandit(
                NormalPosterior(environment.num_arms, environment.num_objectives, environment.stds))
        elif alg == "TTPFTS_UNI":
            alg_objs[alg] = TTPFTSBandit(
                TPosterior(environment.num_arms, environment.num_objectives, alpha=-1/2),
                num_warmup_pulls=4)
        elif alg == "CPFTS_UNI":
            alg_objs[alg] = CPFTSBandit(
                TPosterior(environment.num_arms, environment.num_objectives, alpha=-1/2),
                num_warmup_pulls=4)
        elif alg == "CPFTS_GKV":
            alg_objs[alg] = CPFTSBandit(
                NormalPosterior(environment.num_arms, environment.num_objectives, environment.stds))
        elif alg == "RPFTS_UNI":
            alg_objs[alg] = RPFTSBandit(
                TPosterior(environment.num_arms, environment.num_objectives, alpha=-1/2),
                num_warmup_pulls=4)
        elif alg == "RPFTS_GKV":
            alg_objs[alg] = RPFTSBandit(
                NormalPosterior(environment.num_arms, environment.num_objectives, environment.stds))
        elif alg == "TTPFTS_BETA":
            alg_objs[alg] = TTPFTSBandit(
                BetaBernoulliPosterior(environment.num_arms, environment.num_objectives),
                num_warmup_pulls=0)
        elif alg == "TTPFTS_GAMMA":
            alg_objs[alg] = TTPFTSBandit(
                GammaExponentialPosterior(environment.num_arms, environment.num_objectives),
                num_warmup_pulls=0)

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
                _run_single_anytime_experiment(exp_id, environment, algorithm_name, bandit, max_budget, step)
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
    parser.add_argument('--algorithms', required=True,
                        help='Comma-separated list: EGE_SR,EGE_SH,Uniform,PUCB1,TTPFTS_GKV,TTPFTS_UNI,TTPFTS_BETA,TTPFTS_GAMMA')
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
                        choices=['gaussian', 'bernoulli', 'exponential'],
                        help='Reward distribution family (default: gaussian)')
    args = parser.parse_args()

    algorithms = [a.strip() for a in args.algorithms.split(',')]
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
            # N50VS doesn't accept dist parameter
            environment = env_cls()
        budget_list = budgets if budgets else [default_budget] * len(environments)
        max_budget = budget_list[i]
        results_file = os.path.join(
            args.results_dir,
            f"{env_name}_{algorithm_names}_{max_budget}_{args.num_runs}.csv"
        )
        if "EGE_SR" in algorithms:
            run_EGE_experiment(args.num_runs, max_budget, environment, EGE_SR,
                               results_file=results_file, write=not args.no_write,
                               step=args.step, n_jobs=args.n_jobs)
        if "EGE_SH" in algorithms:
            run_EGE_experiment(args.num_runs, max_budget, environment, EGE_SH,
                               results_file=results_file, write=not args.no_write,
                               step=args.step, n_jobs=args.n_jobs)
        anytime_algs = [alg for alg in algorithms if alg in [
            "Uniform", "PUCB1", "TTPFTS_GKV", "TTPFTS_UNI", "TTPFTS_BETA", "TTPFTS_GAMMA",
            "CPFTS_UNI", "CPFTS_GKV", "RPFTS_UNI", "RPFTS_GKV"
        ]]
        if anytime_algs:
            run_anytime_experiment(args.num_runs, max_budget, environment, anytime_algs,
                                   results_file=results_file, write=not args.no_write,
                                   step=args.step)
