import argparse
from tqdm import tqdm
import os
from bandits.ege_kone import EGE_SH, EGE_SR
from bandits.TTPFTS import TTPFTSBandit
from bandits.UCB import PUCB1Bandit
from bandits.Uniform import UniformBandit
from bandits.Posteriors import NormalIGPosterior, TPosterior, NormalPosterior
from environments import CovBoost, EgeExp1, EgeExp2, EgeExp3, EgeExp4, EgeExp5, EgeExp6, EgeExp7, EgeExp8, N50VS

# Mapping strings to classes/functions for argparse compatibility
ENVIRONMENTS = {
    "EgeExp1": (EgeExp1.EgeExp1, 5000),
    "EgeExp2": (EgeExp2.EgeExp2, 5000),
    "EgeExp3": (EgeExp3.EgeExp3, 5000),
    "EgeExp4": (EgeExp4.EgeExp4, 5000),
    "EgeExp5": (EgeExp5.EgeExp5, 5000),
    "EgeExp6": (EgeExp6.EgeExp6, 5000),
    "EgeExp7": (EgeExp7.EgeExp7, 5000),
    "EgeExp8": (EgeExp8.EgeExp8, 5000),
    "N50VS": (N50VS.N50VS, 5000),
    "CovBoost": (CovBoost.CovBoost, 5000),
}


def run_EGE_experiment(num_runs, max_budget, environment, ege_func, results_file=None, write=True, step=1):
    for experiment in tqdm(range(num_runs), desc=f"Running {ege_func.__name__} experiments", unit="experiment"):
        environment.reset()
        for budget in range(0, max_budget + 1, step):
            recommended_arms = ege_func(
                T=budget,
                K=environment.num_arms,
                D=environment.num_objectives,
                environment=environment
            )
            bernoulli_metric = environment.bernoulli_metric(recommended_arms)
            jaccard_metric = environment.jaccard_metric(recommended_arms)
            mis_id_metric = environment.mis_id_metric(recommended_arms)
            if results_file is not None and write:
                with open(results_file, "a") as file:
                    file.write(
                        f"{ege_func.__name__},{experiment},{budget},{bernoulli_metric},{jaccard_metric},{mis_id_metric}\n")


def run_anytime_experiment(num_runs, max_budget, environment, algorithms, results_file=None, write=True, step=1):
    alg_objs = {}
    for alg in algorithms:
        if alg == "Uniform":
            alg_objs[alg] = UniformBandit(environment.num_arms, environment.num_objectives)
        elif alg == "PUCB1":
            alg_objs[alg] = PUCB1Bandit(environment.num_arms, environment.num_objectives)
        elif alg == "TTPFTS":
            alg_objs[alg] = TTPFTSBandit(NormalPosterior(environment.num_arms, environment.num_objectives, environment.stds))
        else:
            continue
    for algorithm_name, bandit in alg_objs.items():
        for experiment in tqdm(range(num_runs), desc=f"Running {algorithm_name} experiments", unit="experiment"):
            bandit.reset()
            environment.reset()
            for t in range(0, max_budget + 1, step):
                arm = bandit.choose_arm()
                reward = environment.pull_arm(arm)
                bandit.learn(arm, reward)
                recommended_arms = bandit.get_top_arms()
                bernoulli_metric = environment.bernoulli_metric(recommended_arms)
                jaccard_metric = environment.jaccard_metric(recommended_arms)
                mis_id_metric = environment.mis_id_metric(recommended_arms)
                if results_file is not None and write:
                    with open(results_file, "a") as file:
                        file.write(
                            f"{algorithm_name},{experiment},{t},{bernoulli_metric},{jaccard_metric},{mis_id_metric}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithms', required=True, help='Comma-separated list: EGE_SR,EGE_SH,Uniform,PUCB1,TTPFTS')
    parser.add_argument('--environments', required=True,
                        help='Comma-separated environments: EgeExp2,EgeExp3,CovBoost...')
    parser.add_argument('--budgets', required=False, help='Comma-separated budgets, overrides default for environment.',
                        default=None)
    parser.add_argument('--num_runs', type=int, default=100)
    parser.add_argument('--results_dir', default='results')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--no_write', action='store_true')
    args = parser.parse_args()

    algorithms = [a.strip() for a in args.algorithms.split(',')]
    algorithm_names = algorithms.__str__().replace("'", "").replace(", ", "_")
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
        environment = env_cls()
        budget_list = budgets if budgets else [default_budget] * len(environments)
        max_budget = budget_list[i]
        results_file = os.path.join(args.results_dir, f"{env_name}_{algorithm_names}_{max_budget}_{args.num_runs}.csv")
        if "EGE_SR" in algorithms:
            run_EGE_experiment(args.num_runs, max_budget, environment, EGE_SR, results_file=results_file,
                               write=not args.no_write, step=args.step)
        if "EGE_SH" in algorithms:
            run_EGE_experiment(args.num_runs, max_budget, environment, EGE_SH, results_file=results_file,
                               write=not args.no_write, step=args.step)
        anytime_algs = [alg for alg in algorithms if alg in ["Uniform", "PUCB1", "TTPFTS"]]
        if anytime_algs:
            run_anytime_experiment(args.num_runs, max_budget, environment, anytime_algs, results_file=results_file,
                                   write=not args.no_write, step=args.step)
