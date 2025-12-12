from pymoo.core import algorithm
from tqdm import tqdm
from bandits.ege_kone import EGE_SH, EGE_SR
from bandits.TTPFTS import TTPFTSBandit, UncertaintyDirectedTTPFTSBandit, DoubleUncertaintyDirectedTTPFTSBandit
from bandits.UCB import PUCB1Bandit
from bandits.Uniform import UniformBandit
from bandits.Posteriors import NormalIGPosterior, TPosterior, NormalPosterior
from environments import CovBoost, EgeExp1, EgeExp2, EgeExp3, EgeExp4, EgeExp5, EgeExp6, EgeExp7, EgeExp8, N50VS


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

            # Calculate the metrics
            bernoulli_metric = environment.bernoulli_metric(recommended_arms)
            jaccard_metric = environment.jaccard_metric(recommended_arms)
            mis_id_metric = environment.mis_id_metric(recommended_arms)

            if results_file is not None and write:
                with open(results_file, "a") as file:
                    file.write(f"{ege_func.__name__},{experiment},{budget},{bernoulli_metric},{jaccard_metric},{mis_id_metric}\n")


def run_anytime_experiment(num_runs, max_budget, environment, results_file=None, write=True, step=1):
    algorithms = {
        # "Uniform": UniformBandit(environment.num_arms, environment.num_objectives),
        # "PUCB1": PUCB1Bandit(environment.num_arms, environment.num_objectives, kappa=1),
        # "TTPFTS_NIG": TTPFTSBandit(NormalIGPosterior(environment.num_arms, environment.num_objectives)),
        # "TTPFTS_T_Uni": TTPFTSBandit(TPosterior(environment.num_arms, environment.num_objectives, alpha=-1/2), num_warmup_pulls=4),
        # "TTPFTS_T_Ref": TTPFTSBandit(TPosterior(environment.num_arms, environment.num_objectives, alpha=0)),
        # "TTPFTS_T_Jef": TTPFTSBandit(TPosterior(environment.num_arms, environment.num_objectives, alpha=1/2)),
        "TTPFTS_Rand": TTPFTSBandit(NormalPosterior(environment.num_arms, environment.num_objectives, environment.stds)),
        # "TTPFTS_UQ_argmax": UncertaintyDirectedTTPFTSBandit(NormalPosterior(environment.num_arms, environment.num_objectives, environment.stds), UQ_mode="argmax"),
        "TTPFTS_UQ_linear": UncertaintyDirectedTTPFTSBandit(NormalPosterior(environment.num_arms, environment.num_objectives, environment.stds), UQ_mode="linear"),
        "TTPFTS_UQ_double_linear": DoubleUncertaintyDirectedTTPFTSBandit(NormalPosterior(environment.num_arms, environment.num_objectives, environment.stds), ff_UQ_mode="linear", sf_UQ_mode="linear"),
        # "TTPFTS_UQ_softmax": UncertaintyDirectedTTPFTSBandit(NormalPosterior(environment.num_arms, environment.num_objectives, environment.stds),UQ_mode="softmax"),
    }

    for algorithm_name, bandit in algorithms.items():
        for experiment in tqdm(range(num_runs), desc=f"Running {algorithm_name} experiments", unit="experiment"):
            environment.reset()
            bandit.reset(environment.stds)

            for t in range(0, max_budget + 1, step):

                if algorithm_name == "TTPFTS_NKV" and t % 100 == 0:
                    bandit.posterior.log(f"results_corr/posteriors/TTPFTS_post_{environment_name}_e{experiment}_t{t}.parquet")

                arm = bandit.choose_arm()
                reward = environment.pull_arm(arm)
                bandit.learn(arm, reward)

                recommended_arms = bandit.get_top_arms()

                # Calculate the metrics
                bernoulli_metric = environment.bernoulli_metric(recommended_arms)
                jaccard_metric = environment.jaccard_metric(recommended_arms)
                mis_id_metric = environment.mis_id_metric(recommended_arms)

                if results_file is not None and write:
                    with open(results_file, "a") as file:
                        file.write(f"{algorithm_name},{experiment},{t},{bernoulli_metric},{jaccard_metric},{mis_id_metric}\n")


if __name__ == "__main__":
    # Set the parameters for the experiments
    num_runs = 190
    environments = {
        "EgeExp1": {"environment": EgeExp1.EgeExp1(), "budget": 5000},
        "EgeExp2": {"environment": EgeExp2.EgeExp2(), "budget": 5000},
        "EgeExp3": {"environment": EgeExp3.EgeExp3(), "budget": 5000},
        "EgeExp4": {"environment": EgeExp4.EgeExp4(), "budget": 5000},
        "EgeExp5": {"environment": EgeExp5.EgeExp5(), "budget": 5000},
        "EgeExp6": {"environment": EgeExp6.EgeExp6(), "budget": 5000},
        "EgeExp7": {"environment": EgeExp7.EgeExp7(), "budget": 5000},
        "EgeExp8": {"environment": EgeExp8.EgeExp8(), "budget": 5000},
        "N50VS": {"environment": N50VS.N50VS(), "budget": 5000},
        "CovBoost": {"environment": CovBoost.CovBoost(), "budget": 5000},
    }

    for environment_name, env_dict in environments.items():
        print(f"\nRunning experiments for {environment_name}...")
        environment = env_dict["environment"]
        max_budget = env_dict["budget"]
        results_file = f"results5000/TTPFTS_Rand_UnDi_DoubleUnDi{environment_name}_{100}_{max_budget}.csv"
        # run_EGE_experiment(num_runs, max_budget, environment, EGE_SR, results_file=results_file, write=False, step=1)
        run_anytime_experiment(num_runs, max_budget, environment, results_file=results_file, write=True, step=1)
