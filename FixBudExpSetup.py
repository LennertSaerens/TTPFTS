from bandits.EGE import EGE
from bandits.TTPFTS import NormalTTPFTSBandit
from bandits.UCB import PUCB1Bandit
from bandits.Uniform import UniformBandit
from environments import Molecules


def run_EGE_experiment(num_runs, max_budget, environment, results_file=None, write=True):
    for experiment in range(num_runs):
        print(f"Running EGE experiment {experiment + 1}/{num_runs}...")

        for budget in range(201, max_budget + 1):
            # Initialize the bandit with the current budget
            bandit = EGE(environment.num_arms, environment.num_objectives, budget)

            for t in range(budget):
                # Pull the current arm
                arm = bandit.choose_arm()
                reward = environment.pull_arm(arm)
                bandit.learn(arm, reward)

            # Get the recommended arms after the budget is exhausted
            recommended_arms = bandit.get_top_arms()

            # Calculate the metrics
            bernoulli_metric = environment.bernoulli_metric(recommended_arms)
            jaccard_metric = environment.jaccard_metric(recommended_arms)
            mis_id_metric = environment.mis_id_metric(recommended_arms)

            if results_file is not None and write:
                with open(results_file, "a") as file:
                    file.write(f"EGE,{experiment},{budget},{bernoulli_metric},{jaccard_metric},{mis_id_metric}\n")


def run_anytime_experiment(num_runs, max_budget, environment, results_file=None, write=True):
    algorithms = {
        "Uniform": UniformBandit(environment.num_arms, environment.num_objectives),
        "TTPFTS": NormalTTPFTSBandit(environment.num_arms, environment.num_objectives, p=0.5),
        "PUCB1": PUCB1Bandit(environment.num_arms, environment.num_objectives, kappa=1),
    }

    for algorithm_name, bandit in algorithms.items():
        for experiment in range(num_runs):
            print(f"Running {algorithm_name} experiment {experiment + 1}/{num_runs}...")
            bandit.reset()

            for t in range(201, max_budget + 1):
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
    num_runs = 100
    write = True
    # plt.rcParams.update({'font.family': 'serif'})
    # plt.rcParams.update({'font.size': 15})
    environments = {
        # "EgeExp1": {"environment": EgeExp1.EgeExp1(), "budget": 3000},
        # "EgeExp2": {"environment": EgeExp2.EgeExp2(), "budget": 2000},
        # "EgeExp3": {"environment": EgeExp3.EgeExp3(), "budget": 3000},
        # "EgeExp4": {"environment": EgeExp4.EgeExp4(), "budget": 3000},
        # "EgeExp5": {"environment": EgeExp5.EgeExp5(), "budget": 3000},
        # "EgeExp6": {"environment": EgeExp6.EgeExp6(), "budget": 3000},
        # "EgeExp7": {"environment": EgeExp7.EgeExp7(), "budget": 3000},
        # "EgeExp8": {"environment": EgeExp8.EgeExp8(), "budget": 3000},
        # "N50VS": {"environment": N50VS.N50VS(), "budget": 3000},
        # "CovBoost": {"environment": CovBoost.CovBoost(), "budget": 2500},
        "Molecules": {"environment": Molecules.Molecules(num_arms=100), "budget": 10_000},
    }

    for environment_name, env_dict in environments.items():
        print(f"\nRunning experiments for {environment_name}...")
        results_file = f"results/EGEvsTTPFTSvsPUCB1vsUniform_{environment_name}.csv"
        environment = env_dict["environment"]
        max_budget = env_dict["budget"]
        # run_EGE_experiment(num_runs, max_budget, environment, results_file=results_file, write=write)
        run_anytime_experiment(num_runs, max_budget, environment, results_file=results_file, write=write)
