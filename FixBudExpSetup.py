from bandits.EGE import EGE
from bandits.TTPFTS import NormalTTPFTSBandit
from bandits.Uniform import UniformBandit

from environments import N50VS, EgeExp1, EgeExp2, EgeExp3, EgeExp4, CovBoost


def run_EGE_experiment(num_runs, max_budget, environment, results_file=None, write=True):
    for experiment in range(num_runs):
        print(f"Running EGE experiment {experiment + 1}/{num_runs}...")

        for budget in range(91, max_budget + 1):
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


def run_TTPFTS_experiment(num_runs, max_budget, environment, results_file=None, write=True):
    for experiment in range(num_runs):
        print(f"Running TTPFTS experiment {experiment + 1}/{num_runs}...")
        bandit = NormalTTPFTSBandit(environment.num_arms, environment.num_objectives, p=0.5)

        for t in range(91, max_budget + 1):
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
                    file.write(f"TTPFTS,{experiment},{t},{bernoulli_metric},{jaccard_metric},{mis_id_metric}\n")


def run_baseline_experiment(num_runs, max_budget, environment, results_file=None, write=True):
    for experiment in range(num_runs):
        print(f"Running Uniform Sampling experiment {experiment + 1}/{num_runs}...")
        bandit = UniformBandit(environment.num_arms, environment.num_objectives)

        for t in range(91, max_budget + 1):
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
                    file.write(f"Uniform,{experiment},{t},{bernoulli_metric},{jaccard_metric},{mis_id_metric}\n")


if __name__ == "__main__":
    # Set the parameters for the experiments
    num_runs = 100
    max_budget = 2500

    environments = {
        # "EgeExp1": EgeExp1.EgeExp1(),
        # "EgeExp2": EgeExp2.EgeExp2(),
        # "EgeExp3": EgeExp3.EgeExp3(),
        # "EgeExp4": EgeExp4.EgeExp4(),
        # "N50VS": N50VS.N50VS(),
        "CovBoost": CovBoost.CovBoost()
    }
    write = True

    for environment_name, environment in environments.items():
        print(f"\nRunning experiments for {environment_name}...")
        results_file = f"results/TTPFTSvsUniform_{environment_name}.csv"
        run_EGE_experiment(num_runs, max_budget, environment, results_file, write)
        run_TTPFTS_experiment(num_runs, max_budget, environment, results_file, write)
        run_baseline_experiment(num_runs, max_budget, environment, results_file, write)

    # Run the experiments
    # run_EGE_experiment(num_runs, max_budget, environment, results_file, write)
    # run_TTPFTS_experiment(num_runs, max_budget, environment, results_file, write)
    # run_baseline_experiment(num_runs, max_budget, environment, results_file, write)
