import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV
from sklearn.preprocessing import MinMaxScaler

from bandits.Annealing import APBandit
from bandits.KnowledgeGradient import PKGBandit
from bandits.TTPFTS import NormalTTPFTSBandit
from bandits.ThompsonSampling import NormalPTSBandit
from bandits.UCB import PUCB1Bandit
from bandits.Uniform import UniformBandit

from plotting import plot_vaccination_data

data = pd.read_csv('results/Experiment1Extended.csv')

medical_burden = data['Medical Burden'].values.reshape(-1, 1)
monetary_cost = data['Monetary Cost'].values.reshape(-1, 1)

max_range = 100

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, max_range))
normalized_medical_burden = scaler.fit_transform(medical_burden)
normalized_monetary_cost = scaler.fit_transform(monetary_cost)

inverted_arms = [(normalized_medical_burden[i][0], normalized_monetary_cost[i][0]) for i in
                 range(len(normalized_medical_burden))]

# transform back into a dataframe
data['Medical Burden'] = normalized_medical_burden
data['Monetary Cost'] = normalized_monetary_cost

# Invert the data for maximization
maximized_medical_burden = max_range - normalized_medical_burden
maximized_monetary_cost = max_range - normalized_monetary_cost

arms = [(maximized_medical_burden[i][0], maximized_monetary_cost[i][0]) for i in range(len(maximized_medical_burden))]

std = 10

reference_point = np.array([max_range, max_range])

optimal_arms = [0, 5, 6, 8, 14, 30, 31, 32]

num_runs = 100
horizon = 30_000


def is_completely_cor_rec(recommended, pareto_arms):
    """
    Check if the recommended arms are completely correct and recommended.
    :param recommended: The recommended arms.
    :param pareto_arms: The pareto optimal arms.
    :return: True if the recommended arms are completely correct and recommended, False otherwise.
    """
    return int(set(recommended) == set(pareto_arms))


def calculate_jaccard_similarity(recommended, pareto_arms):
    """
    Calculate the Jaccard similarity between the recommended arms and the pareto optimal arms.
    :param recommended: The recommended arms.
    :param pareto_arms: The pareto optimal arms.
    :return: The Jaccard similarity.
    """
    return len(set(recommended).intersection(set(pareto_arms))) / len(set(recommended).union(set(pareto_arms)))


def calc_hypervolume(recommended):
    """
    Calculate the hypervolume of the recommended arms using pymoo.
    :param recommended: The recommended arms.
    :return: The hypervolume.
    """
    F = np.array([inverted_arms[arm] for arm in recommended])
    ind = HV(ref_point=reference_point)
    hv = ind.do(F)
    return hv


def pull_arm(arm):
    return [np.random.normal(arm[0], std), np.random.normal(arm[1], std)]


def run_experiment(num_arms, num_objectives, arms, pareto_indices, results_file=None, rec_file=None):
    setup = {
        # "Pareto UCB1": {"agent": PUCB1Bandit(num_arms, num_objectives, 10)},
        # "Pareto Thompson Sampling": {"agent": NormalPTSBandit(num_arms, num_objectives)},
        # "Pareto Knowledge Gradient": {"agent": PKGBandit(num_arms, num_objectives, horizon, 6)},
        # "Annealing Pareto": {"agent": APBandit(num_arms, num_objectives, 40, 1, 0.9999)},
        # "TTPFTS": {"agent": NormalTTPFTSBandit(num_arms, num_objectives, 0.9)},
        "Uniform Sampling": {"agent": UniformBandit(num_arms, num_objectives)}
    }

    for algorithm in setup:
        agent = setup[algorithm]["agent"]

        for experiment in range(num_runs):
            print(f"Experiment {experiment} for algorithm {algorithm}")
            agent.reset()

            for t in range(horizon):
                arm = agent.choose_arm()
                reward = pull_arm(arms[arm])
                agent.learn(arm, reward)
                recommended = agent.get_top_arms()

                bernoulli_metric = is_completely_cor_rec(recommended, pareto_indices)
                jaccard_metric = calculate_jaccard_similarity(recommended, pareto_indices)
                hypervolume = calc_hypervolume(recommended)

                if results_file is not None:
                    with open(results_file, "a") as file:
                        file.write(
                            f"{algorithm},{experiment},{t},{bernoulli_metric},{jaccard_metric},{hypervolume},{arm}\n")

                if rec_file is not None:
                    with open(rec_file, "a") as file:
                        file.write(f"{algorithm},{recommended}\n")


if __name__ == '__main__':
    run_experiment(len(arms), 2, arms, optimal_arms, results_file="results/baseline.csv", rec_file="results/baseline_recs.csv")
