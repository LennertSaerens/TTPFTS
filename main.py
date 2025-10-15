import numpy as np
from pymoo.indicators.hv import HV
import plotting
from bandits.ThompsonSampling import PTSBandit, LSTSBandit
from bandits.TTPFTS import TTPFTSBandit
from bandits.UCB import PUCB1Bandit, LSUCB1Bandit
from bandits.KnowledgeGradient import PKGBandit, LSKGArmsBandit, LSKGObjectivesBandit
from bandits.Annealing import APBandit

num_runs = 100  # Number of experiments M
horizon = 250_000  # Number of time steps T

# Configuration for the first experiment
e1_arms = [(0.55, 0.5), (0.53, 0.51), (0.52, 0.54), (0.5, 0.57), (0.51, 0.51), (0.5, 0.5)] + 14 * [(0.48, 0.48)]
e1_num_arms = len(e1_arms)
e1_num_objectives = len(e1_arms[0])
e1_pareto_arms = [0, 1, 2, 3]
e1_weights = [(x, 1 - x) for x in np.linspace(0, 1, 11)]
e1_cheby_ref = (0.47, 0.47)


def calculate_pareto_regret(arm, arms, pareto_arms):
    """
    Calculate the pareto regret for reward that was received at time step t for pulling arm.
    The regret is zero if the arm is pareto optimal. Otherwise, the regret is the distance between the success
    rates of the arm and the success rates of the closest pareto optimal arm.
    :param pareto_arms: The pareto optimal arms.
    :param arms: The arms of the bandit.
    :param arm: The arm that was pulled.
    :return: The pareto regret.
    """
    if arm in pareto_arms:
        return 0
    else:
        pareto_distances = [np.sqrt(sum((np.array(arms[arm]) - np.array(arms[pareto_arm])) ** 2)) for pareto_arm in
                            pareto_arms]
        return min(pareto_distances)


def calculate_unfairness_regret(arm_pulls, pareto_arms):
    """
    Calculate the unfairness regret that was received at time step t for pulling arm.
    The unfairness of a MOMAB algorithm is defined as the variance of the arms in the Pareto front.
    :param pareto_arms: The pareto optimal arms.
    :param arm_pulls: The number of times each arm has been pulled.
    :return: The unfairness regret.
    """
    pareto_arm_pulls = [arm_pulls[pareto_arm] for pareto_arm in pareto_arms]
    return np.var(pareto_arm_pulls)


def pull(arm, arms, num_objectives):
    """
    Pull an arm of the bandit and return the reward for each objective. The rewards ri of arm i are drawn from
    a Bernoulli distribution. The success rate for each objective is given by the arms list.
    :param num_objectives: The number of objectives.
    :param arms: The arms of the bandit.
    :param arm: The arm to pull.
    :return: The reward for each objective.
    """
    return [int(np.random.random() < arms[arm][o]) for o in range(num_objectives)]


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
    ref_point = np.array([1, 1])
    F = np.array([e1_arms[arm] for arm in recommended])
    ind = HV(ref_point=ref_point)
    hv = ind.do(F)
    return hv


def run_experiment(num_arms, num_objectives, arms, pareto_arms, weights, result_file, write=False, log=False):
    setup = {
        # "Pareto UCB1": {
        #     "agent": PUCB1Bandit(num_arms, num_objectives, 0.5),
        #     "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
        #     "cumulative_unfairness_regrets": [[] for _ in range(num_runs)],
        #     "arm_pulls": [np.zeros(num_arms) for _ in range(num_runs)]
        # },
        "Linear Scalarized UCB1": {
            "agent": LSUCB1Bandit(num_arms, num_objectives, weights, 0.5),
            "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
            "cumulative_unfairness_regrets": [[] for _ in range(num_runs)],
            "arm_pulls": [np.zeros(num_arms) for _ in range(num_runs)],
            "cor_rec_per": [[] for _ in range(num_runs)]
        },
        # "Pareto Thompson Sampling": {
        #     "agent": PTSBandit(num_arms, num_objectives),
        #     "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
        #     "cumulative_unfairness_regrets": [[] for _ in range(num_runs)],
        #     "arm_pulls": [np.zeros(num_arms) for _ in range(num_runs)]
        # },
        #     "Linear Scalarized Thompson Sampling": {
        #         "agent": LSTSBandit(num_arms, num_objectives, weights),
        #         "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
        #         "cumulative_unfairness_regrets": [[] for _ in range(num_runs)],
        #         "arm_pulls": [np.zeros(num_arms) for _ in range(num_runs)],
        #         "cor_rec_per": [[] for _ in range(num_runs)]
        #     },
        # "Pareto Knowledge Gradient": {
        #     "agent": PKGBandit(num_arms, num_objectives, horizon, 5),
        #     "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
        #     "cumulative_unfairness_regrets": [[] for _ in range(num_runs)],
        #     "arm_pulls": [np.zeros(num_arms) for _ in range(num_runs)]
        # },
        "Linear Scalarized Knowledge Gradient (arms)": {
            "agent": LSKGArmsBandit(num_arms, num_objectives, horizon, 5, weights),
            "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
            "cumulative_unfairness_regrets": [[] for _ in range(num_runs)],
            "arm_pulls": [np.zeros(num_arms) for _ in range(num_runs)],
            "cor_rec_per": [[] for _ in range(num_runs)]
        },
        "Linear Scalarized Knowledge Gradient (objectives)": {
            "agent": LSKGObjectivesBandit(num_arms, num_objectives, horizon, 5, weights),
            "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
            "cumulative_unfairness_regrets": [[] for _ in range(num_runs)],
            "arm_pulls": [np.zeros(num_arms) for _ in range(num_runs)],
            "cor_rec_per": [[] for _ in range(num_runs)]
        },
        # "Annealing Pareto": {
        #     "agent": APBandit(num_arms, num_objectives, horizon, 5, 1, 0.9995),
        #     "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
        #     "cumulative_unfairness_regrets": [[] for _ in range(num_runs)],
        #     "arm_pulls": [np.zeros(num_arms) for _ in range(num_runs)]
        # },
        # "Top-Two PF TS": {
        #     "agent": TTPFTSBandit(num_arms, num_objectives),
        #     "cumulative_pareto_regrets": [[] for _ in range(num_runs)],
        #     "cumulative_unfairness_regrets": [[] for _ in range(num_runs)],
        #     "arm_pulls": [np.zeros(num_arms) for _ in range(num_runs)]
        # }
    }

    for algorithm in setup:
        agent = setup[algorithm]["agent"]
        cumulative_pareto_regrets = setup[algorithm]["cumulative_pareto_regrets"]
        cumulative_unfairness_regrets = setup[algorithm]["cumulative_unfairness_regrets"]

        for experiment in range(num_runs):
            print(f"Experiment {experiment} for algorithm {algorithm}")
            arm_pulls = np.zeros(num_arms)  # Number of times each arm has been pulled
            agent.reset()

            for t in range(horizon):
                # Choose an arm
                arm = agent.choose_arm()
                # Pull the arm and receive the reward
                reward = pull(arm, arms, num_objectives)
                arm_pulls[arm] += 1
                # Learn from the reward
                agent.learn(arm, reward)

                # Calculate the pareto regret and the unfairness regret
                # pareto_regret = calculate_pareto_regret(arm, arms, pareto_arms)
                # unfairness_regret = calculate_unfairness_regret(arm_pulls, pareto_arms)

                # recommended = agent.get_top_arms()
                # cor_rec = is_completely_cor_rec(recommended, pareto_arms)
                # jaccard_sim = calculate_jaccard_similarity(recommended, pareto_arms)
                # hypervolume = calc_hypervolume(recommended)

                # if log:
                #     print(
                #         f"t: {t}, arm: {arm}, reward: {reward}, pareto regret: {pareto_regret}, unfairness regret: {unfairness_regret}")

                # Update the cumulative pareto regret and the cumulative unfairness regret
                # cumulative_pareto_regret = cumulative_pareto_regrets[experiment][
                #                                -1] + pareto_regret if t > 0 else pareto_regret
                # cumulative_unfairness_regret = cumulative_unfairness_regrets[experiment][
                #                                    -1] + unfairness_regret if t > 0 else unfairness_regret
                #
                # cumulative_pareto_regrets[experiment].append(cumulative_pareto_regret)
                # cumulative_unfairness_regrets[experiment].append(cumulative_unfairness_regret)

                if result_file is not None and write:
                    with open(result_file, 'a') as file:
                        # file.write(
                        #     f"{algorithm},{experiment},{t},{arm},{cumulative_pareto_regret},{cumulative_unfairness_regret},{cor_rec},{jaccard_sim},{hypervolume}\n")
                        file.write(
                            f"{algorithm},{experiment},{t},{arm}\n")

            # setup[algorithm]["arm_pulls"][experiment] = arm_pulls

    # Plot the cumulative pareto regrets and the cumulative unfairness regrets
    # plotting.plot_regrets(setup)
    # plotting.plot_bernoulli_metric(setup)
    # plotting.plot_jaccard_metric(setup)
    # plotting.plot_hypervolume(setup)


if __name__ == "__main__":
    run_experiment(e1_num_arms, e1_num_objectives, e1_arms, e1_pareto_arms, e1_weights, "results/bandits/test2.csv",
                   log=False, write=True, )
