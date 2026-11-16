import random
import numpy as np
import scipy.stats as stats


class PTSBandit:
    """
    Pareto Thompson Sampling Bandit (PTS)
    From "Thompson Sampling for Multi-Objective Multi-Armed Bandits Problem" by Saba Yahyaa and Bernard Manderick
    """

    def __init__(self, num_arms, num_objectives):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.alphas = np.ones((num_arms, num_objectives))
        self.betas = np.ones((num_arms, num_objectives))

    def choose_arm(self):
        """
        Find all Pareto optimal arms and choose one uniformly at random.
        :return: The arm to pull.
        """
        # pareto_arms = []
        samples = np.random.beta(self.alphas, self.betas)
        is_strictly_worse = np.all(samples[:, None, :] < samples[None, :, :], axis=2)
        pareto_indices = np.where(~np.any(is_strictly_worse, axis=1))[0]
        # Compare each of the arms with all the other arms, except itself. An arm is appended to the pareto_arms list if
        # its samples for each of the objectives are greater than or equal to the samples of all the other arms.
        # for arm in range(self.num_arms):
        #     is_pareto = True
        #     for other_arm in range(self.num_arms):
        #         if arm == other_arm:
        #             continue
        #         if np.all(samples[arm] < samples[other_arm]):
        #             is_pareto = False
        #             break
        #     if is_pareto:
        #         pareto_arms.append(arm)
        # Choose an arm uniformly at random from the list of pareto optimal arms
        return random.choice(pareto_indices)

    def get_top_arms(self):
        """
        Get the arms that are considered to be Pareto optimal by the bandit.
        :return: The top arms.
        """
        samples = np.random.beta(self.alphas, self.betas)
        is_strictly_worse = np.all(samples[:, None, :] < samples[None, :, :], axis=2)
        pareto_indices = np.where(~np.any(is_strictly_worse, axis=1))[0]
        return pareto_indices

    def learn(self, arm, reward):
        """
        Learn from the reward that was received for pulling the arm.
        :param arm: The arm that was pulled.
        :param reward: The reward for each objective.
        :return: None
        """
        for o in range(self.num_objectives):
            self.alphas[arm][o] += reward[o]
            self.betas[arm][o] += 1 - reward[o]

    def reset(self):
        """
        Reset the agent.
        :return: None
        """
        self.alphas = np.ones((self.num_arms, self.num_objectives))
        self.betas = np.ones((self.num_arms, self.num_objectives))


class LSTSBandit:
    """
    Linear Scalarized Thompson Sampling Bandit (LSF-TS)
    From "Thompson Sampling for Multi-Objective Multi-Armed Bandits Problem" by Saba Yahyaa and Bernard Manderick
    """

    def __init__(self, num_arms, num_objectives, scalarization_functions):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.scalarization_functions = scalarization_functions
        self.num_scalarization_functions = len(scalarization_functions)
        self.alphas = np.ones((self.num_scalarization_functions, num_arms, num_objectives))
        self.betas = np.ones((self.num_scalarization_functions, num_arms, num_objectives))
        self.MRU = False  # Most Recently Used scalarization function

    def choose_arm(self):
        """
        Find the Pareto optimal arm by sampling from the beta distribution and using the scalarization functions.
        :return: The arm to pull.
        """
        samples = np.random.beta(self.alphas, self.betas)
        # pick a random scalarization function
        scalarization_function = random.randint(0, self.num_scalarization_functions - 1)
        # Calculate the scalarized values for each arm
        scalarized_values = np.dot(samples[scalarization_function],
                                   self.scalarization_functions[scalarization_function])
        # Set the most recently used scalarization function
        self.MRU = scalarization_function
        # Find the arm with the highest scalarized value
        return np.argmax(scalarized_values)

    def learn(self, arm, reward):
        """
        Learn from the reward that was received for pulling the arm.
        :param arm: The arm that was pulled.
        :param reward: The reward for each objective.
        :return: None
        """
        for o in range(self.num_objectives):
            self.alphas[self.MRU][arm][o] += reward[o]
            self.betas[self.MRU][arm][o] += 1 - reward[o]

    def reset(self):
        """
        Reset the agent.
        :return: None
        """
        self.alphas = np.ones((self.num_scalarization_functions, self.num_arms, self.num_objectives))
        self.betas = np.ones((self.num_scalarization_functions, self.num_arms, self.num_objectives))


class NormalPTSBandit:
    """
    Variant of the Pareto Thompson Sampling bandit that uses Normal-Inverse-Gamma Distribution instead of Beta.
    Change made because the rewards are not binary but sampled from a normal distribution.
    """

    def __init__(self, num_arms, num_objectives):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.mu = np.zeros((num_arms, num_objectives))  # mean
        self.precision = np.ones((num_arms, num_objectives))  # precision, also called lambda
        self.alpha = np.ones((num_arms, num_objectives))  # shape
        self.beta = np.ones((num_arms, num_objectives))  # scale

    def choose_arm(self):
        """
        Find all Pareto optimal arms and choose one uniformly at random.
        :return: The arm to pull.
        """
        variances = stats.invgamma.rvs(self.alpha, scale=self.beta)
        samples = np.random.normal(self.mu, np.sqrt(variances / self.precision))
        is_strictly_worse = np.all(samples[:, None, :] < samples[None, :, :], axis=2)
        pareto_indices = np.where(~np.any(is_strictly_worse, axis=1))[0]
        return random.choice(pareto_indices)

    # def get_top_arms(self):
    #     """
    #     Get the arms that are considered to be Pareto optimal by the bandit.
    #     :return: The top arms.
    #     """
    #     variances = stats.invgamma.rvs(self.alpha, scale=self.beta)
    #     samples = np.random.normal(self.mu, np.sqrt(variances / self.precision))
    #     is_strictly_worse = np.all(samples[:, None, :] < samples[None, :, :], axis=2)
    #     pareto_indices = np.where(~np.any(is_strictly_worse, axis=1))[0]
    #     return pareto_indices

    def get_top_arms(self):
        """
        get the Pareto optimal arms based on the estimated arm means
        :return: The top arms.
        """
        is_strictly_worse = np.all(self.mu[:, None, :] < self.mu[None, :, :], axis=2)
        pareto_indices = np.where(~np.any(is_strictly_worse, axis=1))[0]
        return pareto_indices

    def learn(self, arm, reward):
        """
        Learn from the reward that was received for pulling the arm.
        :param arm: The arm that was pulled.
        :param reward: The reward for each objective.
        :return: None
        """
        for o in range(self.num_objectives):
            self.beta[arm][o] += 0.5 * (reward[o] - self.mu[arm][o]) ** 2 * (
                        self.precision[arm][o] / (self.precision[arm][o] + 1))
            self.mu[arm][o] = (self.mu[arm][o] * self.precision[arm][o] + reward[o]) / (self.precision[arm][o] + 1)
            self.precision[arm][o] += 1
            self.alpha[arm][o] += 0.5

    def reset(self):
        """
        Reset the agent.
        :return: None
        """
        self.mu = np.zeros((self.num_arms, self.num_objectives))
        self.precision = np.ones((self.num_arms, self.num_objectives))
        self.alpha = np.ones((self.num_arms, self.num_objectives))
        self.beta = np.ones((self.num_arms, self.num_objectives))


class NormalLSTSBandit:
    """
    Variant of the Linear Scalarized Thompson Sampling bandit that uses Normal-Inverse-Gamma Distribution instead of Beta.
    Change made because the rewards are not binary but sampled from a normal distribution.
    """

    def __init__(self, num_arms, num_objectives, scalarization_functions):
        self.num_arms = num_arms
        self.num_objectives = num_objectives
        self.scalarization_functions = scalarization_functions
        self.num_scalarization_functions = len(scalarization_functions)
        self.mu = np.zeros((self.num_scalarization_functions, num_arms, num_objectives))  # mean
        self.precision = np.ones(
            (self.num_scalarization_functions, num_arms, num_objectives))  # precision, also called lambda
        self.alpha = np.ones((self.num_scalarization_functions, num_arms, num_objectives))  # shape
        self.beta = np.ones((self.num_scalarization_functions, num_arms, num_objectives))  # scale
        self.MRU = False  # Most Recently Used scalarization function

    def choose_arm(self):
        """
        Find the Pareto optimal arm by sampling from the beta distribution and using the scalarization functions.
        :return: The arm to pull.
        """
        variances = stats.invgamma.rvs(self.alpha, scale=self.beta)
        samples = np.random.normal(self.mu, np.sqrt(variances / self.precision))
        # pick a random scalarization function
        scalarization_function = random.randint(0, self.num_scalarization_functions - 1)
        # Calculate the scalarized values for each arm
        scalarized_values = np.dot(samples[scalarization_function],
                                   self.scalarization_functions[scalarization_function])
        # Set the most recently used scalarization function
        self.MRU = scalarization_function
        # Find the arm with the highest scalarized value
        return np.argmax(scalarized_values)

    def learn(self, arm, reward):
        """
        Learn from the reward that was received for pulling the arm.
        :param arm: The arm that was pulled.
        :param reward: The reward for each objective.
        :return: None
        """
        for o in range(self.num_objectives):
            self.beta[self.MRU][arm][o] += 0.5 * (reward[o] - self.mu[self.MRU][arm][o]) ** 2 * (
                        self.precision[self.MRU][arm][o] / (self.precision[self.MRU][arm][o] + 1))
            self.mu[self.MRU][arm][o] = (self.mu[self.MRU][arm][o] * self.precision[self.MRU][arm][o] + reward[o]) / (
                        self.precision[self.MRU][arm][o] + 1)
            self.precision[self.MRU][arm][o] += 1
            self.alpha[self.MRU][arm][o] += 0.5

    def reset(self):
        """
        Reset the agent.
        :return: None
        """
        self.mu = np.zeros((self.num_scalarization_functions, self.num_arms, self.num_objectives))
        self.precision = np.ones((self.num_scalarization_functions, self.num_arms, self.num_objectives))
        self.alpha = np.ones((self.num_scalarization_functions, self.num_arms, self.num_objectives))
        self.beta = np.ones((self.num_scalarization_functions, self.num_arms, self.num_objectives))
