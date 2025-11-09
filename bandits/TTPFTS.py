import random

import numpy as np
from paretoset import paretoset


class TTPFTSBandit:
    def __init__(self, posterior, p):
        self.posterior = posterior
        self.p = p
        self.num_objectives = posterior.num_objectives

    def choose_arm(self):
        samples = self.posterior.sample()
        pareto_mask = paretoset(samples, sense=["max"] * self.num_objectives)
        pareto_indices = np.where(pareto_mask)[0]
        if np.random.random() < self.p:
            return random.choice(pareto_indices)
        else:
            non_pareto_indices = np.where(~pareto_mask)[0]
            if len(non_pareto_indices) == 0:
                return random.choice(pareto_indices)
            non_pareto_samples = samples[non_pareto_indices]
            non_pareto_pareto_mask = paretoset(non_pareto_samples, ["max"] * self.num_objectives)
            non_dominated_indices = np.where(non_pareto_pareto_mask)[0]
            non_dominated_indices = non_pareto_indices[non_dominated_indices]
            return random.choice(non_dominated_indices)

    def get_top_arms(self):
        means = self.posterior.get_mean()
        pareto_mask = paretoset(means, sense=["max"] * self.num_objectives)
        pareto_indices = np.where(pareto_mask)[0]
        return pareto_indices

    def learn(self, arm, reward):
        self.posterior.update(arm, reward)

    def reset(self):
        self.posterior.reset()
