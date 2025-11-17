import random

import numpy as np
from paretoset import paretoset
from bandits.InterfaceMOMABPFI import BaseMOMABAlgorithm


class TTPFTSBandit(BaseMOMABAlgorithm):
    def __init__(self, posterior, p=0.5, num_warmup_pulls=2):
        super().__init__(posterior.num_arms, posterior.num_objectives)
        self.posterior = posterior
        self.p = p
        self.num_objectives = posterior.num_objectives
        self.num_warmup_pulls = num_warmup_pulls
        self.current_warmup_arm = 0
        self.warmup_pulls = np.zeros(posterior.num_arms, dtype=int)

    def choose_arm(self):
        # Warm-up phase
        if np.any(self.warmup_pulls < self.num_warmup_pulls):
            arm = self.current_warmup_arm
            self.warmup_pulls[arm] += 1
            self.current_warmup_arm = (self.current_warmup_arm + 1) % self.posterior.num_arms
            return arm
        # Main TTPFTS sampling strategy
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

    def reset(self, env_stds):
        self.posterior.reset(env_stds)
        self.warmup_pulls = np.zeros(self.posterior.num_arms, dtype=int)
        self.current_warmup_arm = 0
