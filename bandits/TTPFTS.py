import random

import numpy as np
from paretoset import paretoset
from bandits.InterfaceMOMABPFI import BaseMOMABAlgorithm
from uncertainty_quantification import diag_cov_from_stds, bhattacharyya_coeff_gaussians


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


class UncertaintyDirectedTTPFTSBandit(BaseMOMABAlgorithm):
    """
    TTP-FTS bandit with uncertainty-directed exploration:
    - With probability p: exploitation: sample and play a sampled-Pareto arm (uniform).
    - Else: exploration: among the (sampled) second Pareto front arms, select the one
      that maximizes the Bhattacharyya coefficient with any sampled first-front arm,
      using the current posterior means and posterior stds (diagonal covariances).
    """

    def __init__(self, posterior, p=0.5, num_warmup_pulls=2, UQ_mode="argmax"):
        """
        posterior: object following your NormalPosterior interface (has .num_arms, .num_objectives,
                   .counts, .known_stds, .means, and methods sample(), update(), get_mean(), reset()).
        p: probability of exploitation (play sampled Pareto arm).
        num_warmup_pulls: how many times each arm is pulled during warm-up.
        large_std: used for unpulled arms' posterior std (matches your original design).
        """
        super().__init__(posterior.num_arms, posterior.num_objectives)
        self.UQ_mode = UQ_mode
        self.posterior = posterior
        self.p = p
        self.num_objectives = posterior.num_objectives
        self.num_warmup_pulls = num_warmup_pulls
        self.current_warmup_arm = 0
        self.warmup_pulls = np.zeros(self.posterior.num_arms, dtype=int)

    def choose_arm(self):
        # Warm-up phase
        if np.any(self.warmup_pulls < self.num_warmup_pulls):
            arm = self.current_warmup_arm
            self.warmup_pulls[arm] += 1
            self.current_warmup_arm = (self.current_warmup_arm + 1) % self.posterior.num_arms
            return arm

        # Main sampling
        samples = self.posterior.sample()  # shape (num_arms, num_objectives)
        pareto_mask = paretoset(samples, sense=["max"] * self.num_objectives)
        pareto_indices = np.where(pareto_mask)[0]

        if np.random.random() < self.p:
            # exploitation: sample-based Pareto front (uniform)
            if len(pareto_indices) == 0:
                # should not happen, but fallback gracefully
                return random.randrange(self.posterior.num_arms)
            return random.choice(pareto_indices)

        # exploration: pick second front arm with largest BC overlap to any first-front arm
        non_pareto_indices = np.where(~pareto_mask)[0]
        if len(non_pareto_indices) == 0:
            # nothing in second front, fallback to pareto
            return random.choice(pareto_indices)

        # From the non-pareto arms, compute their own Pareto (i.e., second front relative to samples)
        non_pareto_samples = samples[non_pareto_indices]
        non_pareto_pareto_mask = paretoset(non_pareto_samples, ["max"] * self.num_objectives)
        non_dominated_indices = np.where(non_pareto_pareto_mask)[0]
        # map back to global indices
        second_front_indices = non_pareto_indices[non_dominated_indices]

        if len(second_front_indices) == 0:
            # no second front - fallback
            return random.choice(non_pareto_indices)

        # Compute current posterior means & stds (not the random samples)
        means = self.posterior.get_mean()  # (num_arms, num_objectives)
        stds = self.posterior.get_stds()  # (num_arms, num_objectives)

        # For each second-front arm, compute max BC versus any sampled first-front arm
        max_bcs = []
        for sec_arm in second_front_indices:
            mu_sec = means[sec_arm]
            std_sec = stds[sec_arm]
            Sigma_sec = diag_cov_from_stds(std_sec)

            best_bc = 0.0
            for opt_arm in pareto_indices:
                mu_opt = means[opt_arm]
                std_opt = stds[opt_arm]
                Sigma_opt = diag_cov_from_stds(std_opt)
                bc = bhattacharyya_coeff_gaussians(mu_opt, Sigma_opt, mu_sec, Sigma_sec)
                if bc > best_bc:
                    best_bc = bc
            max_bcs.append(best_bc)

        max_bcs = np.array(max_bcs, dtype=float)

        # Select according to UQ_mode
        if self.UQ_mode == "argmax":
            chosen_arm = second_front_indices[np.argmax(max_bcs)]
        elif self.UQ_mode == "linear":
            probs = max_bcs / max_bcs.sum()
            chosen_arm = np.random.choice(second_front_indices, p=probs)
        elif self.UQ_mode == "softmax":
            temp = 0.1  # temperature parameter; could be made adjustable
            exp_bcs = np.exp(max_bcs / temp)
            probs = exp_bcs / exp_bcs.sum()
            chosen_arm = np.random.choice(second_front_indices, p=probs)
        else:
            raise ValueError(f"Unknown UQ_mode: {self.UQ_mode}")
        return chosen_arm

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


class DoubleUncertaintyDirectedTTPFTSBandit(BaseMOMABAlgorithm):
    """
    TTP-FTS bandit with uncertainty-directed exploration:
    - With probability p: exploitation: sample and play a sampled-Pareto arm (uniform).
    - Else: exploration: among the (sampled) second Pareto front arms, select the one
      that maximizes the Bhattacharyya coefficient with any sampled first-front arm,
      using the current posterior means and posterior stds (diagonal covariances).
    """

    def __init__(self, posterior, p=0.5, num_warmup_pulls=2, ff_UQ_mode="linear", sf_UQ_mode="linear"):
        """
        posterior: object following your NormalPosterior interface (has .num_arms, .num_objectives,
                   .counts, .known_stds, .means, and methods sample(), update(), get_mean(), reset()).
        p: probability of exploitation (play sampled Pareto arm).
        num_warmup_pulls: how many times each arm is pulled during warm-up.
        large_std: used for unpulled arms' posterior std (matches your original design).
        """
        super().__init__(posterior.num_arms, posterior.num_objectives)
        self.ff_UQ_mode = ff_UQ_mode
        self.sf_UQ_mode = sf_UQ_mode
        self.posterior = posterior
        self.p = p
        self.num_objectives = posterior.num_objectives
        self.num_warmup_pulls = num_warmup_pulls
        self.current_warmup_arm = 0
        self.warmup_pulls = np.zeros(self.posterior.num_arms, dtype=int)

    def choose_arm(self):
        # Warm-up phase
        if np.any(self.warmup_pulls < self.num_warmup_pulls):
            arm = self.current_warmup_arm
            self.warmup_pulls[arm] += 1
            self.current_warmup_arm = (self.current_warmup_arm + 1) % self.posterior.num_arms
            return arm

        # Main sampling
        samples = self.posterior.sample()  # shape (num_arms, num_objectives)
        pareto_mask = paretoset(samples, sense=["max"] * self.num_objectives)
        first_front_indices = np.where(pareto_mask)[0]

        non_pareto_indices = np.where(~pareto_mask)[0]
        non_pareto_samples = samples[non_pareto_indices]
        non_pareto_pareto_mask = paretoset(non_pareto_samples, ["max"] * self.num_objectives)
        non_dominated_indices = np.where(non_pareto_pareto_mask)[0]
        # map back to global indices
        second_front_indices = non_pareto_indices[non_dominated_indices]

        means = self.posterior.get_mean()  # (num_arms, num_objectives)
        stds = self.posterior.get_stds()  # (num_arms, num_objectives)
        bcs_grid = np.zeros((len(first_front_indices), len(second_front_indices)), dtype=float)

        for i, ff_arm in enumerate(first_front_indices):
            mu_ff = means[ff_arm]
            std_ff = stds[ff_arm]
            Sigma_ff = diag_cov_from_stds(std_ff)
            for j, sf_arm in enumerate(second_front_indices):
                mu_sf = means[sf_arm]
                std_sf = stds[sf_arm]
                Sigma_sf = diag_cov_from_stds(std_sf)
                bc = bhattacharyya_coeff_gaussians(mu_ff, Sigma_ff, mu_sf, Sigma_sf)
                bcs_grid[i, j] = bc

        if np.random.random() < self.p:
            if len(first_front_indices) == 0 or bcs_grid.size == 0:
                # should not happen, but fallback gracefully
                return random.randrange(self.posterior.num_arms)
            # exploitation: sample-based Pareto front (uncertainty-directed)
            max_bcs_ff = bcs_grid.max(axis=1)
            if self.ff_UQ_mode == "argmax":
                probs = np.zeros_like(max_bcs_ff)
                probs[np.argmax(max_bcs_ff)] = 1.0
            elif self.ff_UQ_mode == "linear":
                probs = max_bcs_ff / max_bcs_ff.sum()
            elif self.ff_UQ_mode == "softmax":
                temp = 0.1
                exp_bcs = np.exp(max_bcs_ff / temp)
                probs = exp_bcs / exp_bcs.sum()
            elif self.ff_UQ_mode == "random":
                probs = np.ones_like(max_bcs_ff) / len(max_bcs_ff)
            else:
                raise ValueError(f"Unknown ff_UQ_mode: {self.ff_UQ_mode}")
            chosen_ff_arm = np.random.choice(first_front_indices, p=probs)
            return chosen_ff_arm
        else:
            if len(second_front_indices) == 0:
                # nothing in second front, fallback to pareto
                return random.choice(first_front_indices)
            # exploration: pick second front arm with largest BC versus any first-front arm
            max_bcs_sf = bcs_grid.max(axis=0)
            if self.sf_UQ_mode == "argmax":
                probs = np.zeros_like(max_bcs_sf)
                probs[np.argmax(max_bcs_sf)] = 1.0
            elif self.sf_UQ_mode == "linear":
                probs = max_bcs_sf / max_bcs_sf.sum()
            elif self.sf_UQ_mode == "softmax":
                temp = 0.1  # temperature parameter; could be made adjustable
                exp_bcs = np.exp(max_bcs_sf / temp)
                probs = exp_bcs / exp_bcs.sum()
            elif self.sf_UQ_mode == "random":
                probs = np.ones_like(max_bcs_sf) / len(max_bcs_sf)
            else:
                raise ValueError(f"Unknown sf_UQ_mode: {self.sf_UQ_mode}")
            chosen_sf_arm = np.random.choice(second_front_indices, p=probs)
            return chosen_sf_arm

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
