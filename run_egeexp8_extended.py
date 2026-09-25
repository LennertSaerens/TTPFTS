import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

from environments.EgeExp8 import EgeExp8
from bandits.TTPFTS import TTPFTSBandit
from uncertainty_quantification import uncertainty_quantification

NUM_RUNS = 100
MAX_BUDGET = 500000
LOG_INTERVAL = 1000
RESULTS_FILE = "results/egeexp8_extended_uq.csv"


def run_single_experiment(seed):
    np.random.seed(seed)
    env = EgeExp8()
    
    # Initialize the posterior using NormalInverseGamma (as in momab_main.py)
    # The default for gaussian envs is usually NormalInverseGamma.
    from bandits.Posteriors import TPosterior
    
    posterior = TPosterior(
        num_arms=env.num_arms,
        num_objectives=env.num_objectives,
        alpha=-0.5
    )
    
    bandit = TTPFTSBandit(posterior, p=0.5, num_warmup_pulls=4)
    bandit.reset(env.stds)
    env.reset()
    
    results = []
    
    for t in range(1, MAX_BUDGET + 1):
        arm = bandit.choose_arm()
        reward = env.pull_arm(arm)
        bandit.learn(arm, reward)
        
        if t % LOG_INTERVAL == 0:
            # Evaluate Jaccard
            recommended_arms = bandit.get_top_arms()
            jaccard = env.jaccard_metric(recommended_arms)
            
            # Compute UQ
            means = bandit.posterior.get_mean()
            stds = bandit.posterior.get_stds()
            
            posterior_df = pd.DataFrame({
                "means": list(means),
                "stds": list(stds)
            })
            
            uq = uncertainty_quantification(posterior_df)
            
            results.append({
                "run": seed,
                "timestep": t,
                "jaccard": jaccard,
                "uq": uq
            })
            
    return results


def main():
    os.makedirs("results", exist_ok=True)
    
    all_results = []
    with Pool() as pool:
        for res in tqdm(pool.imap_unordered(run_single_experiment, range(NUM_RUNS)), total=NUM_RUNS):
            all_results.extend(res)
            
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"Saved extended results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
