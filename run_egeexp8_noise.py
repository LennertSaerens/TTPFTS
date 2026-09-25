import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

from environments.EgeExp8 import EgeExp8
from bandits.TTPFTS import TTPFTSBandit
from uncertainty_quantification import uncertainty_quantification

NUM_RUNS = 100
MAX_BUDGET = 5000
LOG_INTERVAL = 100
NOISE_SCALES = [1.0, 0.5, 0.25, 0.1, 0.05]
RESULTS_FILE = "results/egeexp8_noise_uq.csv"


def run_single_experiment(args):
    seed, noise_scale = args
    np.random.seed(seed)
    env = EgeExp8(noise_scale=noise_scale)
    
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
            recommended_arms = bandit.get_top_arms()
            jaccard = env.jaccard_metric(recommended_arms)
            
            means = bandit.posterior.get_mean()
            stds = bandit.posterior.get_stds()
            
            posterior_df = pd.DataFrame({
                "means": list(means),
                "stds": list(stds)
            })
            
            uq = uncertainty_quantification(posterior_df)
            
            results.append({
                "noise_scale": noise_scale,
                "run": seed,
                "timestep": t,
                "jaccard": jaccard,
                "uq": uq
            })
            
    return results


def main():
    os.makedirs("results", exist_ok=True)
    
    all_results = []
    tasks = [(seed, ns) for ns in NOISE_SCALES for seed in range(NUM_RUNS)]
    
    with Pool() as pool:
        for res in tqdm(pool.imap_unordered(run_single_experiment, tasks), total=len(tasks)):
            all_results.extend(res)
            
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_FILE, index=False)
    print(f"Saved noise results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
