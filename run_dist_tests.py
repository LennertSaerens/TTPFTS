"""
Small validation run: every environment × compatible distribution × algorithm.
10 runs, 500 arm pulls, step=10. One CSV per (environment, distribution).
Results saved to results_dist_test/.
"""
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

import csv
import os
import time
import numpy as np
from tqdm import tqdm

from bandits.Posteriors import TPosterior, NormalPosterior
from bandits.TTPFTS import TTPFTSBandit
from bandits.UCB import PUCB1Bandit
from bandits.Uniform import UniformBandit
from environments.EgeExp1 import EgeExp1
from environments.EgeExp2 import EgeExp2
from environments.EgeExp3 import EgeExp3
from environments.EgeExp4 import EgeExp4
from environments.EgeExp5 import EgeExp5
from environments.EgeExp6 import EgeExp6
from environments.EgeExp7 import EgeExp7
from environments.EgeExp8 import EgeExp8

NUM_RUNS = 10
MAX_BUDGET = 500
STEP = 10
RESULTS_DIR = "results_dist_test"
CSV_HEADER = ["algorithm", "experiment", "timestep", "bernoulli", "jaccard", "misidentification"]

# Environment classes with compatible distributions
ENV_DIST_COMPAT = {
    "EgeExp1": (EgeExp1, ["gaussian", "bernoulli", "exponential"]),
    "EgeExp2": (EgeExp2, ["gaussian", "bernoulli", "exponential"]),
    "EgeExp3": (EgeExp3, ["gaussian"]),  # has negative means
    "EgeExp4": (EgeExp4, ["gaussian", "bernoulli", "exponential"]),
    "EgeExp5": (EgeExp5, ["gaussian", "bernoulli", "exponential"]),
    "EgeExp6": (EgeExp6, ["gaussian"]),  # has negative means and means > 1
    "EgeExp7": (EgeExp7, ["gaussian", "bernoulli", "exponential"]),
    "EgeExp8": (EgeExp8, ["gaussian", "bernoulli", "exponential"]),
}

ALGORITHMS = ["TTPFTS_UNI", "Uniform", "PUCB1"]


def make_bandit(alg_name, num_arms, num_objectives, stds):
    if alg_name == "Uniform":
        return UniformBandit(num_arms, num_objectives)
    elif alg_name == "PUCB1":
        return PUCB1Bandit(num_arms, num_objectives)
    elif alg_name == "TTPFTS_UNI":
        return TTPFTSBandit(
            TPosterior(num_arms, num_objectives, alpha=-0.5),
            num_warmup_pulls=4)
    raise ValueError(f"Unknown algorithm: {alg_name}")


def run_single(env, bandit, alg_name, exp_id):
    env.reset()
    bandit.reset(env.stds)
    rows = []
    for t in range(MAX_BUDGET + 1):
        arm = bandit.choose_arm()
        reward = env.pull_arm(arm)
        bandit.learn(arm, reward)
        if t % STEP == 0:
            top = bandit.get_top_arms()
            rows.append([
                alg_name, exp_id, t,
                env.bernoulli_metric(top),
                env.jaccard_metric(top),
                env.mis_id_metric(top),
            ])
    return rows


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    total_combos = sum(len(dists) for _, (_, dists) in ENV_DIST_COMPAT.items())
    print(f"Running {total_combos} env×dist combos × {len(ALGORITHMS)} algorithms × {NUM_RUNS} runs")
    print(f"Budget={MAX_BUDGET}, step={STEP}\n")

    t0 = time.time()
    for env_name, (env_cls, dists) in ENV_DIST_COMPAT.items():
        for dist in dists:
            out_file = os.path.join(RESULTS_DIR, f"{env_name}_{dist}.csv")
            print(f"▸ {env_name} / {dist}")

            try:
                env = env_cls(dist=dist) if env_name not in ("N50VS",) else env_cls()
            except Exception as e:
                print(f"  ✗ Failed to create env: {e}")
                continue

            all_rows = []
            for alg_name in ALGORITHMS:
                bandit = make_bandit(alg_name, env.num_arms, env.num_objectives, env.stds)
                for exp_id in tqdm(range(NUM_RUNS), desc=f"  {alg_name}", unit="run", leave=False):
                    all_rows.extend(run_single(env, bandit, alg_name, exp_id))

            with open(out_file, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(CSV_HEADER)
                w.writerows(all_rows)
            print(f"  ✓ {len(all_rows)} rows → {out_file}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Results in {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
