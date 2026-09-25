"""
Compare three uncertainty quantification measures for Pareto Set Identification:
  1. Bhattacharyya coefficient (existing)
  2. Symmetric KL-divergence
  3. Posterior entropy

Loads pre-computed TTPFTS posteriors from results_corr/posteriors/ and
PFI metrics from results_corr/pfi_metrics/.

Produces:
  - Time-evolution plots for all three measures (side-by-side)
  - Per-experiment correlation boxplots with the Jaccard metric
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt

from uncertainty_quantification import uncertainty_quantification_all
from plotting import plot_uq_comparison, plot_uq_correlation_boxplots, plot_uq_combined

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = "results_corr/posteriors/"
PFI_DIR = "results_corr/pfi_metrics/"
FIGURES_DIR = "figures/"

ENVIRONMENTS = [f"EgeExp{i}" for i in range(1, 9)]
EXPERIMENTS = np.arange(100)
TIMESTEPS = np.arange(100, 5001, 100)         # 50 timesteps

MEASURE_NAMES = ("Bhattacharyya Coeff.", "Symmetric KL-Divergence", "Posterior Entropy")

# ── 1. Load posteriors ─────────────────────────────────────────────────────
print("Loading posterior parquet files …")
env_post_dfs = {}
for env in ENVIRONMENTS:
    env_post_dfs[env] = [
        [
            pd.read_parquet(
                f"{DATA_DIR}TTPFTS_post_{env}_e{e}_t{t}.parquet",
                columns=["arm", "means", "stds"],
            )
            for t in TIMESTEPS
        ]
        for e in EXPERIMENTS
    ]
print("  done.")

# ── 2. Compute all three measures in a single pass ─────────────────────────
print("Computing UQ measures …")
env_measures = {name: {} for name in MEASURE_NAMES}

for env in ENVIRONMENTS:
    n_exp = len(EXPERIMENTS)
    n_ts = len(TIMESTEPS)
    bc_arr = np.zeros((n_exp, n_ts))
    kl_arr = np.zeros((n_exp, n_ts))
    ent_arr = np.zeros((n_exp, n_ts))

    for i in range(n_exp):
        for j in range(n_ts):
            bc, kl, ent = uncertainty_quantification_all(env_post_dfs[env][i][j])
            bc_arr[i, j] = bc
            kl_arr[i, j] = kl
            ent_arr[i, j] = ent
        if (i + 1) % 25 == 0:
            print(f"  {env}: {i + 1}/{n_exp} experiments")

    env_measures[MEASURE_NAMES[0]][env] = bc_arr
    env_measures[MEASURE_NAMES[1]][env] = kl_arr
    env_measures[MEASURE_NAMES[2]][env] = ent_arr

print("  done.")

# ── 3. Time-evolution plots ────────────────────────────────────────────────
print("Generating time-evolution comparison plot …")
plt.rcParams.update({"font.size": 12, "font.family": "serif"})

plot_uq_comparison(
    env_measures,
    TIMESTEPS,
    measure_names=MEASURE_NAMES,
    save_png=True,
    save_file=f"{FIGURES_DIR}uq_comparison_evolution.png",
)
print(f"  saved to {FIGURES_DIR}uq_comparison_evolution.png")

# ── 4. Correlation with Jaccard ────────────────────────────────────────────
print("Computing per-experiment correlations with Jaccard …")

# Load PFI metrics (old format: no header)
pfi_dfs = {}
for env in ENVIRONMENTS:
    df = pd.read_csv(
        f"{PFI_DIR}TTPFTS_{env}_100_5000.csv",
        names=["algorithm", "experiment", "arm_pull", "bernoulli", "jaccard", "mis_id"],
        header=None,
    )
    pfi_dfs[env] = df[df["arm_pull"].isin(TIMESTEPS)]

# For each measure, compute per-experiment Pearson correlation with Jaccard
all_envs_corr_dfs = {}
for m_idx, measure_name in enumerate(MEASURE_NAMES):
    rows = []
    for env in ENVIRONMENTS:
        uq_vals = env_measures[measure_name][env]   # (n_exp, n_ts)
        pfi = pfi_dfs[env]
        for e in EXPERIMENTS:
            exp_pfi = pfi[pfi["experiment"] == e]
            jaccard_vals = exp_pfi["jaccard"].values
            uq_exp = uq_vals[e]
            if len(jaccard_vals) == len(uq_exp) and len(jaccard_vals) > 1:
                # Skip experiments where either series is constant (yields NaN)
                if np.std(jaccard_vals) < 1e-12 or np.std(uq_exp) < 1e-12:
                    continue
                corr = np.corrcoef(uq_exp, jaccard_vals)[0, 1]
                rows.append({"experiment": e, "environment": env, "jaccard": corr})
    all_envs_corr_dfs[measure_name] = pd.DataFrame(rows)

print("  done.")

# ── 5. Correlation boxplots ────────────────────────────────────────────────
print("Generating correlation boxplots …")
plot_uq_correlation_boxplots(
    all_envs_corr_dfs,
    measure_names=MEASURE_NAMES,
    environments=ENVIRONMENTS,
    save_png=True,
    save_file=f"{FIGURES_DIR}uq_comparison_correlation.png",
)
print(f"  saved to {FIGURES_DIR}uq_comparison_correlation.png")

# ── 6. Print summary statistics ────────────────────────────────────────────
print("\n=== Median per-experiment correlation with Jaccard ===")
for name in MEASURE_NAMES:
    df = all_envs_corr_dfs[name]
    med = df["jaccard"].median()
    mean = df["jaccard"].mean()
    print(f"  {name:30s}  median={med:+.4f}  mean={mean:+.4f}")

print("\nDone.")

# ── 7. Individual evolution plots (matching existing BC style) ─────────────
print("\nGenerating individual UQ evolution plots …")
from plotting import plot_uncertainty_all_envs

labels_and_files = [
    (MEASURE_NAMES[0], "Uncertainty (Bhattacharyya Avg)", "uq_bhattacharyya_evolution.png"),
    (MEASURE_NAMES[1], "Symmetric KL-Divergence", "uq_kl_divergence_evolution.png"),
    (MEASURE_NAMES[2], "Posterior Entropy (nats)", "uq_entropy_evolution.png"),
]

for measure_name, ylabel, filename in labels_and_files:
    plot_uncertainty_all_envs(
        env_measures[measure_name],
        TIMESTEPS,
        ylabel=ylabel,
        title=measure_name,
        save_png=True,
        save_file=f"{FIGURES_DIR}{filename}",
    )
    print(f"  saved to {FIGURES_DIR}{filename}")

print("\nAll plots generated.")

# ── 8. Combined 2×3 figure ─────────────────────────────────────────────────
print("\nGenerating combined 2×3 figure …")
plot_uq_combined(
    env_measures,
    all_envs_corr_dfs,
    TIMESTEPS,
    measure_names=MEASURE_NAMES,
    save_png=True,
    save_file=f"{FIGURES_DIR}uq_comparison_combined.png",
)
print(f"  saved to {FIGURES_DIR}uq_comparison_combined.png")

