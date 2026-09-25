#!/bin/bash
#SBATCH --job-name=poisson_10k
#SBATCH --partition=batch
#SBATCH --nodelist=node102
#SBATCH --cpus-per-task=60
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=hpc/logs/%x-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/DENV-Bandits}"
source "$HOME/.venv/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

RESULTS_DIR="$REPO_DIR/results_poisson"
mkdir -p "$RESULTS_DIR"

echo "Running on $(hostname) with $(nproc) CPUs"
echo "Results → $RESULTS_DIR"
echo "Start: $(date)"

python momab_main.py \
    --algorithms EGE_SR,EGE_SH,Uniform,TTPFTS_POISSON \
    --environments PoissonExp \
    --budgets 10000 \
    --num_runs 50 \
    --n_jobs 60 \
    --results_dir "$RESULTS_DIR"

echo "Done: $(date)"
