#!/bin/bash
#SBATCH --job-name=poisson_exp
#SBATCH --partition=batch
#SBATCH --exclude=master
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/DENV-Bandits}"
source "$HOME/.venv/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

RESULTS_DIR="$REPO_DIR/results_poisson"
mkdir -p "$RESULTS_DIR"

echo "Running on $(hostname) with $(nproc) CPUs"
echo "Results → $RESULTS_DIR"

python momab_main.py \
    --algorithms EGE_SR,EGE_SH,Uniform,TTPFTS_POISSON \
    --environments PoissonExp \
    --budgets 5000 \
    --num_runs 50 \
    --n_jobs 8 \
    --results_dir "$RESULTS_DIR"

echo "Done."
