#!/bin/bash
#SBATCH --job-name=mots-dist
#SBATCH --partition=batch
#SBATCH --exclude=master
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-23
#SBATCH --output=hpc/logs/%x-%A_%a.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/DENV-Bandits}"
source "$HOME/.venv/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

# Install any missing dependencies
pip install --user paretoset pymoo numba pyarrow 2>/dev/null || true

# Read parameters for this array task
PARAMS_FILE="$REPO_DIR/hpc/experiment_params.txt"
# Skip comment/empty lines, pick line by array task ID
LINE=$(grep -v '^#' "$PARAMS_FILE" | grep -v '^$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

ENV=$(echo "$LINE" | awk '{print $1}')
DIST=$(echo "$LINE" | awk '{print $2}')
ALGS=$(echo "$LINE" | awk '{print $3}')

RESULTS_DIR="$REPO_DIR/results_hpc/${DIST}"
mkdir -p "$RESULTS_DIR"

echo "=== Task $SLURM_ARRAY_TASK_ID ==="
echo "Environment: $ENV"
echo "Distribution: $DIST"
echo "Algorithms: $ALGS"
echo "Running on $(hostname) with $(nproc) CPUs"
echo "Start: $(date)"

python momab_main.py \
    --algorithms "$ALGS" \
    --environments "$ENV" \
    --budgets 5000 \
    --num_runs 25 \
    --step 1 \
    --dist "$DIST" \
    --results_dir "$RESULTS_DIR"

echo "Done: $(date)"
