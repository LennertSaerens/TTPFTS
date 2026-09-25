#!/bin/bash
#SBATCH --job-name=mots-100
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

pip install --user paretoset pymoo numba pyarrow 2>/dev/null || true

# Read parameters for this array task
PARAMS_FILE="$REPO_DIR/hpc/experiment_params.txt"
LINE=$(grep -v '^#' "$PARAMS_FILE" | grep -v '^$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")

ENV=$(echo "$LINE" | awk '{print $1}')
DIST=$(echo "$LINE" | awk '{print $2}')
ALGS=$(echo "$LINE" | awk '{print $3}')

RESULTS_DIR="$REPO_DIR/results_hpc/${DIST}"
mkdir -p "$RESULTS_DIR"

# Rename existing 25-run files to 100-run naming so the resume logic picks them up
ALGS_JOINED=$(echo "$ALGS" | tr ',' '_')
OLD_FILE="${RESULTS_DIR}/${ENV}_${ALGS_JOINED}_5000_25.csv"
NEW_FILE="${RESULTS_DIR}/${ENV}_${ALGS_JOINED}_5000_100.csv"

if [ -f "$OLD_FILE" ] && [ ! -f "$NEW_FILE" ]; then
    echo "Renaming $OLD_FILE -> $NEW_FILE"
    mv "$OLD_FILE" "$NEW_FILE"
elif [ -f "$OLD_FILE" ] && [ -f "$NEW_FILE" ]; then
    echo "Both files exist — appending old data to new file"
    # Skip header from old file if new file already has header
    tail -n +2 "$OLD_FILE" >> "$NEW_FILE"
    rm "$OLD_FILE"
fi

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
    --num_runs 100 \
    --step 1 \
    --dist "$DIST" \
    --results_dir "$RESULTS_DIR"

echo "Done: $(date)"
