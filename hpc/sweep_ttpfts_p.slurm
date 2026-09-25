#!/bin/bash
#SBATCH --job-name=ttpfts_p_sweep
#SBATCH --partition=batch
#SBATCH --exclude=master
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-87
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

# Define parameter grid
ENVIRONMENTS=(EgeExp1 EgeExp2 EgeExp3 EgeExp4 EgeExp5 EgeExp6 EgeExp7 EgeExp8)
P_VALUES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Determine current parameters from SLURM_ARRAY_TASK_ID
ENV_IDX=$((SLURM_ARRAY_TASK_ID % 8))
P_IDX=$((SLURM_ARRAY_TASK_ID / 8))

ENV_NAME=${ENVIRONMENTS[$ENV_IDX]}
P_VAL=${P_VALUES[$P_IDX]}

# Set working directory
REPO_DIR="${REPO_DIR:-$HOME/DENV-Bandits}"

set +u
source /etc/profile
module use /srv/cluster/software/x86-64-v3/easybuild/2023b/modules/all/ || true
module load Python/3.11.5-GCCcore-13.2.0 || true
set -u

# Activate virtual environment
source "$HOME/.venv/bin/activate"

# Change to project root and set PYTHONPATH
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

# Ensure output directories exist
mkdir -p results
mkdir -p logs

echo "Running TTPFTS_GKV on $ENV_NAME with p=$P_VAL on $(hostname)"

# Execute benchmark
python momab_main.py \
    --algorithms TTPFTS_GKV \
    --environments "$ENV_NAME" \
    --p "$P_VAL" \
    --num_runs 100 \
    --results_dir results
