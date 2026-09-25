#!/bin/bash
#SBATCH --job-name=ape_experiment
#SBATCH --partition=batch
#SBATCH --exclude=master
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

ENVIRONMENTS=(EgeExp1 EgeExp2 EgeExp3 EgeExp4 EgeExp5 EgeExp6 EgeExp7 EgeExp8)
ENV_NAME=${ENVIRONMENTS[$SLURM_ARRAY_TASK_ID]}

REPO_DIR="${REPO_DIR:-$HOME/DENV-Bandits}"

set +u
source /etc/profile
module use /srv/cluster/software/x86-64-v3/easybuild/2023b/modules/all/ || true
module load Python/3.11.5-GCCcore-13.2.0 || true
set -u

source "$HOME/.venv/bin/activate"

cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

mkdir -p results
mkdir -p logs

echo "Benchmarking APE on $ENV_NAME on $(hostname)"

python momab_main.py \
    --algorithms APE \
    --environments "$ENV_NAME" \
    --num_runs 100 \
    --results_dir results
