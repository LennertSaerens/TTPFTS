#!/bin/bash
# Merge and independently validate a completed hardness-budget UQ replay.
# FINAL_RESULTS_DIR must point at the validated final PSI result directory.
#SBATCH --job-name=momab-uq-validate
#SBATCH --partition=batch
#SBATCH --nodelist=node102,node103
#SBATCH --exclude=master
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=hpc/logs/%x-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/DENV-Bandits-final}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results_uq_hardness}"
FINAL_RESULTS_DIR="${FINAL_RESULTS_DIR:?Set FINAL_RESULTS_DIR to validated final PSI results}"

set +u
source /etc/profile.d/lmod.sh
module use /srv/cluster/software/x86-64-v3/easybuild/2023b/modules/all/
module load Python/3.11.5-GCCcore-13.2.0
set -u
source "$HOME/.venv/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

UQ_HASH="$(python - <<'PY'
from experiments.uq.config import config_hash, load_config
print(config_hash(load_config()))
PY
)"
RESULTS_DIR="${RESULTS_ROOT}_${UQ_HASH:0:12}"
RUN_DIR="$RESULTS_DIR/run"
for required in "$RUN_DIR" "$FINAL_RESULTS_DIR/EgeExp1.csv" "$REPO_DIR/experiments/uq/validate.py"; do
    if [[ ! -e "$required" ]]; then
        echo "FATAL: required validation input missing: $required" >&2
        exit 1
    fi
done

python -m experiments.uq.merge --run-dir "$RUN_DIR" --results-dir "$RESULTS_DIR"
python -m experiments.uq.validate --run-dir "$RUN_DIR" --final-results-dir "$FINAL_RESULTS_DIR"
