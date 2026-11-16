#!/bin/bash
# Merge, validate against final_results_dir, and generate summary/plot outputs after replay + random complete.
#SBATCH --job-name=momab-uq-kv-post
#SBATCH --partition=batch
#SBATCH --nodelist=node102,node103
#SBATCH --exclude=master
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=hpc/logs/%x-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/DENV-Bandits-uq}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results_uq_known_var}"
FINAL_RESULTS_DIR="${FINAL_RESULTS_DIR:-$HOME/DENV-Bandits-final/results_preregistered_final_ff29cc456351}"

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
from experiments.uq_kv.config import config_hash, load_config
print(config_hash(load_config()))
PY
)"
RESULTS_DIR="${RESULTS_ROOT}_${UQ_HASH:0:12}"
RUN_DIR="$RESULTS_DIR/run"

echo "=== Merge & Validate Known-Variance UQ ==="
python -m experiments.uq_kv.merge_validate_kv \
    --run-dir "$RUN_DIR" \
    --results-dir "$RESULTS_DIR" \
    --final-results-dir "$FINAL_RESULTS_DIR"

echo "=== Running Analysis & Rendering Figures ==="
python -m experiments.uq_kv.plot \
    --results-dir "$RESULTS_DIR" \
    --random-trajectories "$RESULTS_DIR/random/random_trajectories.csv" \
    --output-dir "$RESULTS_DIR/report_and_figures"

echo "Done: $(date --iso-8601=seconds)"
