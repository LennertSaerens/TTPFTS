#!/bin/bash
# Exact posterior-state replay for TTPFTS_KNOWN_VAR (NormalPosterior) across EgeExp1..8.
# One array task owns 5 repetitions of one environment (160 tasks total).
#SBATCH --job-name=momab-uq-kv
#SBATCH --partition=batch
#SBATCH --nodelist=node102,node103
#SBATCH --exclude=master
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=0-159
#SBATCH --output=hpc/logs/%x-%A_%a.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/DENV-Bandits-uq}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/experiments/uq_kv/config.json}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results_uq_known_var}"
CHUNK_SIZE="${CHUNK_SIZE:-5}"

for required in "$CONFIG_PATH" "$REPO_DIR/experiments/uq_kv/replay_kv.py" "$REPO_DIR/SOURCE_PROVENANCE.txt"; do
    if [[ ! -f "$required" ]]; then
        echo "FATAL: required UQ KV replay file missing: $required" >&2
        exit 1
    fi
done

set +u
source /etc/profile.d/lmod.sh
module use /srv/cluster/software/x86-64-v3/easybuild/2023b/modules/all/
module load Python/3.11.5-GCCcore-13.2.0
set -u
source "$HOME/.venv/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

read -r UQ_HASH ENVIRONMENTS NUM_RUNS <<EOF
$(python - <<'PY'
from experiments.final_config_loader import load_final_config
from experiments.uq_kv.config import config_hash, load_config
cfg = load_config()
print(config_hash(cfg), ",".join(x["name"] for x in load_final_config()["environments"]), cfg["num_runs"])
PY
)
EOF

IFS=',' read -r -a ENV_ARRAY <<< "$ENVIRONMENTS"
CHUNKS_PER_ENV=$((NUM_RUNS / CHUNK_SIZE))
ENV_INDEX=$((SLURM_ARRAY_TASK_ID / CHUNKS_PER_ENV))
CHUNK_INDEX=$((SLURM_ARRAY_TASK_ID % CHUNKS_PER_ENV))
ENV_NAME="${ENV_ARRAY[$ENV_INDEX]}"
START=$((CHUNK_INDEX * CHUNK_SIZE))
END=$((START + CHUNK_SIZE - 1))
REPETITIONS="$(seq -s, "$START" "$END")"
RESULTS_DIR="${RESULTS_ROOT}_${UQ_HASH:0:12}"
RUN_DIR="$RESULTS_DIR/run"

mkdir -p "$RESULTS_DIR" hpc/logs
echo "=== Known-Variance UQ replay ==="
echo "Host: $(hostname) | Env: $ENV_NAME | Reps: $REPETITIONS"

python -m experiments.uq_kv.replay_kv \
    --run-dir "$RUN_DIR" \
    --environments "$ENV_NAME" \
    --repetitions "$REPETITIONS" \
    --n-jobs 1
