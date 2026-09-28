#!/bin/bash
# Exact posterior-state replay for the final hardness-budget TTPFTS experiment.
# One array task owns one environment/repetition chunk (10 trajectories).
#SBATCH --job-name=momab-uq
#SBATCH --partition=batch
#SBATCH --nodelist=node102,node103
#SBATCH --exclude=master
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=02:00:00
#SBATCH --array=0-79%16
#SBATCH --output=hpc/logs/%x-%A_%a.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/DENV-Bandits-final}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/experiments/uq/config.json}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results_uq_hardness}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"

for required in "$CONFIG_PATH" "$REPO_DIR/experiments/uq/replay.py" "$REPO_DIR/SOURCE_PROVENANCE.txt"; do
    if [[ ! -f "$required" ]]; then
        echo "FATAL: required UQ replay file missing: $required" >&2
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

SOURCE_COMMIT="$(awk -F'= *' '/^commit/{print $2}' SOURCE_PROVENANCE.txt)"
SOURCE_DIRTY="$(awk -F'= *' '/^dirty_files/{print $2}' SOURCE_PROVENANCE.txt)"
if [[ -z "$SOURCE_COMMIT" || "$SOURCE_DIRTY" != "0" ]]; then
    echo "FATAL: invalid source provenance; refusing an unattributable UQ run." >&2
    exit 1
fi

read -r UQ_HASH ENVIRONMENTS NUM_RUNS <<EOF
$(python - <<'PY'
from experiments.final_config_loader import load_final_config
from experiments.uq.config import config_hash, load_config
cfg = load_config()
print(config_hash(cfg), ",".join(x["name"] for x in load_final_config()["environments"]), cfg["num_runs"])
PY
)
EOF
IFS=',' read -r -a ENV_ARRAY <<< "$ENVIRONMENTS"
ENV_INDEX=$((SLURM_ARRAY_TASK_ID / (NUM_RUNS / CHUNK_SIZE)))
CHUNK_INDEX=$((SLURM_ARRAY_TASK_ID % (NUM_RUNS / CHUNK_SIZE)))
ENV_NAME="${ENV_ARRAY[$ENV_INDEX]}"
START=$((CHUNK_INDEX * CHUNK_SIZE))
END=$((START + CHUNK_SIZE - 1))
REPETITIONS="$(seq -s, "$START" "$END")"
RESULTS_DIR="${RESULTS_ROOT}_${UQ_HASH:0:12}"
RUN_DIR="$RESULTS_DIR/run"

mkdir -p "$RESULTS_DIR" hpc/logs
echo "=== Hardness-budget UQ replay ==="
echo "Commit: $SOURCE_COMMIT"
echo "Config hash: $UQ_HASH"
echo "Host: $(hostname)"
echo "Array task: $SLURM_ARRAY_TASK_ID; environment: $ENV_NAME; repetitions: $REPETITIONS"
echo "Run dir: $RUN_DIR"

python -m experiments.uq.replay \
    --run-dir "$RUN_DIR" \
    --environments "$ENV_NAME" \
    --repetitions "$REPETITIONS" \
    --n-jobs 1
