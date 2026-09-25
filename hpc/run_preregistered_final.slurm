#!/bin/bash
# Preregistered final MOMAB experiment, one SLURM array task per EgeExp
# environment. This script is generated from/validated against
# experiments/final_config.json; do not edit budgets or algorithms here.
#SBATCH --job-name=momab-final
#SBATCH --partition=batch
#SBATCH --nodelist=node102,node103
#SBATCH --exclude=master
#SBATCH --cpus-per-task=25
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=hpc/logs/%x-%A_%a.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/DENV-Bandits-final}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/experiments/final_config.json}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results_preregistered_final}"
NUM_RUNS="${NUM_RUNS:-100}"

# Fail immediately and legibly if REPO_DIR points at the wrong tree. Without
# this, a stale REPO_DIR surfaces only as a confusing ModuleNotFoundError from
# the config loader, and a scientific run must never start from the wrong
# source tree.
for required in "$CONFIG_PATH" "$REPO_DIR/momab_main.py" "$REPO_DIR/experiments/final_config_loader.py"; do
    if [[ ! -f "$required" ]]; then
        echo "FATAL: expected file missing: $required" >&2
        echo "       REPO_DIR=$REPO_DIR -- set REPO_DIR to the preregistered source tree." >&2
        exit 1
    fi
done

# ~/.venv is built on the EasyBuild Python 3.11.5 interpreter. Batch shells do
# not source /etc/profile, so initialise Lmod and load that exact interpreter
# before activating the venv; otherwise libpython3.11.so.1.0 is missing.
set +u
source /etc/profile.d/lmod.sh
module use /srv/cluster/software/x86-64-v3/easybuild/2023b/modules/all/
module load Python/3.11.5-GCCcore-13.2.0
set -u

source "$HOME/.venv/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

# Parallelism is at the repetition level via joblib; keep BLAS single-threaded.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python - <<'PY'
import importlib
missing = []
for name in ["numpy", "pandas", "joblib", "tqdm", "scipy", "paretoset", "pymoo"]:
    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - fail-fast dependency report
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("Missing required Python dependencies; refusing to pip install in scientific job:\n" + "\n".join(missing))
PY

CONFIG_HASH="$(python -m experiments.final_config_loader --config "$CONFIG_PATH" --hash)"
ENV_NAME="$(
python - "$CONFIG_PATH" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)
idx = int(sys.argv[2])
print(cfg["environments"][idx]["name"])
PY
)"
BUDGET="$(
python - "$CONFIG_PATH" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = json.load(f)
idx = int(sys.argv[2])
print(cfg["environments"][idx]["budget"])
PY
)"

RESULTS_DIR="${RESULTS_ROOT}_${CONFIG_HASH:0:12}"
RUN_DIR="$RESULTS_DIR/run"
N_JOBS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$RESULTS_DIR" hpc/logs

echo "=== Preregistered MOMAB final experiment ==="
echo "Array task:   ${SLURM_ARRAY_TASK_ID}"
echo "Environment:  ${ENV_NAME}"
echo "Budget:       ${BUDGET}"
echo "Config:       ${CONFIG_PATH}"
echo "Config hash:  ${CONFIG_HASH}"
echo "Results dir:  ${RESULTS_DIR}"
echo "Run dir:      ${RUN_DIR}"
echo "Commit:       $(git rev-parse HEAD 2>/dev/null || echo 'n/a')"
echo "Dirty files:  $(git status --short 2>/dev/null | wc -l | tr -d ' ')"
echo "Host:         $(hostname) with $(nproc) CPUs, n_jobs=${N_JOBS}"
echo "Start:        $(date --iso-8601=seconds)"

read -r -a MOMAB_ARGS <<< "$(python -m experiments.final_config_loader \
    --config "$CONFIG_PATH" \
    --print-momab-args \
    --results-dir "$RESULTS_DIR" \
    --run-dir "$RUN_DIR" \
    --n-jobs "$N_JOBS" \
    --num-runs "$NUM_RUNS" \
    --environments "$ENV_NAME" \
    --budgets "$BUDGET")"

python momab_main.py "${MOMAB_ARGS[@]}"

echo "Done: $(date --iso-8601=seconds)"
