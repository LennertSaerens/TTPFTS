#!/bin/bash
# Run held-out random-instance suite (R1 + R2) and UV posterior shard rescoring on HPC.
#SBATCH --job-name=momab-uq-rand
#SBATCH --partition=batch
#SBATCH --nodelist=node102,node103
#SBATCH --exclude=master
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=hpc/logs/%x-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/DENV-Bandits-uq}"
RESULTS_ROOT="${RESULTS_ROOT:-$REPO_DIR/results_uq_known_var}"
UV_RESULTS_DIR="${UV_RESULTS_DIR:-$REPO_DIR/results_uq_hardness_7bc58a101c75}"

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
N_JOBS="${SLURM_CPUS_PER_TASK:-48}"
mkdir -p "$RESULTS_DIR" hpc/logs

echo "=== Held-Out Random Instances Suite (R1 + R2) ==="
echo "Host: $(hostname) | CPUs: $N_JOBS | Output: $RESULTS_DIR"
python -m experiments.uq_kv.random_instances \
    --output-dir "$RESULTS_DIR/random" \
    --mode all \
    --n-jobs "$N_JOBS"

if [[ -d "$UV_RESULTS_DIR" ]]; then
    echo "=== Re-scoring TTPFTS_UNKNOWN_VAR Shards ==="
    python -m experiments.uq_kv.rescore_uv \
        --uv-results-dir "$UV_RESULTS_DIR" \
        --output-dir "$RESULTS_DIR/uv_rescored" \
        --n-jobs "$N_JOBS"
fi

echo "Done: $(date --iso-8601=seconds)"
