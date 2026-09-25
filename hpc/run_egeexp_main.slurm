#!/bin/bash
# Main baseline comparison on the eight EgeExp environments, re-run with the
# patched code base (correct --step semantics, seeded RNGs, fixed EGE
# initialisation/tie-breaking, paretoset(distinct=False)).
#
# Grid: 8 environments x {TTPFTS_UNI, Uniform, EGE_SR, EGE_SH, APE}
#       100 trajectories, 5000 arm pulls, every timestep logged.
#SBATCH --job-name=egeexp-main
#SBATCH --partition=batch
#SBATCH --exclude=master
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=1-8
#SBATCH --output=hpc/logs/%x-%A_%a.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/DENV-Bandits}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/results_fixed}"
SEED="${SEED:-0}"
NUM_RUNS="${NUM_RUNS:-100}"
BUDGET="${BUDGET:-5000}"
DIST="${DIST:-gaussian}"
ALGS="${ALGS:-TTPFTS_UNI,Uniform,EGE_SR,EGE_SH,APE}"

# ~/.venv is built on the EasyBuild Python 3.11.5 interpreter, so that module
# must be loaded first or the interpreter cannot find libpython3.11.so.1.0.
# Batch scripts do not source /etc/profile, so Lmod has to be initialised here.
set +u
source /etc/profile.d/lmod.sh
module use /srv/cluster/software/x86-64-v3/easybuild/2023b/modules/all/
module load Python/3.11.5-GCCcore-13.2.0
set -u

source "$HOME/.venv/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

# The workload is parallelised over trajectories by joblib; keep the BLAS
# backends single-threaded so the workers do not oversubscribe the allocation.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -c "import paretoset, pymoo, joblib, pandas, numpy, tqdm, scipy" \
    || pip install paretoset pymoo joblib pandas numpy tqdm scipy

ENV_NAME="EgeExp${SLURM_ARRAY_TASK_ID}"
N_JOBS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$RESULTS_DIR"

echo "=== Task $SLURM_ARRAY_TASK_ID ==="
echo "Environment:  $ENV_NAME"
echo "Algorithms:   $ALGS"
echo "Distribution: $DIST"
echo "Budget:       $BUDGET   Runs: $NUM_RUNS   Seed: $SEED"
echo "Results dir:  $RESULTS_DIR"
echo "Commit:       $(git rev-parse HEAD 2>/dev/null || echo 'n/a')"
# The job runs the rsynced working tree, which may differ from the commit above,
# so fingerprint the sources that actually determine the results.
echo "Source hash:  $(find momab_main.py bandits environments -name '*.py' \
    | sort | xargs cat | sha256sum | cut -c1-16)"
echo "Host:         $(hostname) with $(nproc) CPUs, n_jobs=$N_JOBS"
echo "Start:        $(date)"

python momab_main.py \
    --algorithms "$ALGS" \
    --environments "$ENV_NAME" \
    --budgets "$BUDGET" \
    --num_runs "$NUM_RUNS" \
    --step 1 \
    --dist "$DIST" \
    --seed "$SEED" \
    --n_jobs "$N_JOBS" \
    --results_dir "$RESULTS_DIR"

echo "Done: $(date)"
