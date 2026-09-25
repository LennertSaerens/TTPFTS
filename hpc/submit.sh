#!/bin/bash
# Sync code to HPC and submit experiments.
# Run this after loading your SSH key: ssh-add ~/.ssh/id_rsa

set -euo pipefail

HPC="lsaerens@comopc3.vub.ac.be"
REMOTE_DIR="DENV-Bandits"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Step 1: Syncing code to HPC ==="
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='.venv' --exclude='venv' --exclude='node_modules' \
  --exclude='results*' --exclude='figures' --exclude='.idea' \
  --exclude='.vscode' --exclude='*.egg-info' \
  "$LOCAL_DIR/" "$HPC:~/$REMOTE_DIR/"

echo ""
echo "=== Step 2: Creating logs directory ==="
ssh "$HPC" "mkdir -p ~/$REMOTE_DIR/hpc/logs ~/$REMOTE_DIR/results_hpc"

echo ""
echo "=== Step 3: Submitting SLURM array job ==="
ssh "$HPC" "cd ~/$REMOTE_DIR && sbatch hpc/run_experiments.slurm"

echo ""
echo "=== Step 4: Checking job queue ==="
ssh "$HPC" "squeue -u lsaerens"

echo ""
echo "Done! Monitor with: ssh $HPC 'squeue -u lsaerens'"
echo "Fetch results with: rsync -avz $HPC:~/$REMOTE_DIR/results_hpc/ results_hpc/"
