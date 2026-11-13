#!/usr/bin/env bash
#SBATCH --job-name=cs550-smoke
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=cs550-smoke-%j.log
# No partition/account specified: uses your configured defaults. If none, consult your allocation.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
echo "Node: $(hostname)"
echo "Project: $(pwd)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
bash scripts/verify_linux.sh
