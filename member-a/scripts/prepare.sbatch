#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=sc4001-data
#SBATCH --output=logs/data_%j.out
#SBATCH --error=logs/data_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate.sh
python -m sc4001_a.prepare "$@"
