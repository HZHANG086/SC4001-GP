#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=sc4001-A
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate.sh
# All checks below execute inside the allocated compute node.
nvidia-smi
python -m sc4001_a.train "$@"
