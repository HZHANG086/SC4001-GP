#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=05:50:00
#SBATCH --job-name=sc4001-formal
#SBATCH --output=logs/formal_%j.out
#SBATCH --error=logs/formal_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate.sh
export PYTHONPATH="$PWD/protocol/locked_src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
python -m sc4001_a.sweep --seeds "$@"
