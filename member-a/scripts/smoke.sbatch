#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=sc4001-smoke
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate.sh
nvidia-smi
python tests/check_model.py
python -m sc4001_a.train --method M1 --budget 400 --seed 13 --lr 0.001 --max-steps 2 --out runs/smoke-M1
python -m sc4001_a.train --method M2 --budget 400 --seed 13 --lr 0.00002 --max-steps 2 --out runs/smoke-M2
