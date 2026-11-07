#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --job-name=sc4001-B-smoke
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
# Two optimiser steps per method on the real model and data: plumbing and memory only.
source scripts/activate.sh
nvidia-smi
common=(--budget 400 --seed 13 --max-steps 2)
python -m sc4001_b.train --method M3 --lr 2e-5 --out runs/smoke-M3 "${common[@]}"
python -m sc4001_b.train --method M6 --lr 2e-5 --lam 1 --out runs/smoke-M6 "${common[@]}"
python -m sc4001_b.train --method L2SP --lr 2e-5 --lam 0.1 --out runs/smoke-L2SP "${common[@]}"
python -m sc4001_b.train --method M4S1 --lr 1e-3 --out runs/smoke-M4S1 "${common[@]}"
python -m sc4001_b.train --method M4 --lr 2e-5 --stage1 runs/smoke-M4S1 --out runs/smoke-M4 "${common[@]}"
echo SMOKE_PASSED
