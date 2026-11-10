#!/bin/bash
#SBATCH --partition=6000ada
#SBATCH --job-name=sc4001-gp-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=member-c/logs/smoke-%j.out
#SBATCH --error=member-c/logs/smoke-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source member-c/scripts/ada/activate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
trap 'code=$?; echo "EXIT_CODE=$code"' EXIT
nvidia-smi
python -m sc4001_c.lora_train --data member-c/artifacts/data-v1 \
  --data-reference member-a/remote-records/data-v1 --protocol member-c/protocol/M5-v1.json \
  --out member-c/runs/M5-smoke --budget 400 --seed 13 --lr 0.0003 --smoke
