#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=sc4001-pilot
#SBATCH --output=logs/pilot_%j.out
#SBATCH --error=logs/pilot_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate.sh
for length in 256 512; do
    python -m sc4001_a.train --method M1 --budget 400 --seed 13 --lr 0.001 \
        --length "$length" --stage pilot --out "runs/pilot-M1-b400-s13-l${length}"
    python -m sc4001_a.train --method M2 --budget 400 --seed 13 --lr 0.00002 \
        --length "$length" --stage pilot --out "runs/pilot-M2-b400-s13-l${length}"
done
