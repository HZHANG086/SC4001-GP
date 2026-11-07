#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --job-name=sc4001-B-pilot
#SBATCH --output=logs/pilot_%j.out
#SBATCH --error=logs/pilot_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
# Source-only development pilot (IMDb validation only; no test or target data are read).
# Purpose: full-run time and memory, and regulariser magnitudes to confirm the lambda grids.
source scripts/activate.sh
nvidia-smi
cell=(--budget 400 --seed 13 --stage pilot)
python -m sc4001_b.train --method M3 --lr 2e-5 --out runs/pilot-M3-b400 "${cell[@]}"
python -m sc4001_b.train --method M6 --lr 2e-5 --lam 1 --out runs/pilot-M6-b400 "${cell[@]}"
python -m sc4001_b.train --method L2SP --lr 2e-5 --lam 0.1 --out runs/pilot-L2SP-b400 "${cell[@]}"
# Throughput at the largest budget: 200 optimiser steps of M3 and M6 (marked smoke).
python -m sc4001_b.train --method M3 --lr 2e-5 --budget 6400 --seed 13 --max-steps 200 --out runs/pilot-M3-b6400-steps
python -m sc4001_b.train --method M6 --lr 2e-5 --lam 1 --budget 6400 --seed 13 --max-steps 200 --out runs/pilot-M6-b6400-steps
python - <<'PY'
import json
from pathlib import Path
print('| run | val acc | train s | peak GiB | val drift | last regulariser |')
for run in sorted(Path('runs').glob('pilot-*')):
    v = json.loads((run / 'validation.json').read_text())
    hist = json.loads((run / 'history.json').read_text())
    reg = [h[k] for h in hist for k in h if k.startswith('train_') and k not in ('train_loss', 'train_runtime',
           'train_samples_per_second', 'train_steps_per_second')]
    print(f"| {run.name} | {v['eval_accuracy']:.4f} | {v['train_wall_seconds_this_invocation']:.0f} | "
          f"{v['peak_cuda_allocated_bytes'] / 2**30:.2f} | {v.get('val_feature_distance', float('nan')):.4f} | "
          f"{reg[-1] if reg else '-'} |")
PY
