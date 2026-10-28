#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=sc4001-check
#SBATCH --output=logs/check_%j.out
#SBATCH --error=logs/check_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate.sh
nvidia-smi
python -m pip check
python - <<'PY'
import os, sys
assert sys.prefix == os.environ['CONDA_PREFIX'], (sys.prefix, os.environ['CONDA_PREFIX'])
import torch, numpy
assert torch.__file__.startswith(sys.prefix + '/'), torch.__file__
assert numpy.__file__.startswith(sys.prefix + '/'), numpy.__file__
print('Python prefix:', sys.prefix)
print('torch path:', torch.__file__)
print(torch.__config__.show())
print('torch', torch.__version__, 'cuda build', torch.version.cuda)
assert torch.cuda.is_available(), 'GPU unavailable in allocated node'
print('GPU', torch.cuda.get_device_name())
x = torch.randn(128, 128, device='cuda')
print('CUDA matmul finite:', torch.isfinite(x @ x).all().item())
PY
python -m unittest discover -s tests -v
python tests/check_model.py
