#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=sc4001-B-check
#SBATCH --output=logs/check_%j.out
#SBATCH --error=logs/check_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source scripts/activate.sh
nvidia-smi
python -m pip check
python - <<'PY'
import torch, transformers
assert torch.cuda.is_available(), 'GPU unavailable in allocated node'
print('torch', torch.__version__, 'transformers', transformers.__version__, 'GPU', torch.cuda.get_device_name())
PY
cd tests && python -m unittest discover -p 'test_*.py' -v
