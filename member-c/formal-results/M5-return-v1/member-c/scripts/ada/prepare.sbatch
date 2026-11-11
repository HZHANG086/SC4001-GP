#!/bin/bash
#SBATCH --partition=6000ada
#SBATCH --job-name=sc4001-gp-data
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=member-c/logs/data-%j.out
#SBATCH --error=member-c/logs/data-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source member-c/scripts/ada/activate.sh
trap 'code=$?; echo "EXIT_CODE=$code"' EXIT
python -m sc4001_a.prepare --out member-c/artifacts/data-v1 \
  --source-lock member-a/remote-records/data-v1/sources.json
python - <<'PY'
from pathlib import Path
from sc4001_c.cka import verify_data
print('DATA_MATCHES_A',verify_data(Path('member-c/artifacts/data-v1'),Path('member-a/remote-records/data-v1')))
PY
