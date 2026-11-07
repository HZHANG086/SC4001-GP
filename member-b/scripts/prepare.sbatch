#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=sc4001-B-prepare
#SBATCH --output=logs/prepare_%j.out
#SBATCH --error=logs/prepare_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
# Rebuild data-v1 with A-v1's exact Hub revisions, then check it file by file against A-v1.
source scripts/activate.sh
python -m sc4001_a.prepare --out artifacts/data-v1 \
    --source-lock ../member-a/remote-records/data-v1/sources.json
python - <<'PY'
from pathlib import Path
from sc4001_b.sweep import check_data_matches_a
check_data_matches_a(Path('artifacts/data-v1'), '../member-a/remote-records/data-v1/manifest_hashes.json')
print('DATA_MATCHES_A_V1')
PY
