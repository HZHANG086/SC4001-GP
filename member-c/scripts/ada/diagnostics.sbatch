#!/bin/bash
#SBATCH --partition=6000ada
#SBATCH --job-name=sc4001-gp-diag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=member-c/logs/diag-%j.out
#SBATCH --error=member-c/logs/diag-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source member-c/scripts/ada/activate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
finish() {
  code=$?
  python -m sc4001_c.job_record --event exit --exit-code "$code" || true
  echo "EXIT_CODE=$code"
  exit "$code"
}
trap finish EXIT
python -m sc4001_c.job_record --event start
python - <<'PY'
from pathlib import Path
from sc4001_c.common import load,digest
spec=load('member-c/protocol/M5-diagnostics-v1.json')
for name,expected in spec['code_hashes'].items():
    assert digest(Path('member-c/src/sc4001_c')/name)==expected, name
print('DIAGNOSTIC_CODE_HASHES_VERIFIED')
PY
python -m sc4001_c.m5_audit
python -m sc4001_c.m5_cka
for repeat in 1 2 3; do
  for method in M1 M2 M3 M4 M6 L2SP; do
    python -m sc4001_c.benchmark --data member-c/artifacts/data-v1 \
      --data-reference member-a/remote-records/data-v1 --method "$method" --repeat "$repeat" \
      --out "member-c/results/Ada-cost/$method/clean/repeat-$repeat"
  done
  python -m sc4001_c.m5_benchmark --repeat "$repeat" \
    --out "member-c/results/Ada-cost/M5/clean/repeat-$repeat"
done
python -m sc4001_c.m5_collect
python -m sc4001_c.m5_package
echo 'M5_AUDIT_CKA_AND_MATCHED_ADA_COST_COMPLETE'
