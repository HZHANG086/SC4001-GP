#!/bin/bash
#SBATCH --partition=6000ada
#SBATCH --job-name=sc4001-gp-M5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=member-c/logs/M5-%j.out
#SBATCH --error=member-c/logs/M5-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source member-c/scripts/ada/activate.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
stage="${1:?Supply stage 0..5}"
[[ "$stage" =~ ^[0-5]$ ]] || { echo 'Invalid stage'; exit 2; }
finish() {
  code=$?
  python -m sc4001_c.job_record --event exit --exit-code "$code" || true
  echo "EXIT_CODE=$code stage=$stage"
  exit "$code"
}
trap finish EXIT
python -m sc4001_c.job_record --event start
python - <<'PY'
from pathlib import Path
from sc4001_c.common import load
r=Path('member-c/runs/M5-smoke')
assert (r/'COMPLETE').read_text().strip()=='M5 smoke complete'
v=load(r/'validation.json');assert v['global_step']==2 and v['merge_max_abs_logit_difference']<=1e-4
assert '6000 Ada' in v['gpu']
print('GPU_SMOKE_VERIFIED',v['gpu'])
PY
if [ "$stage" -lt 5 ]; then
  seeds=(13 21 42 87 100)
  python -u -m sc4001_c.lora_sweep train --seed "${seeds[$stage]}"
  if [ "$stage" -eq 4 ]; then python -m sc4001_c.lora_sweep freeze-selections; fi
  next=$((stage+1))
  job=$(sbatch --parsable --dependency="afterok:$SLURM_JOB_ID" member-c/scripts/ada/formal.sbatch "$next")
  printf '%s\t%s\t%s\n' "$next" "$job" submitted >> member-c/logs/M5-jobs.tsv
  echo "NEXT_STAGE=$next JOB=$job"
else
  python -u -m sc4001_c.lora_sweep evaluate
  python -m sc4001_c.lora_sweep collect
  echo 'M5_ALL_45_CANDIDATES_AND_15_SELECTED_TESTS_COMPLETE'
  job=$(sbatch --parsable --dependency="afterok:$SLURM_JOB_ID" member-c/scripts/ada/diagnostics.sbatch)
  printf '%s\t%s\t%s\n' diagnostics "$job" submitted >> member-c/logs/M5-jobs.tsv
  echo "DIAGNOSTICS_JOB=$job"
fi
printf '%s\t%s\t%s\n' "$stage" "$SLURM_JOB_ID" completed >> member-c/logs/M5-jobs.tsv
