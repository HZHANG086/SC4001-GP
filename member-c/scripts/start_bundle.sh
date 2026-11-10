#!/bin/bash
# Run from member-b on the login node. No GPU computation happens here.
set -euo pipefail
mode="${1:-}"
case "$mode" in smoke|full|resume|package) ;; *) echo 'Usage: bash ../member-c/scripts/start_bundle.sh smoke|full|resume STAGE|package'; exit 2;; esac
test -f formal-results/selected_models.tsv
mkdir -p logs analysis/C-handoff-v2
# Refuse double launch / exceeding the account quota. Check before environment setup.
if [ "$mode" != package ]; then
  active=$(squeue -h -u "$(id -un)" -o '%i')
  if [ -n "$active" ]; then echo 'Please wait until your current submitted jobs finish; avoids duplicate chains and quota errors.'; exit 2; fi
fi
export SC4001_CODE=formal
source scripts/activate.sh
export PYTHONPATH="$(cd ../member-c && pwd)/src:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [ "$mode" = package ]; then python -m sc4001_c.handoff package; exit 0; fi
python -m sc4001_c.handoff preflight
if [ "$mode" = smoke ]; then stage=smoke
else
  test "$(cat analysis/C-handoff-v2/SMOKE_COMPLETE)" = 'C-handoff-v2 smoke completed'
  stage=0
  if [ "$mode" = resume ]; then stage="${2:?Supply failed/unsubmitted stage 0..17}"; fi
  [[ "$stage" =~ ^[0-9]+$ ]] && [ "$stage" -le 17 ] || { echo 'Invalid resume stage'; exit 2; }
fi
job=$(sbatch --parsable ../member-c/scripts/run_bundle.sbatch "$stage")
printf '%s\t%s\t%s\n' "$stage" "$job" submitted >> analysis/C-handoff-v2/jobs.tsv
echo "Submitted stage $stage as $job. The full chain submits one successor only after success."
