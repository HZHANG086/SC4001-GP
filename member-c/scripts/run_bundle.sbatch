#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=05:50:00
#SBATCH --job-name=sc4001-C-v2
#SBATCH --output=logs/c_bundle_%j.out
#SBATCH --error=logs/c_bundle_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export SC4001_CODE=formal
source scripts/activate.sh
export PYTHONPATH="$(cd ../member-c && pwd)/src:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
root=analysis/C-handoff-v2
stage="${1:?stage required}"
mkdir -p "$root"
trap 'echo "FAILED stage $stage job $SLURM_JOB_ID; preserve logs. No next job submitted." >&2' ERR
cka_cell() {
  python -u -m sc4001_c.cka --models formal-results/selected_models.tsv \
    --model-base . --archive formal-results --data artifacts/data-v1 \
    --data-reference ../member-a/remote-records/data-v1 \
    --methods M3 M4 M6 L2SP --seeds "$1" --budgets "$2" --samples 256 --batch 8 --out "$3"
}
if [ "$stage" = smoke ]; then
  cka_cell 13 400 "$root/smoke/cka"
  for method in M1 M2 M3 M4 M6 L2SP; do
    python -u -m sc4001_c.benchmark --method "$method" --repeat 1 --smoke --out "$root/smoke/$method"
  done
  echo 'C-handoff-v2 smoke completed' > "$root/SMOKE_COMPLETE"
  echo 'Smoke passed. Submit the full chain from the login node with start_bundle.sh full.'
  exit 0
fi
[[ "$stage" =~ ^[0-9]+$ ]] && [ "$stage" -le 17 ] || { echo 'Stage must be 0..17'; exit 2; }
if [ "$stage" -le 14 ]; then
  seeds=(13 21 42 87 100); budgets=(400 1600 6400)
  seed="${seeds[$((stage / 3))]}"; budget="${budgets[$((stage % 3))]}"
  cka_cell "$seed" "$budget" "$root/cka/b${budget}-s${seed}"
else
  repeat=$((stage - 14))
  # Rotate method order across repeats to reduce ordering / GPU warm-up confounding.
  methods=(M1 M2 M3 M4 M6 L2SP M3 M4 L2SP)
  profiles=(clean clean clean clean clean clean diagnostic_teacher diagnostic_teacher diagnostic_teacher)
  for offset in {0..8}; do
    idx=$(((offset + (repeat-1)*3) % 9))
    method="${methods[$idx]}"; profile="${profiles[$idx]}"
    python -u -m sc4001_c.benchmark --method "$method" --profile "$profile" --repeat "$repeat" \
      --out "$root/cost/r${repeat}-${method}-${profile}"
  done
fi
printf '%s\t%s\t%s\n' "$stage" "$SLURM_JOB_ID" completed >> "$root/jobs.tsv"
if [ "$stage" -lt 17 ]; then
  next=$((stage + 1))
  echo "$next" > "$root/NEXT_STAGE"
  # At most the current job and one successor are submitted by this chain.
  job=$(sbatch --parsable --dependency="afterok:$SLURM_JOB_ID" ../member-c/scripts/run_bundle.sbatch "$next")
  printf '%s\t%s\t%s\n' "$next" "$job" submitted >> "$root/jobs.tsv"
  echo "Next stage $next submitted as $job"
else
  python -m sc4001_c.handoff package
  echo 'All 60 CKA models and 27 cost probes validated. Return analysis/C-return-v2.tar.gz.'
fi
