#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=05:50:00
#SBATCH --job-name=sc4001-C-CKA
#SBATCH --output=logs/cka_c_%j.out
#SBATCH --error=logs/cka_c_%j.err
set -euo pipefail
# Submit FROM member-b/: sbatch ../member-c/scripts/cka.sbatch 13 smoke
# After checking the smoke result: sbatch ../member-c/scripts/cka.sbatch 13 full
# Submit other seeds as slots become available; respect the 2-job/account limit.
cd "${SLURM_SUBMIT_DIR:?}"
test -f formal-results/selected_models.tsv
export SC4001_CODE=formal
source scripts/activate.sh
export PYTHONPATH="$(cd ../member-c && pwd)/src:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1
seed="${1:?Supply one seed: 13, 21, 42, 87 or 100}"
mode="${2:-full}"
case "$seed" in 13|21|42|87|100) ;; *) echo "Invalid seed"; exit 2;; esac
case "$mode" in
  smoke) budgets=(400); output="analysis/cka-C-v1/smoke-s${seed}" ;;
  full) budgets=(400 1600 6400); output="analysis/cka-C-v1/seed-${seed}" ;;
  *) echo "Mode must be smoke or full"; exit 2 ;;
esac
python -u -m sc4001_c.cka \
  --models formal-results/selected_models.tsv \
  --model-base . --archive formal-results \
  --data artifacts/data-v1 --data-reference ../member-a/remote-records/data-v1 \
  --methods M3 M6 --seeds "$seed" --budgets "${budgets[@]}" \
  --samples 256 --batch 8 --out "$output"
