#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=05:50:00
#SBATCH --job-name=sc4001-B-formal
#SBATCH --output=logs/formal_%j.out
#SBATCH --error=logs/formal_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
# Usage: sbatch scripts/formal.sbatch <M3|M4|M6|L2SP> <seed> [<seed> ...]
# Frozen code only (protocol/locked_src). Re-submitting the same command resumes and skips
# completed candidates, so a TIMEOUT loses at most the epoch in progress.
export SC4001_CODE=formal
source scripts/activate.sh
phase="$1"; shift
python -m sc4001_b.sweep --phase "$phase" --seeds "$@"
