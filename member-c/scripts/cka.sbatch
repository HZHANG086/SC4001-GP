#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
echo 'Old C-v1 entry retired. From member-b run: bash ../member-c/scripts/start_bundle.sh smoke' >&2
exit 2
