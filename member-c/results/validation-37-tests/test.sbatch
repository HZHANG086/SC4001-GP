#!/bin/bash
#SBATCH --partition=6000ada
#SBATCH --job-name=sc4001-gp-audit-test
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=audit-test-%j.out
#SBATCH --error=audit-test-%j.err
set -euo pipefail
trap 'code=$?; echo AUDIT_TEST_EXIT_CODE=$code; exit $code' EXIT
base=/share/zhangh/sc4001-gp-c/checks/ZIP-preflight-05b37e3/unpacked
mkdir member-a member-b
cp -R "$base/member-c" .
cp -R "$base/member-a/protocol" member-a/
cp -R "$base/member-b/protocol" "$base/member-b/tests" member-b/
tar -xzf C-audit-code-check.tar.gz
export PATH=/share/zhangh/sc4001-gp-c/env/bin:$PATH
export PYTHONPATH=member-c/src:member-a/protocol/locked_src:member-b/protocol/locked_src
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
python -m unittest discover -s member-c/tests -v
