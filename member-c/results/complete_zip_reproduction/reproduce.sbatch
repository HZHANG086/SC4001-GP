#!/bin/bash
#SBATCH --partition=6000ada
#SBATCH --job-name=sc4001-c-final-repro
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=complete-repro-%j.out
#SBATCH --error=complete-repro-%j.err
set -euo pipefail
trap 'code=$?; echo REPRO_EXIT_CODE=$code; exit $code' EXIT
export PATH=/share/zhangh/sc4001-gp-c/env/bin:$PATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
python - <<'PY'
from pathlib import Path,PurePosixPath
import hashlib,zipfile
p=Path('SC4001-C-code-v1.zip');assert hashlib.sha256(p.read_bytes()).hexdigest()==Path('SC4001-C-code-v1.zip.sha256').read_text().split()[0]
with zipfile.ZipFile(p) as z:
 for name in z.namelist():
  p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts
 assert not Path('unpacked').exists();z.extractall('unpacked')
PY
cd unpacked
export PYTHONPATH=member-c/src:member-a/protocol/locked_src:member-b/protocol/locked_src
export MPLCONFIGDIR=../matplotlib-cache
python member-c/scripts/export_code.py --verify ../SC4001-C-code-v1.zip
python -m unittest discover -s member-c/tests -v > ../tests.out 2> ../tests.err
python -m sc4001_c.reproduce --out reproduced/C-v1
python -m sc4001_c.m5_import --raw member-c/formal-results/M5-return-v1 --out reproduced/M5-local-audit
python -m sc4001_c.final_analysis --m5 member-c/formal-results/M5-return-v1/member-c/results/M5-v1/per_seed.csv --m5-audit reproduced/M5-local-audit/local_audit.json --out reproduced/C-final
python -m sc4001_c.final_plots --data reproduced/C-final --out reproduced/figures
python -m sc4001_c.final_plots --data reproduced/C-final --grayscale --out reproduced/figures-grayscale
python ../verify_outputs.py
