#!/bin/bash
#SBATCH --partition=6000ada
#SBATCH --job-name=sc4001-gp-raw-audit
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=raw-audit-%j.out
#SBATCH --error=raw-audit-%j.err
set -euo pipefail
trap 'code=$?; echo RAW_AUDIT_EXIT_CODE=$code; exit $code' EXIT
base=/share/zhangh/sc4001-gp-c/checks/ZIP-preflight-05b37e3/unpacked
mkdir member-a member-b
cp -R "$base/member-c" .
cp -R "$base/member-a/protocol" member-a/
cp -R "$base/member-b/protocol" "$base/member-b/tests" member-b/
tar -xzf C-raw-audit-code.tar.gz
export PATH=/share/zhangh/sc4001-gp-c/env/bin:$PATH
export PYTHONPATH=member-c/src:member-a/protocol/locked_src:member-b/protocol/locked_src
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
python - <<'VERIFY'
from pathlib import Path
import hashlib,json
for name,sha in json.loads(Path('source-hashes.json').read_text()).items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
print('EXACT_SOURCE_HASHES_VERIFIED')
VERIFY
python -m unittest discover -s member-c/tests -v
