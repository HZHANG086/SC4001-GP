#!/bin/bash
#SBATCH --partition=6000ada
#SBATCH --job-name=sc4001-gp-zip
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=zip-preflight-%j.out
#SBATCH --error=zip-preflight-%j.err
set -euo pipefail
trap 'code=$?; echo ZIP_PREFLIGHT_EXIT_CODE=$code; exit $code' EXIT
export PATH=/share/zhangh/sc4001-gp-c/env/bin:$PATH
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python - <<'CHECK'
from pathlib import Path
import zipfile,json,hashlib
archive=Path('SC4001-code-PREFLIGHT.zip')
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='005d29c467fc489851cedb1505c1577833b9f8ebb2ee0f974f0c55fee118fb18'
root=Path('unpacked');root.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
 manifest=json.loads(z.read('PACKAGE_MANIFEST.json'))
 z.extractall(root)
for name,row in manifest['files'].items():
 p=root/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],name
assert not (root/'.git').exists()
print('ZIP_TRANSPORT_AND_EXTRACTION_VERIFIED',len(manifest['files']),flush=True)
CHECK
cd unpacked
export PYTHONPATH=member-c/src
python -m sc4001_c.reproduce --out reproduced/C-v1
python - <<'RECORD'
from pathlib import Path
import sys,json,platform,numpy,pandas,scipy,os
record={'python':platform.python_version(),'numpy':numpy.__version__,'pandas':pandas.__version__,'scipy':scipy.__version__,'slurm_job_id':os.environ['SLURM_JOB_ID'],'gpu_requested':False,'audit':json.loads(Path('reproduced/C-v1/reproduction_audit.json').read_text())}
Path('../linux_reproduction.json').write_text(json.dumps(record,indent=2)+'\n')
print('LINUX_ZIP_REPRODUCTION_PASS',flush=True)
RECORD
