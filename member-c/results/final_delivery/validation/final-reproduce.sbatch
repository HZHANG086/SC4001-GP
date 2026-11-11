#!/bin/bash
#SBATCH --partition=6000ada
#SBATCH --job-name=sc4001-final-zip
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=final-repro-%j.out
#SBATCH --error=final-repro-%j.err
set -euo pipefail
trap 'code=$?; echo FINAL_ARCHIVE_EXIT_CODE=$code; exit $code' EXIT
export PATH=/share/zhangh/sc4001-gp-c/env/bin:$PATH
python - <<'PY'
from pathlib import Path,PurePosixPath
import hashlib,zipfile,stat
archive=Path('SC4001-C-code.zip')
assert hashlib.sha256(archive.read_bytes()).hexdigest()==Path('SC4001-C-code.zip.sha256').read_text().split()[0]
with zipfile.ZipFile(archive) as package:
 for info in package.infolist():
  p=PurePosixPath(info.filename)
  assert not p.is_absolute() and '..' not in p.parts and not stat.S_ISLNK(info.external_attr>>16)
 assert not Path('unpacked').exists()
 package.extractall('unpacked')
PY
cd unpacked
python member-c/scripts/export_code.py --verify ../SC4001-C-code.zip
python member-c/scripts/reproduce_all.py --out ../outputs
