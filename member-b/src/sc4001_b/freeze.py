"""Freeze protocol B-v1: hash the B source files and copy them to protocol/locked_src.

    python -m sc4001_b.freeze --protocol protocol/B-v1.json

Run once, after the source-only pilot fixed the grids and before any formal run. Formal
jobs import protocol/locked_src, and sweep.py refuses to run if a hash differs.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

FROZEN = ('__init__.py', 'aggregate.py', 'objectives.py', 'sweep.py', 'train.py')


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', default='protocol/B-v1.json')
    p.add_argument('--src', default='src/sc4001_b')
    a = p.parse_args(argv)
    path, src = Path(a.protocol), Path(a.src)
    spec = json.loads(path.read_text())
    if spec.get('frozen_on'):
        raise SystemExit(f'{path} is already frozen ({spec["frozen_on"]}); create a new version instead')
    locked = path.parent / 'locked_src' / 'sc4001_b'
    if locked.exists():
        raise SystemExit(f'{locked} exists; refusing to overwrite')
    locked.mkdir(parents=True)
    spec['code_hashes'] = {}
    for name in FROZEN:
        shutil.copy2(src / name, locked / name)
        spec['code_hashes'][name] = hashlib.sha256((locked / name).read_bytes()).hexdigest()
    spec['frozen_on'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    path.write_text(json.dumps(spec, indent=2) + '\n')
    print(f'Froze {len(FROZEN)} files into {locked}')


if __name__ == '__main__':
    main()
