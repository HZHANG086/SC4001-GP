"""Export tracked project evidence as a portable, checksummed code ZIP.

This exports a versioned snapshot, not a scientific completion certificate.
Run from a clean Git checkout. An extracted ZIP needs no Git to reproduce the
included predictions/statistics; see member-c/docs/REPRODUCTION.md.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import zipfile


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(archive):
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive names')
        for entry in package.infolist():
            path = PurePosixPath(entry.filename)
            if (path.is_absolute() or '..' in path.parts or str(path) != entry.filename
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError('Unsafe archive member: ' + entry.filename)
        manifest = json.loads(package.read('PACKAGE_MANIFEST.json'))
        if set(names) != set(manifest['files']) | {'PACKAGE_MANIFEST.json'}:
            raise ValueError('Missing or unmanifested package files')
        for name, record in manifest['files'].items():
            h = hashlib.sha256()
            with package.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
            if (h.hexdigest() != record['sha256']
                    or package.getinfo(name).file_size != record['bytes']):
                raise ValueError('Package contents differ: ' + name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out', type=Path)
    group.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        manifest = verify(args.verify)
        print('PACKAGE_CHECKSUMS_PASS', len(manifest['files']), 'files')
        return
    repo = Path(subprocess.check_output(
        ['git', 'rev-parse', '--show-toplevel'], text=True).strip())
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain']).strip():
        raise ValueError('Commit reviewed changes first; export requires a clean checkout')
    target = args.out.resolve()
    if target.exists() or target.with_suffix(target.suffix + '.sha256').exists():
        raise ValueError('Preserve existing deliveries; choose a new output name')
    relative = subprocess.check_output(
        ['git', '-C', str(repo), 'ls-files', '-z']).decode().split('\0')
    relative = sorted(name for name in relative if name)
    excluded_parts = {'.git', '__pycache__', '.venv', 'fonts', 'checkpoints', 'best'}
    excluded_suffixes = {'.pyc', '.safetensors', '.pt', '.pth', '.bin', '.ttf', '.otf'}
    records = {}
    for name in relative:
        path = repo / name
        if (path.is_symlink() or not path.is_file()
                or excluded_parts.intersection(Path(name).parts)
                or path.suffix.lower() in excluded_suffixes
                or path.name == '.env'):
            raise ValueError('Unexpected tracked payload; review before export: ' + name)
        records[name] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    required = {'member-c/CURRENT.md', 'member-c/docs/REPRODUCTION.md',
                'member-c/requirements-final.txt', 'member-c/src/sc4001_c/reproduce.py',
                'member-c/DELIVERY_MANIFEST.json', 'member-a/protocol/A-v1.json',
                'member-b/protocol/B-v1.json'}
    if not required <= records.keys():
        raise ValueError('Required offline reproduction files missing')
    manifest = {
        'format': 'SC4001-code-evidence-snapshot-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'commit': subprocess.check_output(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
        'status': 'Versioned snapshot; consult member-c/CURRENT.md for actual completion status.',
        'read_first': 'member-c/docs/REPRODUCTION.md',
        'scope': 'All tracked code, reports and small experiment evidence; no Git history, model weights, caches or licensed font files.',
        'files': records,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.partial')
    if temporary.exists():
        raise ValueError('Preserve previous partial archive for inspection')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for name in relative:
            package.write(repo / name, name)
        package.writestr('PACKAGE_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    verify(temporary)
    temporary.replace(target)
    target.with_suffix(target.suffix + '.sha256').write_text(sha256(target) + '  ' + target.name + '\n')
    print('PACKAGE_EXPORTED_AND_VERIFIED', target, len(records), 'files', target.stat().st_size, 'bytes')


if __name__ == '__main__':
    main()
