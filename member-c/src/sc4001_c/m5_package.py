"""Create a small, checksummed return archive; keep all large weights on the server."""
import argparse
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path
from .common import load,dump,digest


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('/share/zhangh/sc4001-gp-c/returns'))
    a=p.parse_args(argv)
    if load('member-c/results/M5-diagnostics/audit.json')['status']!='PASS':raise ValueError('Complete diagnostics first')
    a.out.mkdir(parents=True,exist_ok=True)
    paths=[Path('member-c/runs/M5-v1'),Path('member-c/logs'),Path('member-c/protocol'),
        Path('member-c/src'),Path('member-c/scripts/ada')]
    paths += [Path('member-c/results')/name for name in ('M5-v1','M5-audit','M5-CKA','Ada-cost','M5-diagnostics')]
    files=[]
    for root in paths:
        for f in root.rglob('*'):
            if not f.is_file() or f.is_symlink():continue
            if any(x in ('best','checkpoints','__pycache__') or x.startswith('checkpoint-') for x in f.parts):continue
            if f.suffix in ('.pt','.bin','.safetensors','.pyc'):continue
            files.append(f)
    jobs={'12366','12367'}
    ledger=Path('member-c/logs/M5-jobs.tsv')
    if ledger.exists():
        jobs.update(line.split('\t')[1].split(';')[0] for line in ledger.read_text().splitlines() if line.strip())
    if os.getenv('SLURM_JOB_ID'):jobs.add(os.environ['SLURM_JOB_ID'])
    accounting=subprocess.check_output(['sacct','-j',','.join(sorted(jobs)),
        '--format=JobID,JobName,State,ExitCode,Elapsed,Start,End,AllocTRES','-P'],text=True)
    ledger=Path('member-c/results/M5-diagnostics/slurm-at-package.tsv');ledger.write_text(accounting)
    files.append(ledger);files=sorted(set(files))
    manifest=Path('member-c/results/M5-diagnostics/RETURN_MANIFEST.json')
    files=[f for f in files if f!=manifest]
    dump(manifest,{'version':'M5-return-v1','created_unix':time.time(),
        'status':'EXPERIMENT_AND_DIAGNOSTIC_RESULTS_COMPLETE; final local integration pending',
        'slurm_note':'Packaging occurs inside diagnostics job; its final exit state must be collected after this job ends.',
        'files':{str(f):digest(f) for f in files},'excluded':'Large weights and recovery tensors remain on server.'})
    target=a.out/'M5-return-v1.tar.gz';temporary=a.out/'M5-return-v1.partial.tar.gz'
    with tarfile.open(temporary,'w:gz') as archive:
        for f in files+[manifest]:archive.add(f,arcname=str(f),recursive=False)
    temporary.replace(target)
    (a.out/'M5-return-v1.sha256').write_text(digest(target)+'  '+target.name+'\n')
    print('RETURN_ARCHIVE',target,'BYTES',target.stat().st_size)


if __name__=='__main__':main()
