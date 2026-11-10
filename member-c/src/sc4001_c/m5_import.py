"""Verify a small M5 return archive, then independently audit its local predictions.

Large selected weights stay on Ada. Their server checks/replays are retained but
are not mislabeled as independent local weight verification.
"""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import tarfile
import numpy as np
import pandas as pd
from .common import load,dump,digest
from .m5_audit import audit_predictions
from .lora_sweep import validate_run,run_name

MANIFEST='member-c/results/M5-diagnostics/RETURN_MANIFEST.json'


def verified_extract(archive_path,destination):
    if destination.exists():raise ValueError('Preserve the existing import; choose a new destination')
    with tarfile.open(archive_path,'r:gz') as archive:
        members=archive.getmembers();names=[m.name for m in members]
        if len(set(names))!=len(names):raise ValueError('Duplicate archive paths')
        for m in members:
            p=PurePosixPath(m.name)
            if not m.isfile() or p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!='member-c':
                raise ValueError('Unsafe or unexpected archive member')
        if MANIFEST not in names:raise ValueError('Return manifest missing')
        manifest=json.load(archive.extractfile(MANIFEST))
        if set(names)!=set(manifest['files'])|{MANIFEST}:raise ValueError('Unmanifested or missing files')
        for name,sha in manifest['files'].items():
            h=hashlib.sha256()
            with archive.extractfile(name) as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=sha:raise ValueError('Returned file checksum mismatch: '+name)
        destination.mkdir(parents=True,exist_ok=False)
        for m in members:
            target=destination/m.name;target.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(m) as source,target.open('wb') as sink:shutil.copyfileobj(source,sink)
    return manifest


def audit_local(raw,repo,out):
    member=raw/'member-c';root=member/'runs/M5-v1'
    spec=load(repo/'member-c/protocol/M5-v1.json')
    if digest(member/'protocol/M5-v1.json')!=digest(repo/'member-c/protocol/M5-v1.json'):
        raise ValueError('Training protocol differs from frozen local protocol')
    for name,sha in spec['code_hashes'].items():
        if digest(member/'src/sc4001_c'/name)!=sha:raise ValueError('Returned training code differs')
    m5cka=load(member/'results/M5-CKA/manifest.json')
    reference_cka=load(repo/'member-c/formal-results/return-v2/analysis/C-handoff-v2/cka/b400-s13/manifest.json')
    if m5cka['samples']!=reference_cka['samples']:
        raise ValueError('M5 and B CKA sample IDs/labels/text hashes differ')
    if m5cka['estimator_sha256']!=reference_cka['script_sha256']:
        raise ValueError('M5 and B use different CKA estimators')
    lock=load(root/'ALL_SELECTIONS_LOCKED.json')
    if lock['protocol_sha256']!=digest(member/'protocol/M5-v1.json'):raise ValueError('Selection protocol changed')
    expected={(b,s) for b in spec['budgets'] for s in spec['seeds']}
    if len(lock['selections'])!=15 or {(x['budget'],x['seed']) for x in lock['selections']}!=expected:
        raise ValueError('Wrong selected grid')
    refs={}
    for d in ('imdb','yelp','amazon'):
        files=list((repo/'member-b/formal-results/runs/B-v1').glob(f'M3-b400-s13-*/test/{d}_predictions.jsonl'))
        if len(files)!=1:raise ValueError('Ambiguous reference test IDs')
        with files[0].open() as f:refs[d]={x['example_id']:x['label'] for x in (json.loads(line) for line in f)}
    per_seed=pd.read_csv(member/'results/M5-v1/per_seed.csv').set_index(['budget','seed'])
    if len(per_seed)!=15 or set(per_seed.index)!=expected:raise ValueError('Wrong per-seed table grid')
    checks=[];metrics=[]
    for sel in lock['selections']:
        b,s=sel['budget'],sel['seed']
        candidates=[root/run_name(b,s,lr) for lr in spec['learning_rates']]
        vals=[validate_run(p,spec,b,s,lr) for p,lr in zip(candidates,spec['learning_rates'])]
        winner=max(range(len(vals)),key=lambda i:vals[i]['eval_accuracy'])
        if Path(sel['winner']).name!=candidates[winner].name:raise ValueError('Not source-validation winner')
        if sel['validation_sha256']!=[digest(p/'validation.json') for p in candidates]:raise ValueError('Selection evidence changed')
        for p in candidates:
            cfg=load(p/'run_config.json')
            if cfg['protocol_sha256']!=digest(member/'protocol/M5-v1.json'):raise ValueError('Mixed candidate protocols')
            if cfg['split_sha256']!=digest(repo/f'member-a/remote-records/data-v1/splits/b{b}_s{s}.json'):
                raise ValueError('Source split mismatch')
        run=candidates[winner];recorded=load(run/'test/metrics.json');observed=[]
        for d,reference in refs.items():
            path=run/'test'/f'{d}_predictions.jsonl';m,n=audit_predictions(path,reference)
            if n!=recorded[d]['n'] or abs(m['accuracy']-recorded[d]['accuracy'])>1e-12:
                raise ValueError('Recorded test metrics do not reproduce')
            if abs(m['accuracy']-per_seed.loc[(b,s),d+'_accuracy'])>1e-12:raise ValueError('Per-seed table differs')
            if d!='imdb':observed.append(m['accuracy'])
            metrics.append({'method':'M5','budget':b,'seed':s,'domain':d,'n':n,**m})
            checks.append({'path':str(path.relative_to(raw)),'sha256':digest(path),'n':n})
        if abs(np.mean(observed)-per_seed.loc[(b,s),'ood_accuracy'])>1e-12:raise ValueError('OOD average differs')
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(metrics).to_csv(out/'prediction_metrics_recomputed.csv',index=False)
    dump(out/'local_audit.json',{'status':'PASS','candidate_records':45,'selected_models':15,
        'prediction_files':45,'prediction_rows':sum(x['n'] for x in checks),'checks':checks,
        'CKA_inputs':'All 768 sample IDs, labels and text hashes match B, with the same frozen estimator source.',
        'weight_boundary':'Weights not downloaded. Retained remote weight hashes and 45 subset probability replays; not independent local weight hashing.',
        'target_selection':'Source choices independently recomputed from all candidate validation records; target predictions only read after checking each frozen choice.'})


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);p.add_argument('--repo',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path,default=Path('member-c/results/M5-local-audit'))
    a=p.parse_args(argv);manifest=verified_extract(a.archive,a.destination)
    audit_local(a.destination,a.repo,a.out)
    dump(a.out/'import.json',{'archive_sha256':digest(a.archive),'manifest_files':len(manifest['files']),
        'destination':str(a.destination),'status':'CHECKSUM_AND_LOCAL_PREDICTION_AUDIT_PASSED'})
    print('M5_RETURN_IMPORTED_AND_AUDITED')


if __name__=='__main__':main()
