"""Validate the five returned TC1 jobs before forming a combined CKA table."""
import argparse
import json
from pathlib import Path
from .common import load,dump,read_csv,write_csv,digest,BUDGETS,SEEDS,DOMAINS


def collect(root,out):
    rows=[];reference=None;inputs=[];seen=set()
    for seed in SEEDS:
        folder=root/f'seed-{seed}'
        if (folder/'COMPLETE').read_text().strip()!='C-CKA-v1 completed':
            raise ValueError(f'Incomplete or synthetic job: {folder}')
        manifest=load(folder/'manifest.json');run=load(folder/'run.json')
        if run['fixture'] or manifest['fixture'] or run['models_completed']!=6:
            raise ValueError('Wrong job scope')
        comparable={k:manifest[k] for k in ('version','samples','base_model','base_revision',
                                          'batch','max_length','script_sha256','common_sha256')}
        if reference is None: reference=comparable
        if comparable!=reference: raise ValueError('Different inputs/code/settings across jobs')
        current=read_csv(folder/'cka.csv')
        for r in current:
            key=(r['method'],r['reference'],int(r['budget']),int(r['seed']),r['domain'],r['pooling'],int(r['layer']))
            if key in seen: raise ValueError('Duplicate CKA row')
            if int(r['seed'])!=seed: raise ValueError('Wrong seed folder')
            seen.add(key)
        rows.extend(current)
        inputs.append({'folder':str(folder),'cka_sha256':digest(folder/'cka.csv'),'job_id':run['job_id']})
    expected={(m,ref,b,s,d,p,l) for m,ref in [('M3','pretrained'),('M6','pretrained'),('M6','M3')]
              for b in BUDGETS for s in SEEDS for d in DOMAINS for p in ('cls','mean') for l in range(13)}
    if seen!=expected: raise ValueError(f'Missing/extra CKA cells: {len(expected-seen)}/{len(seen-expected)}')
    write_csv(out/'cka_all_seeds.csv',rows)
    dump(out/'cka_collection.json',{'status':'COMPLETE','n_rows':len(rows),'inputs':inputs,
                                  'shared_manifest':reference,'interpretation':'No retraining or causal claim'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();collect(a.input,a.out)


if __name__=='__main__': main()
