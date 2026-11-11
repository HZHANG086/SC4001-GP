"""Validate all 60 checkpoint analyses and archived-probability replay."""
import argparse
import math
from pathlib import Path
from .common import load,dump,read_csv,write_csv,digest,BUDGETS,SEEDS,DOMAINS

METHODS=('M3','M4','M6','L2SP')
COMPARISONS=tuple((m,'pretrained') for m in METHODS)+tuple((m,'M3') for m in METHODS if m!='M3')


def collect(root,out):
    rows=[];replays=[];reference=None;inputs=[];seen=set()
    for seed in SEEDS:
        for budget in BUDGETS:
            folder=root/f'b{budget}-s{seed}'
            if (folder/'COMPLETE').read_text().strip()!='C-CKA-v2 completed':
                raise ValueError(f'Incomplete or synthetic job: {folder}')
            manifest=load(folder/'manifest.json');run=load(folder/'run.json')
            if run['fixture'] or manifest['fixture'] or run['models_completed']!=4:
                raise ValueError('Wrong job scope')
            if {(r['method'],int(r['budget']),int(r['seed'])) for r in manifest['models']}!={(m,budget,seed) for m in METHODS}:
                raise ValueError('Wrong checkpoint manifest')
            comparable={k:manifest[k] for k in ('version','samples','base_model','base_revision',
                'batch','max_length','script_sha256','common_sha256','model_code_sha256','versions')}
            if reference is None: reference=comparable
            if comparable!=reference: raise ValueError('Different inputs/code/settings across jobs')
            current=read_csv(folder/'cka.csv')
            for r in current:
                key=(r['method'],r['reference'],int(r['budget']),int(r['seed']),r['domain'],r['pooling'],int(r['layer']))
                if key in seen: raise ValueError('Duplicate CKA row')
                if int(r['seed'])!=seed or int(r['budget'])!=budget or int(r['n_examples'])!=256:
                    raise ValueError('Wrong cell or sample size')
                if r['status']=='ok':
                    for metric in ('linear_cka','shuffled_linear_cka'):
                        value=float(r[metric])
                        if not math.isfinite(value) or not -1e-8<=value<=1+1e-8:
                            raise ValueError('Invalid CKA value')
                    if not math.isfinite(float(r['debiased_linear_cka'])):
                        raise ValueError('Invalid debiased CKA')
                elif r['status']!='degenerate_zero_variance' or r['linear_cka']!='':
                    raise ValueError('Invalid CKA status')
                seen.add(key)
            replay=read_csv(folder/'prediction_replay.csv')
            expected_replay={(m,budget,seed,d) for m in METHODS for d in DOMAINS}
            if len(replay)!=12 or {(r['method'],int(r['budget']),int(r['seed']),r['domain']) for r in replay}!=expected_replay:
                raise ValueError('Missing replay checks')
            for r in replay:
                delta=float(r['max_probability_difference'])
                if not 0<=delta<=1e-4 or int(r['n'])!=256: raise ValueError('Replay failed')
            rows.extend(current);replays.extend(replay)
            inputs.append({'folder':str(folder),'cka_sha256':digest(folder/'cka.csv'),
                           'replay_sha256':digest(folder/'prediction_replay.csv'),'job_id':run['job_id'],'gpu':run['gpu']})
    expected={(m,ref,b,s,d,p,l) for m,ref in COMPARISONS for b in BUDGETS for s in SEEDS
              for d in DOMAINS for p in ('cls','mean') for l in range(13)}
    if seen!=expected: raise ValueError(f'Missing/extra CKA cells: {len(expected-seen)}/{len(seen-expected)}')
    write_csv(out/'cka_all_seeds.csv',rows);write_csv(out/'prediction_replay_all.csv',replays)
    dump(out/'cka_collection.json',{'status':'COMPLETE','n_rows':len(rows),'inputs':inputs,
                                  'shared_manifest':reference,'interpretation':'Descriptive; no retraining or causal claim'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();collect(a.input,a.out)


if __name__=='__main__': main()
