"""Validate M5 diagnostics and collect hardware-matched training probes."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .common import load,dump,digest


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('member-c/results'))
    a=p.parse_args(argv);out=a.results/'M5-diagnostics';out.mkdir(parents=True,exist_ok=True)
    if load(a.results/'M5-audit/audit.json')['status']!='PASS':raise ValueError('Prediction audit missing')
    folder=a.results/'M5-CKA'
    if (folder/'COMPLETE').read_text().strip()!='M5-CKA-v1 completed':raise ValueError('Incomplete M5 CKA')
    cka=pd.read_csv(folder/'cka.csv');replay=pd.read_csv(folder/'prediction_replay.csv')
    expected={(b,s,d,p,l) for b in (400,1600,6400) for s in (13,21,42,87,100)
        for d in ('imdb','yelp','amazon') for p in ('cls','mean') for l in range(13)}
    keys=set(cka[['budget','seed','domain','pooling','layer']].itertuples(index=False,name=None))
    if len(cka)!=1170 or keys!=expected or set(cka.method)!={'M5'} or set(cka.reference)!={'pretrained'} or set(cka.n_examples)!={256}:
        raise ValueError('Invalid M5 CKA grid')
    deg=(cka.layer==0)&(cka.pooling=='cls')
    similarity=['linear_cka','debiased_linear_cka','shuffled_linear_cka']
    if deg.sum()!=45 or not cka.loc[deg,similarity].isna().all().all() or not np.isfinite(cka.loc[~deg,similarity]).all().all():
        raise ValueError('Unexpected degenerate CKA entries')
    expected_replay={(b,s,d) for b,s,d,_,_ in expected}
    if len(replay)!=45 or set(replay[['budget','seed','domain']].itertuples(index=False,name=None))!=expected_replay or set(replay.n)!={256} or not np.isfinite(replay.max_probability_difference).all() or (replay.max_probability_difference>1e-4).any():
        raise ValueError('Invalid replay grid')
    rows=[];signatures=[];files={}
    for repeat in (1,2,3):
        for method in ('M1','M2','M3','M4','M5','M6','L2SP'):
            run=a.results/'Ada-cost'/method/'clean'/f'repeat-{repeat}'
            if not (run/'COMPLETE').exists():raise ValueError('Missing matched-hardware probe')
            cfg=load(run/'manifest.json');r=load(run/'result.json')
            if (cfg['method'],cfg['repeat'],cfg['profile'])!=(method,repeat,'clean'):raise ValueError('Cost identity mismatch')
            if '6000 Ada' not in r['gpu']:raise ValueError('Mixed cost hardware')
            shared={k:cfg[k] for k in ('base_model','base_revision','split_sha256','example_ids','text_sha256',
                'token_ids_sha256','length','batch','accumulation','precision','warmup_steps','measured_steps','learning_rate')}
            if signatures and shared!=signatures[0]:raise ValueError('Cost input/protocol mismatch')
            signatures.append(shared)
            seconds=r['training_seconds_per_optimizer_step']
            if len(seconds)!=5 or not np.isfinite(seconds).all() or min(seconds)<=0:raise ValueError('Bad measured timings')
            rows.append({'method':method,'repeat':repeat,'gpu':r['gpu'],
                'peak_allocated_gib':r['training_peak']['allocated_bytes']/2**30,
                'seconds_per_step':np.mean(seconds),'trainable_parameters':r['trainable_parameters']})
            for name in ('manifest.json','result.json','COMPLETE'):files[str(run/name)]=digest(run/name)
    costs=pd.DataFrame(rows);costs.to_csv(out/'ada_cost_repeats.csv',index=False)
    summary=costs.groupby('method').agg(peak_allocated_gib_mean=('peak_allocated_gib','mean'),
        peak_allocated_gib_sd=('peak_allocated_gib','std'),seconds_per_step_mean=('seconds_per_step','mean'),
        seconds_per_step_sd=('seconds_per_step','std'),n_process_repeats=('repeat','count'))
    summary.to_csv(out/'ada_cost_summary.csv')
    dump(out/'audit.json',{'status':'PASS','M5_cka_rows':1170,'M5_expected_degenerate_rows':45,'M5_replays':45,
        'cost_processes':21,'files':files,'shared_cost_inputs':signatures[0],
        'limit':'Training microbenchmarks only; M4 stage-2 proxy; no trained-winner deployment speed claim. Cost process repeats are not training seeds.'})
    print('M5_DIAGNOSTICS_COLLECTED')


if __name__=='__main__':main()
