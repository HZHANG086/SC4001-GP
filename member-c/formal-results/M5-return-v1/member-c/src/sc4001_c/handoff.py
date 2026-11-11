"""Preflight, complete-result validation, and small-file return archive."""
import argparse
import json
import math
import statistics
import tarfile
import subprocess
from pathlib import Path
from .common import load,dump,digest,read_csv,write_csv
from .collect_cka import collect,METHODS
from .benchmark import PROFILES

ROOT=Path('analysis/C-handoff-v2')


def preflight():
    # Metadata, existence and versions only: no GPU compute on the login node.
    import torch,transformers,datasets,numpy
    versions={'torch':torch.__version__.split('+')[0],'transformers':transformers.__version__,
              'datasets':datasets.__version__,'numpy':numpy.__version__}
    expected={'torch':'2.5.1','transformers':'4.51.3','datasets':'3.5.0','numpy':'1.26.4'}
    if versions!=expected: raise ValueError(f'Environment differs: {versions}; expected {expected}')
    bundle=Path('../member-c')
    delivered=load(bundle/'DELIVERY_MANIFEST.json')['files']
    for name,sha in delivered.items():
        if name.startswith(('src/','scripts/','protocol/')) and digest(bundle/name)!=sha:
            raise ValueError(f'Bundle file changed or partially copied: {name}')
    from .cka import verify_data
    verify_data(Path('artifacts/data-v1'),Path('../member-a/remote-records/data-v1'))
    rows=read_csv(Path('formal-results/selected_models.tsv'),'\t')
    from .common import BUDGETS,SEEDS
    grid={(r['method'],int(r['budget']),int(r['seed'])) for r in rows}
    if len(rows)!=60 or grid!={(m,b,s) for m in METHODS for b in BUDGETS for s in SEEDS}:
        raise ValueError('Selected model table does not contain the frozen 60 models')
    for r in rows:
        model=Path(r['model_dir']);cfg=model.parent/'run_config.json'
        for path in (model/'model.safetensors',model/'config.json',model/'tokenizer.json',cfg):
            if not path.is_file(): raise FileNotFoundError(path)
        sel=load(Path('formal-results/runs/B-v1')/f"selection-{r['method']}-b{r['budget']}-s{r['seed']}.json")
        if r['model_dir']!=sel['winner']+'/best': raise ValueError('Model selection mismatch')
        for domain in ('imdb','yelp','amazon'):
            path=Path('formal-results')/model.parent/'test'/f'{domain}_predictions.jsonl'
            if not path.is_file(): raise FileNotFoundError(path)
    from transformers.utils.hub import cached_file
    sources=load(Path('artifacts/data-v1/sources.json'))
    for name in ('config.json','model.safetensors','tokenizer.json'):
        cached_file(sources['model_id'],name,revision=sources['model_revision'],local_files_only=True)
    dump(ROOT/'preflight.json',{'status':'PASSED','versions':versions,'selected_models':len(rows),
        'note':'Existence and metadata only; full weight hashes and GPU replay checked in allocated jobs'})
    print('Preflight passed: 60 checkpoint paths, cache, prepared data and environment.')


def collect_cost(root,out):
    rows=[];reference=None;gpu=None
    for repeat in (1,2,3):
        for method,profile in PROFILES:
            folder=root/f'r{repeat}-{method}-{profile}'
            if (folder/'COMPLETE').read_text().strip()!='C-cost-v2 completed': raise ValueError('Incomplete cost probe')
            m=load(folder/'manifest.json');r=load(folder/'result.json')
            if m['smoke'] or r['fixture'] or (m['repeat'],m['method'],m['profile'])!=(repeat,method,profile):
                raise ValueError('Wrong cost scope')
            same={k:v for k,v in m.items() if k not in ('repeat','method','profile','lambda')}
            if reference is None: reference=same;gpu=r['gpu']
            if same!=reference or r['gpu']!=gpu: raise ValueError('Cost probes differ in settings/data/code/GPU model')
            if (m['batch'],m['length'],m['accumulation'],m['warmup_steps'],m['measured_steps'])!=(8,512,4,2,5):
                raise ValueError('Wrong microbenchmark protocol')
            if len(r['training_seconds_per_optimizer_step'])!=5 or r['optimizer_state_tensors']<=0:
                raise ValueError('No complete timed optimizer steps')
            if r['teacher_used_in_loss']!=(method=='M6') or r['teacher_present_training']!=(method=='M6' or profile=='diagnostic_teacher'):
                raise ValueError('Wrong teacher treatment')
            times=r['training_seconds_per_optimizer_step']
            if not all(math.isfinite(x) and x>0 for x in times): raise ValueError('Invalid timing')
            peak=max(r['training_peak']['allocated_bytes'],r['warmup_peak']['allocated_bytes'])
            if peak<=0 or not math.isfinite(peak): raise ValueError('Invalid CUDA memory')
            rows.append({'method':method,'profile':profile,'repeat':repeat,'gpu':gpu,
                'train_peak_allocated_bytes':peak,
                'train_peak_reserved_bytes':max(r['training_peak']['reserved_bytes'],r['warmup_peak']['reserved_bytes']),
                'seconds_per_optimizer_step_mean':statistics.mean(times),
                'train_examples_per_second':r['training_examples_per_second'],
                'inference_peak_allocated_bytes':r['inference_peak']['allocated_bytes'],
                'inference_examples_per_second':r['inference_examples_per_second'],
                'trainable_parameters':r['trainable_parameters'],'job_id':r['job_id'],
                'manifest_sha256':digest(folder/'manifest.json'),'result_sha256':digest(folder/'result.json')})
    write_csv(out/'cost_all_repeats.csv',rows)
    summary=[]
    for method,profile in PROFILES:
        group=[r for r in rows if (r['method'],r['profile'])==(method,profile)]
        row={'method':method,'profile':profile,'n_process_repeats':3,'gpu':gpu}
        for key in ('train_peak_allocated_bytes','seconds_per_optimizer_step_mean','train_examples_per_second',
                    'inference_peak_allocated_bytes','inference_examples_per_second'):
            vals=[r[key] for r in group];row[key+'_mean']=statistics.mean(vals);row[key+'_sd']=statistics.stdev(vals)
        summary.append(row)
    write_csv(out/'cost_summary.csv',summary)
    dump(out/'cost_collection.json',{'status':'COMPLETE','n_processes':len(rows),'shared_manifest':reference,
        'interpretation':'Hardware microbenchmark repeats, not independent trained models. M4 phase-2 proxy; M1 phase-1 proxy. No full-run extrapolation.'})


def package():
    collect(ROOT/'cka',ROOT/'collected')
    collect_cost(ROOT/'cost',ROOT/'collected')
    if (ROOT/'jobs.tsv').exists():
        ids=sorted({line.split('\t')[1].split(';')[0] for line in (ROOT/'jobs.tsv').read_text().splitlines() if line.strip()})
        try:
            result=subprocess.run(['sacct','-j',','.join(ids),'--format=JobID,State,Elapsed,AllocTRES,MaxRSS,ExitCode','-P'],
                                  capture_output=True,text=True,timeout=30)
            (ROOT/'slurm_accounting.txt').write_text(result.stdout+result.stderr)
        except (OSError,subprocess.TimeoutExpired) as error:
            (ROOT/'slurm_accounting.txt').write_text('Accounting unavailable: '+str(error))
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='RETURN_MANIFEST.json']
    files+=list(Path('logs').glob('c_bundle_*.out'))+list(Path('logs').glob('c_bundle_*.err'))
    # Job accounting is useful but not a replacement for measured compute time.
    dump(ROOT/'RETURN_MANIFEST.json',{'status':'ALL_RESULTS_VALIDATED','files':{str(p):digest(p) for p in sorted(files)},
         'note':'Final running job log may grow after this archive; include final logs separately when diagnosing errors.'})
    target=Path('analysis/C-return-v2.tar.gz')
    with tarfile.open(target,'w:gz') as archive:
        for p in sorted(files+[ROOT/'RETURN_MANIFEST.json']): archive.add(p,arcname=str(p),recursive=False)
    print(f'Validated and packaged: {target} ({target.stat().st_size} bytes)')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('preflight','package'));a=p.parse_args()
    if a.action=='preflight': preflight()
    else: package()


if __name__=='__main__': main()
