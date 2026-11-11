"""M5 bounded sweep: train all cells, freeze all selections, then evaluate once.

No target metrics are read during candidate choice. Nonwinner and recovery weights
are pruned only within newly created M5 runs, after successful merged export.
"""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
from .common import load,dump,digest,write_csv
from .cka import verify_data


def run_name(budget,seed,lr):return f'M5-b{budget}-s{seed}-lr{lr:.0e}'


def validate_run(folder,spec,budget,seed,lr):
    if (folder/'COMPLETE').read_text().strip()!='M5 source-validation training complete; no tests evaluated':
        raise ValueError('Incomplete, smoke or fixture run')
    cfg=load(folder/'run_config.json');val=load(folder/'validation.json');history=load(folder/'history.json')
    expected={'method':'M5','budget':budget,'seed':seed,'lr':lr,'epochs':10,'length':512,
              'batch':8,'accumulation':4,'stage':'formal','smoke':False,'fixture':False,'lora':spec['lora']}
    if any(cfg.get(k)!=v for k,v in expected.items()):raise ValueError(f'Invalid completed config: {folder}')
    epochs=[h for h in history if 'eval_accuracy' in h]
    if max(h['epoch'] for h in epochs)<10:raise ValueError('Ten epochs not completed')
    best=max(h['eval_accuracy'] for h in epochs)
    if abs(best-val['eval_accuracy'])>1e-12:raise ValueError('Not validation-best')
    if load(folder/'parameters.json')['trainable']!=296450:raise ValueError('Wrong trainable parameters')
    return val


def select_cell(root,budget,seed,spec):
    candidates=[root/run_name(budget,seed,lr) for lr in spec['learning_rates']]
    vals=[validate_run(p,spec,budget,seed,lr) for p,lr in zip(candidates,spec['learning_rates'])]
    index=max(range(len(candidates)),key=lambda i:vals[i]['eval_accuracy']) # first on tie
    result={'method':'M5','budget':budget,'seed':seed,'lr':spec['learning_rates'][index],
        'winner':str(candidates[index]),'candidate_runs':[str(p) for p in candidates],
        'source_val_accuracy':vals[index]['eval_accuracy'],
        'validation_sha256':[digest(p/'validation.json') for p in candidates],
        'rule':'IMDb validation only; fixed LR-list tie order; no test metrics consulted'}
    target=root/f'selection-M5-b{budget}-s{seed}.json'
    if target.exists() and load(target)!=result:raise ValueError('Previously frozen selection changed')
    dump(target,result)
    return result


def prune(path,root):
    if path.exists():
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe cleanup path')
        shutil.rmtree(path)


def ensure_candidate(root,b,s,lr,args,spec):
    folder=root/run_name(b,s,lr)
    if (folder/'COMPLETE').exists():
        validate_run(folder,spec,b,s,lr);return
    cmd=[sys.executable,'-m','sc4001_c.lora_train','--data',str(args.data),'--out',str(folder),
        '--protocol',str(args.protocol),'--data-reference',str(args.data_reference),
        '--budget',str(b),'--seed',str(s),'--lr',str(lr)]
    if folder.exists():
        complete=[]
        for ck in (folder/'checkpoints').glob('checkpoint-*'):
            required=('trainer_state.json','optimizer.pt','scheduler.pt','rng_state.pth','model.safetensors')
            if all((ck/f).is_file() and (ck/f).stat().st_size>0 for f in required):complete.append(ck)
            else:ck.rename(ck.with_name('incomplete-'+ck.name+'-'+str(time.time_ns())))
        if complete:cmd+=['--resume']
        else:raise ValueError(f'No resumable checkpoint; preserve and inspect {folder} before a fresh attempt')
    log=args.logs/f'{folder.name}.log';args.logs.mkdir(parents=True,exist_ok=True)
    event={'run':str(folder),'job_id':os.getenv('SLURM_JOB_ID'),'command':cmd,'started_unix':time.time()}
    with (root/'attempts.jsonl').open('a') as f:f.write(json.dumps({**event,'event':'start'})+'\n')
    start=time.perf_counter()
    with log.open('a') as stream:status=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT).returncode
    with (root/'attempts.jsonl').open('a') as f:f.write(json.dumps({**event,'event':'finish','returncode':status,'wall_seconds':time.perf_counter()-start})+'\n')
    if status:raise RuntimeError(f'Training failed; inspect {log}')
    validate_run(folder,spec,b,s,lr)
    if digest(folder/'best/model.safetensors')!=load(folder/'validation.json')['checkpoint_weight_sha256']:
        raise ValueError('Merged model checksum mismatch')
    prune(folder/'checkpoints',root)


def freeze_all(root,spec,protocol):
    selections=[select_cell(root,b,s,spec) for s in spec['seeds'] for b in spec['budgets']]
    record={'protocol_sha256':digest(protocol),'selections':selections,
            'rule':'All 45 candidate training runs completed before any M5 test evaluation'}
    path=root/'ALL_SELECTIONS_LOCKED.json'
    if path.exists() and load(path)!=record:raise ValueError('Global lock changed')
    dump(path,record)
    for sel in selections:
        for candidate in sel['candidate_runs']:
            if candidate!=sel['winner']:
                prune(Path(candidate)/'best',root)
                (Path(candidate)/'WEIGHTS_PRUNED').write_text('Nonwinner removed after global source-only selection; never test-evaluated.\n')
    return record


def evaluate(root,args,spec):
    import torch
    if not torch.cuda.is_available():raise ValueError('Allocated GPU required')
    # Recalculate all choices from source records and require the same pre-test lock.
    path=root/'ALL_SELECTIONS_LOCKED.json'
    if not path.exists():raise ValueError('Complete all training and freeze global selection first')
    before=digest(path);record=freeze_all(root,spec,args.protocol)
    if digest(path)!=before:raise ValueError('Selection changed before evaluation')
    for sel in record['selections']:
        if args.seed and sel['seed'] not in args.seed:continue
        run=Path(sel['winner']);test=run/'test'
        if (test/'metrics.json').exists():continue
        if test.exists():test.rename(run/f'test-incomplete-{time.time_ns()}')
        print('EVALUATE_FROZEN_WINNER',run,flush=True)
        with (args.logs/f'eval-{run.name}.log').open('a') as stream:
            subprocess.run([sys.executable,'-m','sc4001_a.evaluate','--run',str(run),'--data',str(args.data),'--protocol-locked'],
                           stdout=stream,stderr=subprocess.STDOUT,check=True)


def collect(root,args,spec):
    record=load(root/'ALL_SELECTIONS_LOCKED.json');rows=[]
    for sel in record['selections']:
        run=Path(sel['winner']);cfg=load(run/'run_config.json');val=load(run/'validation.json');m=load(run/'test/metrics.json')
        rows.append({'method':'M5','budget':cfg['budget'],'seed':cfg['seed'],'lr':cfg['lr'],'run':str(run),
            'source_val_accuracy':sel['source_val_accuracy'],
            **{d+'_accuracy':m[d]['accuracy'] for d in ('imdb','yelp','amazon')},'ood_accuracy':m['ood_mean_accuracy'],
            'trainable_parameters':load(run/'parameters.json')['trainable'],'selected_train_seconds':val['train_wall_seconds_this_invocation'],
            'selected_peak_allocated_gib':val['peak_cuda_allocated_bytes']/2**30,
            'search_train_seconds_recorded':sum(load(Path(p)/'validation.json')['train_wall_seconds_this_invocation'] for p in sel['candidate_runs']),
            'lam':None,'stage1_train_seconds':0.,'val_feature_distance':None,'source':'M5-v1','gpu':val['gpu']})
    expected={(b,s) for b in spec['budgets'] for s in spec['seeds']}
    if len(rows)!=15 or {(r['budget'],r['seed']) for r in rows}!=expected:raise ValueError('Incomplete M5 grid')
    write_csv(args.results/'per_seed.csv',rows)
    dump(args.results/'run_summary.json',{'status':'COMPLETE','selected_models':15,'candidate_runs':45,
         'hardware_stratum':'RTX 6000 Ada; do not pool timing/memory with V100',
         'timing_caveat':'Per-invocation times; audit attempts/Slurm for any interrupted earlier invocation',
         'protocol_sha256':digest(args.protocol),'selection_lock_sha256':digest(root/'ALL_SELECTIONS_LOCKED.json')})


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('train','freeze-selections','evaluate','collect'))
    p.add_argument('--protocol',type=Path,default=Path('member-c/protocol/M5-v1.json'))
    p.add_argument('--root',type=Path,default=Path('member-c/runs/M5-v1'))
    p.add_argument('--data',type=Path,default=Path('member-c/artifacts/data-v1'))
    p.add_argument('--data-reference',type=Path,default=Path('member-a/remote-records/data-v1'))
    p.add_argument('--logs',type=Path,default=Path('member-c/logs'))
    p.add_argument('--results',type=Path,default=Path('member-c/results/M5-v1'))
    p.add_argument('--seed',nargs='+',type=int);a=p.parse_args(argv);spec=load(a.protocol)
    if spec['status']!='FROZEN':raise ValueError('Freeze protocol and code before execution')
    for name,sha in spec['code_hashes'].items():
        if digest(Path(__file__).parent/name)!=sha:raise ValueError(f'M5 code hash changed: {name}')
    a.root.mkdir(parents=True,exist_ok=True);a.logs.mkdir(parents=True,exist_ok=True)
    verify_data(a.data,a.data_reference)
    if a.phase=='train':
        if (a.root/'ALL_SELECTIONS_LOCKED.json').exists():raise ValueError('No training after global selection lock')
        if not a.seed or not set(a.seed)<=set(spec['seeds']):raise ValueError('Supply valid seeds')
        for seed in a.seed:
            for budget in spec['budgets']:
                for lr in spec['learning_rates']:ensure_candidate(a.root,budget,seed,lr,a,spec)
                select_cell(a.root,budget,seed,spec)
    elif a.phase=='freeze-selections':freeze_all(a.root,spec,a.protocol)
    elif a.phase=='evaluate':evaluate(a.root,a,spec)
    else:collect(a.root,a,spec)


if __name__=='__main__':main()
