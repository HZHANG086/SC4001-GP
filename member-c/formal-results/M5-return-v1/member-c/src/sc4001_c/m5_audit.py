"""Audit M5 records and predictions before including them in the final report."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .common import load, dump, digest
from .stats import paired_stats, holm, probability_metrics
from .lora_sweep import validate_run, run_name


def audit_predictions(path, reference):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    ids = [r['example_id'] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(reference):
        raise ValueError(f'Duplicate/missing test examples: {path}')
    y = np.array([r['label'] for r in rows])
    p = np.array([r['p_positive'] for r in rows], dtype=float)
    pred = np.array([r['prediction'] for r in rows])
    if any(r['label'] != reference[r['example_id']] for r in rows):
        raise ValueError('Test labels differ from frozen A/B reference')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid probabilities')
    if not np.array_equal(pred, (p > .5).astype(int)):
        # Original A evaluates the two softmax columns; tolerate only roundoff ties.
        bad = (pred != (p > .5)) & (abs(p - .5) > 1e-7)
        if bad.any(): raise ValueError('Predictions inconsistent with probabilities')
    return probability_metrics(y, pred, p), len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('member-c/runs/M5-v1'))
    parser.add_argument('--protocol', type=Path, default=Path('member-c/protocol/M5-v1.json'))
    parser.add_argument('--out', type=Path, default=Path('member-c/results/M5-audit'))
    parser.add_argument('--b-results', type=Path, default=Path('member-b/formal-results'))
    a = parser.parse_args(argv)
    spec = load(a.protocol); lock = load(a.root/'ALL_SELECTIONS_LOCKED.json')
    if lock['protocol_sha256'] != digest(a.protocol): raise ValueError('Protocol lock changed')
    expected = {(b,s) for b in spec['budgets'] for s in spec['seeds']}
    if len(lock['selections']) != 15 or {(s['budget'],s['seed']) for s in lock['selections']} != expected:
        raise ValueError('Wrong selected grid')
    refs = {}
    for domain in ('imdb','yelp','amazon'):
        candidates = sorted((a.b_results/'runs/B-v1').glob(f'M3-b400-s13-*/test/{domain}_predictions.jsonl'))
        if len(candidates) != 1: raise ValueError('Ambiguous archived reference predictions')
        with candidates[0].open() as f:
            rows = [json.loads(line) for line in f]
        refs[domain] = {r['example_id']:r['label'] for r in rows}
    metrics=[]; selected=[]; hashes={}; interrupted=[]
    for sel in lock['selections']:
        b,s = sel['budget'],sel['seed']
        candidates = [a.root/run_name(b,s,lr) for lr in spec['learning_rates']]
        vals = [validate_run(p,spec,b,s,lr) for p,lr in zip(candidates,spec['learning_rates'])]
        winner = max(range(3), key=lambda i:vals[i]['eval_accuracy'])
        run = candidates[winner]
        if Path(sel['winner']).name != run.name: raise ValueError('Wrong validation winner')
        if sel['validation_sha256'] != [digest(p/'validation.json') for p in candidates]:
            raise ValueError('Validation record changed after selection')
        for p in candidates:
            cfg = load(p/'run_config.json')
            if cfg['protocol_sha256'] != digest(a.protocol): raise ValueError('Mixed training protocols')
            events = [json.loads(x) for x in (p/'invocations.jsonl').read_text().splitlines()]
            if sum(e['event']=='start' for e in events)>1: interrupted.append(str(p))
        if digest(run/'best/model.safetensors') != vals[winner]['checkpoint_weight_sha256']:
            raise ValueError('Selected weights changed')
        recorded = load(run/'test/metrics.json'); results={}
        for domain in refs:
            path = run/'test'/f'{domain}_predictions.jsonl'
            m,n = audit_predictions(path,refs[domain]); results[domain+'_accuracy']=m['accuracy']
            if n != recorded[domain]['n'] or abs(m['accuracy']-recorded[domain]['accuracy'])>1e-12:
                raise ValueError('Metrics do not reproduce from predictions')
            metrics.append({'method':'M5','budget':b,'seed':s,'domain':domain,'n':n,**m})
            hashes[str(path)] = digest(path)
        results['ood_accuracy'] = (results['yelp_accuracy']+results['amazon_accuracy'])/2
        if abs(results['ood_accuracy']-recorded['ood_mean_accuracy'])>1e-12: raise ValueError('OOD mean mismatch')
        selected.append({'method':'M5','budget':b,'seed':s,**results})
    a.out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(metrics).to_csv(a.out/'prediction_metrics.csv',index=False)
    m5 = pd.DataFrame(selected)
    old = pd.read_csv(a.b_results/'results/B-v1/per_seed.csv'); comparisons=[]
    for b in spec['budgets']:
        target=m5[m5.budget==b].set_index('seed').loc[spec['seeds']]
        for method in ('M1','M2','M3','M4','M6','L2SP'):
            ref=old[(old.budget==b)&(old.method==method)].set_index('seed').loc[spec['seeds']]
            differences=100*(target.ood_accuracy-ref.ood_accuracy).to_numpy()
            comparisons.append({'comparison':'M5-'+method,'budget':b,'metric':'ood_accuracy',
                'family':'M5_secondary_18',**paired_stats(differences),
                **{f'diff_seed_{s}':v for s,v in zip(spec['seeds'],differences)}})
    for p in ('p_signflip_exact','p_t_approx'):
        for row,value in zip(comparisons,holm([r[p] for r in comparisons])): row[p+'_holm']=value
    pd.DataFrame(comparisons).to_csv(a.out/'paired_comparisons.csv',index=False)
    dump(a.out/'audit.json',{'status':'PASS','models':15,'candidate_records':45,'prediction_files':45,
        'protocol_sha256':digest(a.protocol),'selection_lock_sha256':digest(a.root/'ALL_SELECTIONS_LOCKED.json'),
        'prediction_sha256':hashes,'interrupted_candidates':interrupted,
        'statistical_note':'18 additional OOD contrasts, separate exploratory Holm family; original focal M6 family unchanged. Unadjusted t intervals; n=5 paired seeds. Fixed test domains.',
        'timing_note':'If interrupted_candidates nonempty, successful invocation times are lower bounds; use Slurm allocation ledger too.'})
    print('M5_AUDIT_PASS')


if __name__=='__main__': main()
