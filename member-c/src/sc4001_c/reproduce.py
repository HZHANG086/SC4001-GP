"""Offline reproduction from archived evidence, without requiring a Git checkout.

The original full audit remains unchanged in analyze.py. This entry verifies its
1,879 input hashes, independently recomputes source choices and every saved test
metric, then invokes the same frozen statistics/error/bootstrap functions. It does
not claim to repeat data preparation, GPU inference or local weight verification.
"""
import argparse
from pathlib import Path
from .common import load,dump,digest,read_csv,read_predictions,selected_rows,run_path,DOMAINS,SIZES
from .analyze import statistics,errors
from .table_compare import compare_tables


def verify_inputs(repo):
    legacy=load(repo/'member-c/DELIVERY_MANIFEST.json')
    for name,sha in legacy['files'].items():
        if digest(repo/'member-c'/name)!=sha:raise ValueError('Frozen C evidence or analysis source changed: '+name)
    evidence=read_csv(repo/'member-c/results/C-v1/source_file_hashes.csv')
    if len(evidence)!=1879:raise ValueError('Incomplete archived input-hash ledger')
    for row in evidence:
        path=repo/row['path']
        if not path.resolve().is_relative_to(repo.resolve()):raise ValueError('Evidence path leaves package')
        if digest(path)!=row['sha256']:raise ValueError('Archived input changed: '+row['path'])
    # Source choices are recomputed without reading target metrics.
    a_spec=load(repo/'member-a/protocol/A-v1.json')
    aw=load(repo/'member-a/formal-results/results/A-v1/winners.json')['winners']
    for sel in aw:
        grid=a_spec['learning_rates'][sel['method']]
        candidates=[repo/'member-a/formal-results'/p for p in sel['candidate_runs']]
        winner=min(candidates,key=lambda p:(-load(p/'validation.json')['eval_accuracy'],grid.index(load(p/'run_config.json')['lr'])))
        if str(winner.relative_to(repo/'member-a/formal-results'))!=sel['run']:raise ValueError('A source choice changed')
    choices=list((repo/'member-b/formal-results/runs/B-v1').glob('selection-*.json'))
    if len(choices)!=60:raise ValueError('Incomplete B source choices')
    for path in choices:
        sel=load(path);candidates=[repo/'member-b/formal-results'/p for p in sel['candidate_runs']]
        winner=min(candidates,key=lambda p:(-load(p/'validation.json')['eval_accuracy'],candidates.index(p)))
        if str(winner.relative_to(repo/'member-b/formal-results'))!=sel['winner']:raise ValueError('B source choice changed')
    return len(evidence)


def predictions(repo,rows):
    selected={run_path(repo,r).resolve():r for r in rows};arrays={};canonical={};files=0;count=0;tested=0
    for member in ('a','b'):
        for path in sorted((repo/f'member-{member}/formal-results/runs').glob('*/*/test/metrics.json')):
            run=path.parent.parent;cfg=load(run/'run_config.json');record=load(path);scores={}
            for domain in DOMAINS:
                ids,y,p,q=read_predictions(run/'test'/f'{domain}_predictions.jsonl',*canonical.get(domain,(None,None)))
                if domain not in canonical:
                    expected={f'imdb:test:{i}' for i in range(25000)} if domain=='imdb' else {
                        f'{domain}:test:{i}' for i in load(repo/f'member-a/remote-records/data-v1/{domain}_test_indices.json')}
                    if set(ids)!=expected:raise ValueError('Fixed test IDs changed')
                    canonical[domain]=(ids,y)
                if len(ids)!=SIZES[domain] or len(ids)!=record[domain]['n']:raise ValueError('Wrong test size')
                scores[domain]=float((y==p).mean())
                if abs(scores[domain]-record[domain]['accuracy'])>1e-12:raise ValueError('Prediction metrics differ')
                if run.resolve() in selected:
                    cell=selected[run.resolve()]
                    if abs(scores[domain]-cell[domain+'_accuracy'])>1e-12:raise ValueError('Selected table differs')
                    arrays[cfg['method'],cfg['budget'],cfg['seed'],domain]=(y,p,q)
                files+=1;count+=len(ids)
            ood=(scores['yelp']+scores['amazon'])/2
            if abs(ood-record['ood_mean_accuracy'])>1e-12:raise ValueError('OOD mean differs')
            if run.resolve() in selected and abs(ood-selected[run.resolve()]['ood_accuracy'])>1e-12:
                raise ValueError('Selected OOD table differs')
            tested+=1
    if (tested,files,count,len(arrays))!=(150,450,5250000,270):raise ValueError('Incomplete archived test grid')
    return arrays,canonical,files,count


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path,default=Path('reproduced/C-v1'));a=p.parse_args(argv);repo=a.repo.resolve()
    if a.out.resolve()==(repo/'member-c/results/C-v1').resolve():raise ValueError('Do not overwrite frozen reference outputs')
    checked=verify_inputs(repo);rows=selected_rows(repo)
    expected={(m,b,s) for m in ('M1','M2','M3','M4','M6','L2SP') for b in (400,1600,6400) for s in (13,21,42,87,100)}
    if len(rows)!=90 or {(r['method'],r['budget'],r['seed']) for r in rows}!=expected:raise ValueError('Wrong selected grid')
    arrays,canonical,files,count=predictions(repo,rows)
    a.out.mkdir(parents=True,exist_ok=True);statistics(rows,a.out);errors(arrays,canonical,a.out)
    names=('accuracy_summary.csv','paired_comparisons.csv','cost_accounting.csv','error_transitions.csv',
           'case_sample_ids.csv','error_stability.csv','bootstrap_sensitivity.csv')
    matches={n:digest(a.out/n)==digest(repo/'member-c/results/C-v1'/n) for n in names}
    checks={n:compare_tables(repo/'member-c/results/C-v1'/n,a.out/n) for n in names}
    equivalent=all(x['equivalent'] for x in checks.values())
    dump(a.out/'reproduction_audit.json',{'status':'PASS' if equivalent else 'OUTPUT_DIFFERENCES_REQUIRE_REVIEW',
        'verified_input_files':checked,'prediction_files':files,'prediction_rows':count,'byte_identical_tables':matches,
        'table_checks':checks,
        'roundoff_policy':'Only approximate t-test probabilities and t-interval endpoints permit absolute differences <=1e-12. Every differing cell is logged. All other fields and every original input hash remain exact.',
        'scope':'Offline source-choice and full-prediction recomputation, then frozen statistical/error/bootstrap functions. No Git history required.',
        'not_repeated':'Training, raw data preparation, GPU inference, checkpoint-file hashing, or historical selection timing verification.'})
    if not equivalent:raise ValueError('Recomputed tables differ beyond the narrow roundoff policy; inspect the audit without overwriting reference outputs')
    print('OFFLINE_REPRODUCTION_PASS',files,'prediction files;',count,'rows;',len(checks),'equivalent tables')


if __name__=='__main__':main()
