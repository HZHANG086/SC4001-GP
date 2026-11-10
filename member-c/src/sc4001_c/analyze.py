"""Audit every archived run and prediction; write reproducible statistics.

Run from repository root: PYTHONPATH=member-c/src python -m sc4001_c.analyze
Raw A/B artifacts are opened read-only. This does not train or select a model.
"""
import argparse
import json
import math
import subprocess
from pathlib import Path
import numpy as np
from .common import (METHODS, BUDGETS, SEEDS, DOMAINS, SIZES, PAIRS, digest, load,
                     dump, read_csv, write_csv, selected_rows, run_path, read_predictions)
from .stats import paired_stats, mean_ci, holm, probability_metrics
from .bootstrap import crossed_bootstrap


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit(repo, out, rows):
    expected = {(m,b,s) for m in METHODS for b in BUDGETS for s in SEEDS}
    check(len(rows) == 90 and {(r['method'],r['budget'],r['seed']) for r in rows} == expected,
          'Missing/duplicate selected cell')
    evidence = []
    def record(p):
        evidence.append({'path': str(p.relative_to(repo)), 'sha256': digest(p), 'bytes': p.stat().st_size})
    data = repo/'member-a/remote-records/data-v1'
    manifests = load(data/'manifest_hashes.json')
    for name, sha in manifests.items():
        p = data/name
        if p.exists():
            check(digest(p) == sha, f'Data metadata hash mismatch: {p}')
            record(p)
    for s in SEEDS:
        splits = [load(data/f'splits/b{b}_s{s}.json') for b in BUDGETS]
        all_train = set().union(*(set(x['train_indices']) for x in splits))
        all_val = set().union(*(set(x['validation_indices']) for x in splits))
        check(not all_train & all_val, f'Train/validation leakage seed {s}')
        for i, (b,x) in enumerate(zip(BUDGETS,splits)):
            check(len(set(x['train_indices'])) == b*4//5, 'Wrong training budget')
            check(len(set(x['validation_indices'])) == b//5, 'Wrong validation budget')
            if i:
                for k in ('train_indices','validation_indices'):
                    check(set(splits[i-1][k]) <= set(x[k]), 'Non-nested split')
    a_spec, b_spec = [load(repo/f'member-{m}/protocol/{m.upper()}-v1.json') for m in ('a','b')]
    for member, spec in (('a',a_spec),('b',b_spec)):
        for name, sha in spec['code_hashes'].items():
            p = repo/f'member-{member}/protocol/locked_src'/f'sc4001_{member}'/name
            check(digest(p) == sha, f'Frozen code mismatch: {p}')
            record(p)
    # Recalculate every source-validation choice without reading its test metric.
    aw = load(repo/'member-a/formal-results/results/A-v1/winners.json')['winners']
    for w in aw:
        candidates = [repo/'member-a/formal-results'/c for c in w['candidate_runs']]
        grid = a_spec['learning_rates'][w['method']]
        winner = min(candidates, key=lambda p: (-load(p/'validation.json')['eval_accuracy'],
                                                grid.index(load(p/'run_config.json')['lr'])))
        check(str(winner.relative_to(repo/'member-a/formal-results')) == w['run'], 'A choice mismatch')
    bselections = sorted((repo/'member-b/formal-results/runs/B-v1').glob('selection-*.json'))
    check(len(bselections) == 60, 'Expected 60 B selections')
    for p in bselections:
        sel = load(p)
        cs = [repo/'member-b/formal-results'/c for c in sel['candidate_runs']]
        best = min(cs, key=lambda c: (-load(c/'validation.json')['eval_accuracy'], cs.index(c)))
        check(str(best.relative_to(repo/'member-b/formal-results')) == sel['winner'], f'B choice mismatch: {p}')
        record(p)
    configs = []
    missing_optional = []
    for member, expected_n in (('a',90),('b',195)):
        paths = sorted((repo/f'member-{member}/formal-results/runs').glob('*/*/run_config.json'))
        check(len(paths) == expected_n, f'{member}: wrong candidate count {len(paths)}')
        for p in paths:
            cfg = load(p)
            # A's archive intentionally omits run-level COMPLETE/environment files.
            # Validate its delivered config/history/validation instead; do not invent markers.
            if not (p.parent/'COMPLETE').exists():
                missing_optional.append(str((p.parent/'COMPLETE').relative_to(repo)))
            val = load(p.parent/'validation.json')
            hist = load(p.parent/'history.json')
            check(any(h.get('epoch', 0) >= 10 for h in hist), f'No final epoch evidence: {p}')
            check(any(abs(h.get('eval_accuracy', -1)-val['eval_accuracy'])<1e-12 for h in hist),
                  f'Chosen validation absent from history: {p}')
            best=max(h['eval_accuracy'] for h in hist if 'eval_accuracy' in h)
            check(abs(best-val['eval_accuracy'])<1e-12, f'Not best validation checkpoint: {p}')
            first=min(h['step'] for h in hist if h.get('eval_accuracy')==best)
            check(val['best_checkpoint'].endswith(f'checkpoint-{first}'), f'Best-epoch tie rule mismatch: {p}')
            check(not cfg['smoke'] and cfg['stage']=='formal', f'Nonformal run: {p}')
            check(cfg['model_revision'] == a_spec['model_revision'], 'Model revision mismatch')
            check(cfg['dataset_revisions'] == a_spec['dataset_revisions'], 'Dataset revisions differ')
            for key,value in (('length',512),('epochs',10),('batch',8),('accumulation',4)):
                check(cfg[key] == value, f'Inconsistent {key}: {p}')
            split = data/f"splits/b{cfg['budget']}_s{cfg['seed']}.json"
            check(cfg['split_sha256'] == digest(split), f'Split hash mismatch: {p}')
            check(cfg['data_manifest_sha256'] == a_spec['data_manifest_sha256'], 'Data manifest differs')
            for filename, sha in cfg['code_sha256'].items():
                name = 'a_'+filename.split('/')[-1] if filename.startswith('sc4001_a/') else filename
                check(digest(p.parent/'source'/name) == sha, f'Run source mismatch: {p} {filename}')
            configs.append(cfg)
            for filename in ('run_config.json','validation.json','history.json','parameters.json','environment.txt'):
                if (p.parent/filename).exists():
                    record(p.parent/filename)
                elif filename=='environment.txt':
                    missing_optional.append(str((p.parent/filename).relative_to(repo)))
                else:
                    raise ValueError(f'Missing {p.parent/filename}')
    # Recompute metrics for ALL saved tests, including preregistered lambda candidates.
    canonical = {}
    selected = {str(run_path(repo,r)):r for r in rows}
    arrays, metric_rows = {}, []
    pred_files = pred_count = tested_runs = 0
    for member in ('a','b'):
        for metrics_path in sorted((repo/f'member-{member}/formal-results/runs').glob('*/*/test/metrics.json')):
            metrics = load(metrics_path)
            cfg = load(metrics_path.parent.parent/'run_config.json')
            domain_accuracies = {}
            for domain in DOMAINS:
                path = metrics_path.parent/f'{domain}_predictions.jsonl'
                ids, labels, preds, probs = read_predictions(path, *canonical.get(domain,(None,None)))
                if domain not in canonical:
                    canonical[domain] = ids, labels
                    expected_ids = ({f'imdb:test:{i}' for i in range(25000)} if domain=='imdb' else
                                    {f'{domain}:test:{i}' for i in load(data/f'{domain}_test_indices.json')})
                    check(set(ids)==expected_ids, f'Test index mismatch {domain}')
                check(len(ids)==SIZES[domain], f'Test sample count mismatch {path}')
                accuracy = float((labels==preds).mean())
                check(abs(accuracy-metrics[domain]['accuracy']) < 1e-12, f'Accuracy mismatch {path}')
                check(len(ids)==metrics[domain]['n'], f'Metric sample count mismatch {path}')
                domain_accuracies[domain] = accuracy
                record(path)
                pred_files += 1
                pred_count += len(ids)
                key = (cfg['method'],cfg['budget'],cfg['seed'],domain)
                if str(metrics_path.parent.parent) in selected:
                    r = selected[str(metrics_path.parent.parent)]
                    check(abs(accuracy-r[f'{domain}_accuracy'])<1e-12, 'Selected CSV mismatch')
                    arrays[key] = (labels,preds,probs)
                    metric_rows.append({'method':cfg['method'],'budget':cfg['budget'],'seed':cfg['seed'],
                                        'domain':domain,**probability_metrics(labels,preds,probs)})
            ood = (domain_accuracies['yelp']+domain_accuracies['amazon'])/2
            check(abs(ood-metrics['ood_mean_accuracy'])<1e-12, 'OOD metric mismatch')
            if str(metrics_path.parent.parent) in selected:
                check(abs(ood-selected[str(metrics_path.parent.parent)]['ood_accuracy'])<1e-12, 'OOD CSV mismatch')
            tested_runs += 1
    check(tested_runs == 150 and pred_files == 450, f'Missing tests: {tested_runs}/{pred_files}')
    check(len(arrays)==270, 'Missing selected predictions')
    manifest_models = read_csv(repo/'member-b/formal-results/selected_models.tsv', '\t')
    check(len(manifest_models)==60, 'Wrong model manifest length')
    for r in manifest_models:
        match = [x for x in rows if (x['method'],x['budget'],x['seed'])==(r['method'],int(r['budget']),int(r['seed']))]
        check(len(match)==1 and r['model_dir']==match[0]['run']+'/best', 'Model manifest mismatch')
    write_csv(out/'source_file_hashes.csv', evidence)
    write_csv(out/'prediction_metrics.csv', metric_rows)
    dump(out/'audit.json', {'status':'PASSED_ARCHIVED_RECORDS', 'source_commit':subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=repo,text=True).strip(), 'candidate_runs':len(configs),
        'selected_cells':len(rows),'tested_runs':tested_runs,'prediction_files':pred_files,
        'prediction_rows':pred_count,'hashed_files':len(evidence),'model_manifest_rows':60,
        'raw_data_content':'Not yet checked by this stage; see text_join_audit.json',
        'trained_weights':'Not downloaded; paths checked against selected configuration only',
        'selection_timing':'Selection content verified; real historical ordering cannot be proven by file contents alone',
        'missing_optional_archive_files':missing_optional,
        'retraining':'Not performed'})
    print(f'Audited {len(configs)} runs and {pred_count:,} predictions.', flush=True)
    return arrays, canonical


def statistics(rows, out):
    by_key = {(r['method'],r['budget'],r['seed']):r for r in rows}
    summaries = []
    metrics = [f'{d}_accuracy' for d in DOMAINS]+['ood_accuracy']
    for m in METHODS:
        for b in BUDGETS:
            for metric in metrics:
                values = [100*by_key[m,b,s][metric] for s in SEEDS]
                summaries.append({'method':m,'budget':b,'metric':metric,**mean_ci(values),'n_seeds':5})
    paired = []
    for left,right in PAIRS:
        for b in BUDGETS:
            for metric in metrics:
                d = [100*(by_key[left,b,s][metric]-by_key[right,b,s][metric]) for s in SEEDS]
                family = 'focal_3' if (left,right,metric)==('M6','M3','ood_accuracy') else 'secondary_69'
                paired.append({'comparison':f'{left}-{right}','budget':b,'metric':metric,'family':family,
                               **paired_stats(d),**{f'diff_seed_{s}':v for s,v in zip(SEEDS,d)}})
    for family in ('focal_3','secondary_69'):
        ix = [i for i,r in enumerate(paired) if r['family']==family]
        for col in ('p_signflip_exact','p_t_approx'):
            adj = holm([paired[i][col] for i in ix])
            for i,v in zip(ix,adj):
                paired[i][col+'_holm'] = v
    write_csv(out/'accuracy_summary.csv',summaries)
    write_csv(out/'paired_comparisons.csv',paired)
    # Absolute costs of the recorded implementation, never clean algorithm benchmarks.
    costs = []
    for r in rows:
        m,b,s = r['method'],r['budget'],r['seed']
        upstream = (by_key['M3',b,s]['search_train_seconds_recorded'] if m in ('M6','L2SP') else
                    by_key['M1',b,s]['search_train_seconds_recorded'] if m=='M4' else 0)
        costs.append({'method':m,'budget':b,'seed':s,
                      'selected_train_seconds_recorded':r['selected_train_seconds'],
                      'incremental_search_seconds_recorded':r['search_train_seconds_recorded'],
                      'upstream_search_seconds_recorded':upstream,
                      'standalone_search_seconds_recorded':r['search_train_seconds_recorded']+upstream,
                      'standalone_is_lower_bound':m=='M6' and b==6400 and s==21,
                      'peak_gib_recorded_implementation':r['selected_peak_allocated_gib'],
                      'diagnostic_teacher_on_gpu':m in ('M3','M4','L2SP'),
                      'trainable_parameters':r['trainable_parameters']})
    write_csv(out/'cost_accounting.csv',costs)


def errors(arrays, canonical, out):
    results, cases = [], []
    import hashlib
    for left,right in PAIRS:
        for b in BUDGETS:
            for s in SEEDS:
                for domain in DOMAINS:
                    y,lp,_ = arrays[left,b,s,domain]
                    _,rp,_ = arrays[right,b,s,domain]
                    lc,rc = lp==y,rp==y
                    groups = {'both_correct':lc&rc,'both_wrong':~lc&~rc,
                              'fixed_by_left':lc&~rc,'broken_by_left':~lc&rc}
                    results.append({'comparison':f'{left}-{right}','budget':b,'seed':s,'domain':domain,
                                    'n':len(y),**{k:int(v.sum()) for k,v in groups.items()},
                                    'net_change_pp':float(100*(lc.mean()-rc.mean()))})
                    # All 3 budgets/domains, seed 13 fixed before text inspection.
                    # Select one example per category by deterministic ID hash.
                    if (left,right,s)==('M6','M3',13):
                        ids = canonical[domain][0]
                        for cat,mask in groups.items():
                            indices = np.flatnonzero(mask)
                            if len(indices):
                                i = min(indices, key=lambda j: hashlib.sha256(
                                    ('C-v1-case|'+ids[j]).encode()).hexdigest())
                                cases.append({'budget':b,'seed':s,'domain':domain,'category':cat,
                                              'example_id':ids[i],'label':int(y[i]),'M3_prediction':int(rp[i]),
                                              'M6_prediction':int(lp[i]),
                                              'M3_p_positive':float(arrays['M3',b,s,domain][2][i]),
                                              'M6_p_positive':float(arrays['M6',b,s,domain][2][i])})
    write_csv(out/'error_transitions.csv',results)
    write_csv(out/'case_sample_ids.csv',cases)
    # Shared test examples recur across seeds. Report item stability descriptively,
    # not as 5*n independent observations or a test of training-method superiority.
    stability=[]
    for b in BUDGETS:
        for domain in DOMAINS:
            y=canonical[domain][1]
            left=np.stack([arrays['M6',b,s,domain][1]==y for s in SEEDS])
            right=np.stack([arrays['M3',b,s,domain][1]==y for s in SEEDS])
            fixed=((left)&(~right)).sum(0)
            broken=((~left)&right).sum(0)
            stability.append({'budget':b,'domain':domain,'n_unique_examples':len(y),
                              'fixed_in_at_least_3_seeds':int((fixed>=3).sum()),
                              'broken_in_at_least_3_seeds':int((broken>=3).sum()),
                              'ever_fixed_and_ever_broken':int(((fixed>0)&(broken>0)).sum())})
    write_csv(out/'error_stability.csv',stability)
    boot=[]
    for b in BUDGETS:
        matrices=[];labels=[]
        for d in ('yelp','amazon'):
            y=canonical[d][1]
            matrices.append(np.stack([(arrays['M6',b,s,d][1]==y).astype(int)-
                                      (arrays['M3',b,s,d][1]==y).astype(int) for s in SEEDS]))
            labels.append(y)
        boot.append({'comparison':'M6-M3','budget':b,'metric':'ood_accuracy',
                     **crossed_bootstrap(matrices,labels)})
    write_csv(out/'bootstrap_sensitivity.csv',boot)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path,default=Path('member-c/results/C-v1'))
    a=p.parse_args(); repo=a.repo.resolve(); a.out.mkdir(parents=True,exist_ok=True)
    rows=selected_rows(repo)
    arrays, canonical=audit(repo,a.out,rows)
    statistics(rows,a.out)
    errors(arrays,canonical,a.out)
    ledger=read_csv(repo/'member-b/formal-results/job-ledger.tsv','\t')
    seconds=sum(sum(int(t)*s for t,s in zip(r['elapsed'].split(':'),(3600,60,1))) for r in ledger)
    dump(a.out/'cost_caveats.json',{'B_job_count':len(ledger),'B_allocation_seconds_including_eval':seconds,
         'B_allocation_hours_including_eval':seconds/3600,
         'resumed_candidate':'M6-b6400-s21-lam1','job_ids':['72808','72909'],
         'impact':'Search time for M6 b6400 s21 omits prior invocation; selected model lam0.01 itself was not resumed.',
         'unrecoverable_from_validation_json':'Full pre-interruption per-candidate training time',
         'do_not_sum_standalone_costs':'Upstream searches are shared in this actual project.',
         'memory':'B M3/M4/L2SP include an otherwise unnecessary diagnostic teacher; raw memory is implementation-specific.'})
    print('Completed archive audit, statistics, cost accounting, and prediction-level error analysis.',flush=True)


if __name__=='__main__':
    main()
