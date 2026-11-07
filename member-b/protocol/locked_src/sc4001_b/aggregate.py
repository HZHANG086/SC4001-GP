"""Combined A+B summary over paired seeds; no candidate-level pseudo-replication.

Reads Member B's selection files and run folders plus Member A's committed per_seed.csv
(M1, M2) and writes, in the protocol's results directory:
  per_seed.csv   one row per selected method-budget-seed cell (A and B)
  ablation.csv   every evaluated M6 / L2-SP lambda candidate, plus M3 as lambda = 0
  summary.json   means, sample SDs, paired differences, cost and drift
  summary.md     the same as tables
  COMPLETE       only when all 4 B methods x 3 budgets x 5 seeds are present
"""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ('imdb_accuracy', 'yelp_accuracy', 'amazon_accuracy', 'ood_accuracy')
B_METHODS = ('M3', 'M4', 'M6', 'L2SP')
ORDER = ('M1', 'M2', 'M3', 'M4', 'M6', 'L2SP')
PAIRS = (('M3', 'M2'), ('M4', 'M3'), ('M6', 'M3'), ('M6', 'L2SP'), ('M6', 'M4'), ('L2SP', 'M3'))


def load(path):
    return json.loads(Path(path).read_text())


def train_seconds(run):
    return load(Path(run) / 'validation.json')['train_wall_seconds_this_invocation']


def test_metrics(run):
    m = load(Path(run) / 'test' / 'metrics.json')
    return {'imdb_accuracy': m['imdb']['accuracy'], 'yelp_accuracy': m['yelp']['accuracy'],
            'amazon_accuracy': m['amazon']['accuracy'], 'ood_accuracy': m['ood_mean_accuracy']}


def b_rows(runs_root):
    rows = []
    for path in sorted(Path(runs_root).glob('selection-*.json')):
        sel = load(path)
        method = path.stem.split('-')[1]
        run = Path(sel['winner'])
        cfg, val = load(run / 'run_config.json'), load(run / 'validation.json')
        stage1_seconds = train_seconds(sel['stage1_run']) if sel.get('stage1_run') else 0.0
        row = {'method': method, 'budget': cfg['budget'], 'seed': cfg['seed'], 'lr': cfg['lr'],
               'lam': cfg['lam'], 'run': str(run), 'source_val_accuracy': sel['source_val_accuracy'],
               **test_metrics(run),
               'trainable_parameters': load(run / 'parameters.json')['trainable'],
               'selected_train_seconds': val['train_wall_seconds_this_invocation'] + stage1_seconds,
               'selected_peak_allocated_gib': val['peak_cuda_allocated_bytes'] / 2**30,
               'search_train_seconds_recorded': sum(train_seconds(c) for c in sel['candidate_runs']) + stage1_seconds,
               'stage1_train_seconds': stage1_seconds,
               'val_feature_distance': val.get('val_feature_distance'), 'source': 'B-v1'}
        rows.append(row)
    return rows


def a_rows(per_seed_csv):
    with open(per_seed_csv, newline='') as stream:
        rows = list(csv.DictReader(stream))
    out = []
    for r in rows:
        row = {k: r[k] for k in r}
        for k in ('budget', 'seed', 'trainable_parameters'):
            row[k] = int(row[k])
        for k in ('lr', 'source_val_accuracy', 'selected_train_seconds', 'selected_peak_allocated_gib',
                  'search_train_seconds_recorded', *METRICS):
            row[k] = float(row[k])
        row.update({'lam': None, 'stage1_train_seconds': 0.0, 'val_feature_distance': None,
                    'source': 'A-v1'})
        out.append(row)
    return out


def ablation_rows(runs_root, by_key):
    rows = []
    for path in sorted(Path(runs_root).glob('selection-*.json')):
        method = path.stem.split('-')[1]
        if method not in ('M6', 'L2SP'):
            continue
        sel = load(path)
        for c in sel['candidate_runs']:
            run = Path(c)
            if not (run / 'test' / 'metrics.json').exists():
                continue
            cfg, val = load(run / 'run_config.json'), load(run / 'validation.json')
            rows.append({'method': method, 'budget': cfg['budget'], 'seed': cfg['seed'], 'lam': cfg['lam'],
                         'selected': str(run) == sel['winner'], 'source_val_accuracy': val['eval_accuracy'],
                         'val_feature_distance': val.get('val_feature_distance'), **test_metrics(run)})
        cell = load(Path(sel['winner']) / 'run_config.json')
        m3 = by_key.get(('M3', cell['budget'], cell['seed']))
        if m3:
            rows.append({'method': method, 'budget': m3['budget'], 'seed': m3['seed'], 'lam': 0.0,
                         'selected': False, 'source_val_accuracy': m3['source_val_accuracy'],
                         'val_feature_distance': m3['val_feature_distance'],
                         **{k: m3[k] for k in METRICS}})
    return rows


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    sd = statistics.stdev(values) if len(values) > 1 else None
    return statistics.mean(values), sd, len(values)


def fmt(mean, sd, scale=100, digits=2):
    if mean is None:
        return 'n/a'
    return f'{mean * scale:.{digits}f}' + ('' if sd is None else f' ± {sd * scale:.{digits}f}')


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', default='protocol/B-v1.json')
    p.add_argument('--a-per-seed', default='../member-a/formal-results/results/A-v1/per_seed.csv')
    p.add_argument('--runs', default=None, help='Override the protocol runs directory')
    p.add_argument('--out', default=None, help='Override the protocol results directory')
    a = p.parse_args(argv)
    spec = load(a.protocol)
    runs, out = Path(a.runs or spec['runs']), Path(a.out or spec['results'])
    rows = a_rows(a.a_per_seed) + b_rows(runs)
    by_key = {(r['method'], r['budget'], r['seed']): r for r in rows}
    if len(by_key) != len(rows):
        raise SystemExit('Duplicate method-budget-seed cells')
    seeds = spec['seeds']
    expected_b = {(m, b, s) for m in B_METHODS for b in spec['budgets'] for s in seeds}
    complete = expected_b <= set(by_key)
    out.mkdir(parents=True, exist_ok=True)

    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / 'per_seed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (ORDER.index(r['method']), r['budget'], r['seed'])))
    ablation = ablation_rows(runs, by_key)
    if ablation:
        with (out / 'ablation.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ablation[0]))
            writer.writeheader()
            writer.writerows(sorted(ablation, key=lambda r: (r['method'], r['budget'], r['seed'], r['lam'])))

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r['method'], r['budget'])].append(r)
    summary = []
    for (method, budget), group in sorted(grouped.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1])):
        entry = {'method': method, 'budget': budget, 'n_seeds': len(group)}
        for metric in METRICS + ('val_feature_distance',):
            entry[metric] = dict(zip(('mean', 'sample_std', 'n'), mean_sd([r[metric] for r in group])))
        for cost in ('trainable_parameters', 'selected_train_seconds', 'search_train_seconds_recorded',
                     'selected_peak_allocated_gib'):
            entry[cost] = statistics.mean(r[cost] for r in group)
        summary.append(entry)
    paired = []
    for left, right in PAIRS:
        for budget in spec['budgets']:
            common = [s for s in seeds if (left, budget, s) in by_key and (right, budget, s) in by_key]
            if not common:
                continue
            for metric in METRICS:
                diffs = [by_key[left, budget, s][metric] - by_key[right, budget, s][metric] for s in common]
                mean, sd, n = mean_sd(diffs)
                paired.append({'comparison': f'{left} - {right}', 'budget': budget, 'metric': metric,
                               'mean_difference': mean, 'sample_std': sd, 'n_seeds': n,
                               'per_seed_differences': dict(zip(common, diffs))})
    lam_table = defaultdict(list)
    for r in ablation:
        lam_table[(r['method'], r['budget'], r['lam'])].append(r)
    lam_summary = [{'method': m, 'budget': b, 'lam': lam, 'n_seeds': len(g),
                    **{k: mean_sd([r[k] for r in g])[0] for k in ('source_val_accuracy', 'val_feature_distance', *METRICS)}}
                   for (m, b, lam), g in sorted(lam_table.items())]
    (out / 'summary.json').write_text(json.dumps({'complete': complete, 'mean_std': summary, 'paired': paired,
                                                  'lambda_ablation': lam_summary}, indent=2))

    lines = ['# Members A + B: formal results', '',
             f'Status: {"COMPLETE" if complete else "INCOMPLETE (interim; cells still missing)"}. '
             'Mean ± sample SD over paired seeds, accuracy in %. M1/M2 from A-v1; M3/M4/M6/L2SP from B-v1.', '',
             '| Method | Labels | n | IMDb | Yelp | Amazon | OOD mean |', '|---|---:|---:|---:|---:|---:|---:|']
    for e in summary:
        lines.append(f"| {e['method']} | {e['budget']} | {e['n_seeds']} | " +
                     ' | '.join(fmt(e[m]['mean'], e[m]['sample_std']) for m in METRICS) + ' |')
    lines += ['', '## Paired differences (percentage points)', '',
              '| Comparison | Labels | IMDb | OOD mean | n |', '|---|---:|---:|---:|---:|']
    for left, right in PAIRS:
        for budget in spec['budgets']:
            cells = {r['metric']: r for r in paired if r['comparison'] == f'{left} - {right}' and r['budget'] == budget}
            if cells:
                imdb, ood = cells['imdb_accuracy'], cells['ood_accuracy']
                lines.append(f"| {left} − {right} | {budget} | {fmt(imdb['mean_difference'], imdb['sample_std'])} | "
                             f"{fmt(ood['mean_difference'], ood['sample_std'])} | {imdb['n_seeds']} |")
    lines += ['', '## Cost and representation drift (means over seeds)', '',
              '| Method | Labels | Trainable params | Selected train min | Search train min | Peak GiB | Val feature drift |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for e in summary:
        drift = e['val_feature_distance']['mean']
        lines.append(f"| {e['method']} | {e['budget']} | {e['trainable_parameters']:,.0f} | "
                     f"{e['selected_train_seconds'] / 60:.1f} | {e['search_train_seconds_recorded'] / 60:.1f} | "
                     f"{e['selected_peak_allocated_gib']:.2f} | {'n/a' if drift is None else f'{drift:.4f}'} |")
    if lam_summary:
        lines += ['', '## Lambda ablation (pre-registered; lambda = 0 is the selected M3 run)', '',
                  '| Method | Labels | lambda | n | Val acc | IMDb | OOD mean | Val feature drift |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in lam_summary:
            drift = r['val_feature_distance']
            lines.append(f"| {r['method']} | {r['budget']} | {r['lam']:g} | {r['n_seeds']} | "
                         f"{fmt(r['source_val_accuracy'], None)} | {fmt(r['imdb_accuracy'], None)} | "
                         f"{fmt(r['ood_accuracy'], None)} | {'n/a' if drift is None else f'{drift:.4f}'} |")
    lines += ['', 'A higher OOD mean with an unchanged or lower IMDb score is not evidence of better transfer by '
              'itself; read both absolute scores. Five seeds do not by themselves establish significance. '
              'M4 times include its stage-1 head training. Drift is the mean pooled-representation cosine '
              'distance from the pretrained encoder on IMDb validation data (correlational evidence only).']
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    if complete:
        (out / 'COMPLETE').write_text('All 60 B cells and 30 A cells summarised; see summary.md\n')
    print('\n'.join(lines[:40]))


if __name__ == '__main__':
    main()
