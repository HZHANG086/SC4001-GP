"""Summarize exactly five paired seeds; no candidate-level pseudo-replication."""
import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(values):
    if len(values) != 5:
        raise ValueError(f'Need five seeds, got {len(values)}')
    return statistics.mean(values), statistics.stdev(values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--winners', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    winners = json.loads(Path(a.winners).read_text())['winners']
    rows = []
    for w in winners:
        run = Path(w['run'])
        metrics = json.loads((run / 'test/metrics.json').read_text())
        val = json.loads((run / 'validation.json').read_text())
        params = json.loads((run / 'parameters.json').read_text())
        row = {k: w[k] for k in ('method', 'budget', 'seed', 'lr', 'run', 'source_val_accuracy')}
        row.update({d + '_accuracy': metrics[d]['accuracy'] for d in ('imdb', 'yelp', 'amazon')})
        row.update({'ood_accuracy': metrics['ood_mean_accuracy'],
                    'trainable_parameters': params['trainable'],
                    'selected_train_seconds': val['train_wall_seconds_this_invocation'],
                    'selected_peak_allocated_gib': val['peak_cuda_allocated_bytes'] / 2**30,
                    'search_train_seconds_recorded': sum(json.loads((Path(c) / 'validation.json').read_text())[
                        'train_wall_seconds_this_invocation'] for c in w['candidate_runs'])})
        rows.append(row)
    if len(rows) != 30 or len({(r['method'], r['budget'], r['seed']) for r in rows}) != 30:
        raise ValueError('Expected 30 unique cells')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'per_seed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['method'], row['budget'])].append(row)
    result = []
    paired = []
    metrics = ('imdb_accuracy', 'yelp_accuracy', 'amazon_accuracy', 'ood_accuracy')
    for (method, budget), group in sorted(grouped.items()):
        assert {r['seed'] for r in group} == {13, 21, 42, 87, 100}
        for metric in metrics:
            mean, sd = summarize([r[metric] for r in group])
            result.append({'method': method, 'budget': budget, 'metric': metric,
                           'mean': mean, 'sample_std': sd, 'n_seeds': 5})
    by_key = {(r['method'], r['budget'], r['seed']): r for r in rows}
    for budget in (400, 1600, 6400):
        for metric in metrics:
            diffs = [by_key['M2', budget, seed][metric] - by_key['M1', budget, seed][metric]
                     for seed in (13, 21, 42, 87, 100)]
            mean, sd = summarize(diffs)
            paired.append({'budget': budget, 'metric': metric, 'M2_minus_M1_mean': mean,
                           'sample_std_of_paired_difference': sd, 'per_seed_differences': diffs})
    (out / 'summary.json').write_text(json.dumps({'mean_std': result, 'paired': paired}, indent=2))
    lines = ['# Member A: formal results', '', 'Five paired seeds; sample standard deviations. All scores are absolute accuracies.', '',
             '| Method | Labels | IMDb | Yelp | Amazon | OOD mean |', '|---|---:|---:|---:|---:|---:|']
    for method, budget in sorted(grouped):
        formatted = []
        for metric in metrics:
            entry = next(e for e in result if e['method'] == method and e['budget'] == budget and e['metric'] == metric)
            formatted.append(f"{entry['mean']*100:.2f} ± {entry['sample_std']*100:.2f}")
        lines.append(f"| {method} | {budget} | " + ' | '.join(formatted) + ' |')
    lines += ['', '## Paired M2 minus M1 (percentage points)', '',
              '| Labels | Metric | Mean paired difference | SD of paired differences |', '|---:|---|---:|---:|']
    for r in paired:
        lines.append(f"| {r['budget']} | {r['metric']} | {r['M2_minus_M1_mean']*100:+.2f} | {r['sample_std_of_paired_difference']*100:.2f} |")
    lines += ['', 'No significance claim follows automatically from five runs. See per_seed.csv for every selected run, LR and recorded cost.', '',
              'Recorded training seconds cover successful final invocations; if a run resumed after interruption, add earlier attempts and SLURM allocation time. Search cost and development/pilot cost must be reported separately. Per-example predictions remain in the selected run directories.']
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')

if __name__ == '__main__':
    main()
