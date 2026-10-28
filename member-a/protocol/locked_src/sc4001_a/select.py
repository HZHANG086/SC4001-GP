"""Freeze winners using only completed source-validation candidates."""
import argparse
import json
from pathlib import Path

GRIDS = {'M1': [1e-4, 1e-3, 1e-2], 'M2': [5e-6, 2e-5, 5e-5]}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='runs')
    p.add_argument('--out', default='winners.json')
    p.add_argument('--length', type=int, choices=[256, 512], required=True)
    a = p.parse_args()
    groups = {}
    for path in sorted(Path(a.runs).glob('*/run_config.json')):
        run = path.parent
        cfg = json.loads(path.read_text())
        if cfg['smoke'] or cfg.get('stage', 'formal') != 'formal' or cfg['length'] != a.length or not (run / 'COMPLETE').exists():
            continue
        if cfg['epochs'] != 10 or cfg['batch'] != 8 or cfg['accumulation'] != 4:
            raise SystemExit(f'Nonstandard formal training config: {run}')
        key = (cfg['method'], cfg['budget'], cfg['seed'])
        result = json.loads((run / 'validation.json').read_text())
        groups.setdefault(key, []).append((cfg, result, str(run)))
    winners = []
    for key, candidates in sorted(groups.items()):
        grid = GRIDS[key[0]]
        if sorted(c[0]['lr'] for c in candidates) != sorted(grid):
            raise SystemExit(f'{key}: need exactly one completed run for every LR in {grid}')
        for field in ('split_sha256', 'model_revision', 'length'):
            if len({c[0][field] for c in candidates}) != 1:
                raise SystemExit(f'{key}: candidates differ in {field}')
        winner = min(candidates, key=lambda c: (-c[1]['eval_accuracy'], grid.index(c[0]['lr'])))
        winners.append({'method': key[0], 'budget': key[1], 'seed': key[2],
                        'run': winner[2], 'source_val_accuracy': winner[1]['eval_accuracy'],
                        'lr': winner[0]['lr'], 'candidate_runs': [c[2] for c in candidates]})
    if not winners:
        raise SystemExit('No completed formal candidate groups')
    with Path(a.out).open('x') as f:
        json.dump({'selection': 'IMDb validation accuracy only', 'length': a.length,
                   'winners': winners}, f, indent=2)
    print(f'Frozen {len(winners)} winners. Incomplete groups are rejected; absent groups are not inferred.')

if __name__ == '__main__':
    main()
