"""Bounded member-A grid, with source-only selection and completed-weight cleanup."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', default='protocol/A-v1.json')
    p.add_argument('--seeds', nargs='+', type=int, required=True)
    p.add_argument('--finish', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    spec = json.loads(Path(a.protocol).read_text())
    if not set(a.seeds) <= set(spec['seeds']) or len(set(a.seeds)) != len(a.seeds):
        raise SystemExit('Invalid/duplicate seed list')
    count = len(a.seeds) * len(spec['budgets']) * sum(len(v) for v in spec['learning_rates'].values())
    if a.dry_run:
        print(json.dumps({'candidate_runs': count, 'result_cells': len(a.seeds) * 6,
                          'length': spec['max_length'], 'stage': 'formal'}))
        return
    for name, expected in spec['code_hashes'].items():
        actual = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f'Frozen code changed: {name}')
    data = Path(spec['data'])
    if not (data / 'READY').is_file():
        raise SystemExit('Data not ready')
    if hashlib.sha256((data / 'manifest_hashes.json').read_bytes()).hexdigest() != spec['data_manifest_sha256']:
        raise SystemExit('Data version mismatch')
    from safetensors import safe_open
    root = Path(spec['runs'])
    root.mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    for seed in a.seeds:
        for budget in spec['budgets']:
            for method, grid in spec['learning_rates'].items():
                candidates = []
                for lr in grid:
                    run = root / f'{method}-b{budget}-s{seed}-lr{lr:.0e}'
                    command = [sys.executable, '-m', 'sc4001_a.train', '--data', str(data),
                               '--out', str(run), '--method', method, '--budget', str(budget),
                               '--seed', str(seed), '--lr', str(lr), '--length', str(spec['max_length']),
                               '--epochs', '10', '--batch', '8', '--accumulation', '4', '--stage', 'formal']
                    if not (run / 'COMPLETE').exists():
                        if run.exists():
                            # Quarantine incomplete saves; resume only a complete epoch checkpoint.
                            valid = []
                            for ck in (run / 'checkpoints').glob('checkpoint-*'):
                                required = ['trainer_state.json', 'optimizer.pt', 'scheduler.pt',
                                            'rng_state.pth', 'model.safetensors']
                                if all((ck / f).is_file() and (ck / f).stat().st_size > 0 for f in required):
                                    valid.append(ck)
                                else:
                                    ck.rename(ck.with_name('incomplete-' + ck.name))
                            if valid:
                                command.append('--resume')
                            else:
                                run.rename(run.with_name(run.name + f'.failed-{time.time_ns()}'))
                        started = time.monotonic()
                        log = Path('logs') / f'formal-{run.name}.log'
                        print('TRAIN', run, flush=True)
                        with log.open('a') as stream:
                            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                        with (root / 'attempts.jsonl').open('a') as stream:
                            stream.write(json.dumps({'run': str(run), 'command': command,
                                                     'wall_seconds': time.monotonic() - started,
                                                     'returncode': result.returncode}) + '\n')
                        if result.returncode:
                            raise SystemExit(f'Training failed: {run}. See {log}')
                    cfg = json.loads((run / 'run_config.json').read_text())
                    expected_cfg = {'method': method, 'budget': budget, 'seed': seed, 'lr': lr,
                                    'length': spec['max_length'], 'stage': 'formal',
                                    'data_manifest_sha256': spec['data_manifest_sha256']}
                    if any(cfg[k] != v for k, v in expected_cfg.items()):
                        raise SystemExit(f'Completed run differs from frozen protocol: {run}')
                    val = json.loads((run / 'validation.json').read_text())
                    candidates.append((val['eval_accuracy'], grid.index(lr), run))
                    # Each completed run already contains a standalone best/ model.
                    # Keep recovery states only while a candidate is incomplete.
                    best_weights = run / 'best/model.safetensors'
                    if best_weights.exists():
                        with safe_open(best_weights, framework='pt', device='cpu') as weights:
                            assert 'classifier.weight' in weights.keys()
                    elif not (run / 'WEIGHTS_PRUNED').exists():
                        raise SystemExit(f'Missing completed model: {run}')
                    ck = run / 'checkpoints'
                    if ck.exists():
                        if ck.is_symlink() or not ck.resolve().is_relative_to(root.resolve()):
                            raise SystemExit('Unsafe checkpoint cleanup path')
                        shutil.rmtree(ck)
                winner = min(candidates, key=lambda c: (-c[0], c[1]))
                with safe_open(winner[2] / 'best/model.safetensors', framework='pt', device='cpu') as weights:
                    assert 'classifier.weight' in weights.keys()
                write_json(root / f'selection-{method}-b{budget}-s{seed}.json',
                           {'winner': str(winner[2]), 'source_val_accuracy': winner[0],
                            'candidate_runs': [str(c[2]) for c in candidates],
                            'selection': 'source validation accuracy, LR-list order for exact ties'})
                for _, _, run in candidates:
                    best = run / 'best'
                    if run != winner[2] and best.exists():
                        if best.is_symlink() or not best.resolve().is_relative_to(root.resolve()):
                            raise SystemExit('Unsafe nonwinner cleanup path')
                        (run / 'WEIGHTS_PRUNED').write_text('Nonwinner weights removed after source-only selection; configs and metrics retained.\n')
                        shutil.rmtree(best)
                print('CELL_COMPLETE', method, budget, seed, 'winner', winner[2], flush=True)
    if a.finish:
        destination = Path(spec['results'])
        destination.mkdir(parents=True, exist_ok=True)
        winners_path = destination / 'winners.json'
        if not winners_path.exists():
            subprocess.run([sys.executable, '-m', 'sc4001_a.select', '--runs', str(root),
                            '--length', str(spec['max_length']), '--out', str(winners_path)], check=True)
        winners = json.loads(winners_path.read_text())['winners']
        if len(winners) != 30:
            raise SystemExit(f'Expected all 30 selected A cells, got {len(winners)}')
        for winner in winners:
            run = Path(winner['run'])
            if (run / 'test/metrics.json').exists():
                continue
            if (run / 'test').exists():
                (run / 'test').rename(run / f'test-incomplete-{time.time_ns()}')
            print('FINAL_EVALUATE', run, flush=True)
            log = Path('logs') / f'eval-{run.name}.log'
            with log.open('a') as stream:
                subprocess.run([sys.executable, '-m', 'sc4001_a.evaluate', '--run', str(run),
                                '--data', str(data), '--protocol-locked'], stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, '-m', 'sc4001_a.aggregate', '--winners', str(winners_path),
                        '--out', str(destination)], check=True)
        (destination / 'COMPLETE').write_text('All 30 A cells evaluated; see summary.md\n')

if __name__ == '__main__':
    main()
