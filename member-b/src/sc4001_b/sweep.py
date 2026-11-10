"""Bounded Member B grid with source-only selection, locked test evaluation and weight cleanup.

    python -m sc4001_b.sweep --phase M3 --seeds 13 21 42 87 100
    python -m sc4001_b.sweep --phase M6 --seeds 13 21          # needs the M3 selection of each cell
    python -m sc4001_b.sweep --finish                          # combined A+B summary

Per cell (method, budget, seed):
  M3       candidates = learning rates            (protocol learning_rates.M3)
  M4       stage 1 = M4S1 with Member A's selected M1 LR for the same cell, then
           candidates = stage-2 learning rates    (protocol learning_rates.M4)
  M6/L2SP  LR = this cell's selected M3 LR, candidates = lambdas (protocol lambdas.*);
           lambda = 0 is M3 itself and is not a candidate.
Selection uses IMDb validation accuracy only (ties: protocol list order). The selection file
is written before any test evaluation of the cell. Test evaluation (Member A's locked
evaluate.py) runs on every winner and, as a pre-registered ablation, on every M6/L2SP
candidate. Non-winner weights are removed after that; configs, metrics and predictions stay.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PHASES = ('M3', 'M4', 'M6', 'L2SP')
ABLATION_PHASES = ('M6', 'L2SP')


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_name(method, budget, seed, lr=None, lam=None):
    """Stable candidate folder names, e.g. M3-b400-s13-lr2e-05, M6-b400-s13-lam1."""
    tail = f'lr{lr:.0e}' if lam is None else f'lam{lam:g}'
    return f'{method}-b{budget}-s{seed}-{tail}'


def check_frozen_code(spec, a_spec):
    import sc4001_a
    import sc4001_b
    b_dir, a_dir = Path(sc4001_b.__file__).parent, Path(sc4001_a.__file__).parent
    for name, expected in spec['code_hashes'].items():
        if sha256(b_dir / name) != expected:
            raise SystemExit(f'Frozen B code changed: {name}')
    for name in spec['member_a_code_used']:
        if sha256(a_dir / name) != a_spec['code_hashes'][name]:
            raise SystemExit(f'Member A code differs from A-v1: {name}')


def a_m1_lr(a_winners, budget, seed):
    for w in a_winners:
        if (w['method'], w['budget'], w['seed']) == ('M1', budget, seed):
            return w['lr']
    raise SystemExit(f'No Member A M1 winner for b{budget} s{seed}')


def selected_m3_lr(root, budget, seed):
    path = root / f'selection-M3-b{budget}-s{seed}.json'
    if not path.exists():
        raise SystemExit(f'Run phase M3 for b{budget} s{seed} first ({path} missing)')
    return json.loads(path.read_text())['lr']


def ensure_trained(command, run, root, expected):
    """Train (or resume) one run unless COMPLETE; then check its config against the protocol."""
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
                command = command + ['--resume']
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
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise SystemExit(f'Completed run differs from frozen protocol: {run}')
    return json.loads((run / 'validation.json').read_text())


def remove_tree(path, root):
    if path.exists():
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise SystemExit(f'Unsafe cleanup path: {path}')
        shutil.rmtree(path)


def evaluate(run, data):
    """Member A's locked test evaluation (writes run/test/)."""
    if (run / 'test' / 'metrics.json').exists():
        return
    if (run / 'test').exists():
        (run / 'test').rename(run / f'test-incomplete-{time.time_ns()}')
    print('FINAL_EVALUATE', run, flush=True)
    with (Path('logs') / f'eval-{run.name}.log').open('a') as stream:
        subprocess.run([sys.executable, '-m', 'sc4001_a.evaluate', '--run', str(run),
                        '--data', str(data), '--protocol-locked'], stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def check_data_matches_a(data, a_manifest_path):
    """Require identical content-defining files (sources, splits, target indices, audit) to A-v1.

    datasets' state.json / dataset_info.json can embed cache-dependent fingerprints, so the
    whole-manifest hash may differ between accounts even when the data are identical.
    """
    ours = json.loads((data / 'manifest_hashes.json').read_text())
    reference = json.loads(Path(a_manifest_path).read_text())
    content = {k: v for k, v in reference.items() if not k.endswith(('state.json', 'dataset_info.json'))}
    if len(content) != 19:   # sources, audit, 15 splits, 2 target index files
        raise SystemExit(f'Unexpected A-v1 manifest layout ({len(content)} content files)')
    differing = sorted(k for k, v in content.items() if ours.get(k) != v)
    if differing:
        raise SystemExit(f'Data differ from A-v1 in {differing}; rebuild with --source-lock')


def run_cell(phase, budget, seed, spec, a_winners, data, root, manifest):
    common = {'budget': budget, 'seed': seed, 'length': spec['max_length'], 'stage': 'formal',
              'epochs': spec['epochs'], 'batch': spec['batch_size'],
              'accumulation': spec['gradient_accumulation'], 'smoke': False,
              'data_manifest_sha256': manifest}
    def train_command(run, method, lr, lam=None, stage1=None):
        command = [sys.executable, '-m', 'sc4001_b.train', '--data', str(data), '--out', str(run),
                   '--method', method, '--budget', str(budget), '--seed', str(seed), '--lr', str(lr),
                   '--length', str(spec['max_length']), '--epochs', str(spec['epochs']),
                   '--batch', str(spec['batch_size']), '--accumulation', str(spec['gradient_accumulation']),
                   '--stage', 'formal']
        if lam is not None:
            command += ['--lam', str(lam)]
        if stage1 is not None:
            command += ['--stage1', str(stage1)]
        return command

    stage1 = None
    if phase == 'M3':
        grid = [(lr, None) for lr in spec['learning_rates']['M3']]
    elif phase == 'M4':
        lr1 = a_m1_lr(a_winners, budget, seed)
        stage1 = root / run_name('M4S1', budget, seed, lr=lr1)
        ensure_trained(train_command(stage1, 'M4S1', lr1), stage1, root,
                       common | {'method': 'M4S1', 'lr': lr1, 'lam': None})
        grid = [(lr, None) for lr in spec['learning_rates']['M4']]
    else:
        lr = selected_m3_lr(root, budget, seed)
        grid = [(lr, lam) for lam in spec['lambdas'][phase]]

    candidates = []
    for index, (lr, lam) in enumerate(grid):
        run = root / run_name(phase, budget, seed, lr=lr, lam=lam)
        if stage1 is not None and not (run / 'COMPLETE').exists() and not (stage1 / 'best').exists():
            raise SystemExit(f'Stage 1 weights already pruned but {run} is incomplete')
        expected = common | {'method': phase, 'lr': lr, 'lam': lam}
        if stage1 is not None:
            expected['stage1'] = str(stage1)
        val = ensure_trained(train_command(run, phase, lr, lam, stage1), run, root, expected)
        candidates.append((val['eval_accuracy'], index, run, lr, lam))
        remove_tree(run / 'checkpoints', root)
        if not (run / 'best/model.safetensors').exists() and not (run / 'WEIGHTS_PRUNED').exists():
            raise SystemExit(f'Missing completed model: {run}')
    if stage1 is not None:
        remove_tree(stage1 / 'checkpoints', root)

    winner = min(candidates, key=lambda c: (-c[0], c[1]))
    selection = root / f'selection-{phase}-b{budget}-s{seed}.json'
    if not selection.exists():
        write_json(selection, {'winner': str(winner[2]), 'source_val_accuracy': winner[0],
                               'lr': winner[3], 'lam': winner[4],
                               'candidate_runs': [str(c[2]) for c in candidates],
                               'stage1_run': str(stage1) if stage1 else None,
                               'selection': 'IMDb validation accuracy, protocol list order for exact ties'})
    elif json.loads(selection.read_text())['winner'] != str(winner[2]):
        raise SystemExit(f'Recomputed winner differs from frozen selection: {selection}')
    # Test access only after the cell's selection is frozen on disk.
    to_evaluate = [c[2] for c in candidates] if phase in ABLATION_PHASES else [winner[2]]
    for run in to_evaluate:
        if (run / 'best').exists() or (run / 'test' / 'metrics.json').exists():
            evaluate(run, data)
        else:
            raise SystemExit(f'Cannot evaluate {run}: weights pruned before evaluation')
    for _, _, run, _, _ in candidates:
        if run != winner[2] and (run / 'best').exists():
            remove_tree(run / 'best', root)
            (run / 'WEIGHTS_PRUNED').write_text('Nonwinner weights removed after source-only selection '
                                                'and pre-registered evaluation; configs and metrics retained.\n')
    if stage1 is not None and (stage1 / 'best').exists():
        remove_tree(stage1 / 'best', root)
        (stage1 / 'WEIGHTS_PRUNED').write_text('Stage-1 weights removed after all stage-2 candidates completed.\n')
    print('CELL_COMPLETE', phase, budget, seed, 'winner', winner[2], flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--protocol', default='protocol/B-v1.json')
    p.add_argument('--a-protocol', default='../member-a/protocol/A-v1.json')
    p.add_argument('--a-winners', default='../member-a/formal-results/results/A-v1/winners.json')
    p.add_argument('--a-manifest', default='../member-a/remote-records/data-v1/manifest_hashes.json')
    p.add_argument('--phase', choices=PHASES)
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--budgets', nargs='+', type=int)
    p.add_argument('--finish', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    spec = json.loads(Path(a.protocol).read_text())
    a_spec = json.loads(Path(a.a_protocol).read_text())
    for key in ('model_id', 'model_revision', 'dataset_revisions', 'max_length', 'epochs',
                'batch_size', 'gradient_accumulation', 'precision', 'budgets', 'seeds'):
        if spec[key] != a_spec[key]:
            raise SystemExit(f'B protocol differs from A-v1 in shared setting {key}')
    root = Path(spec['runs'])
    if a.finish:
        subprocess.run([sys.executable, '-m', 'sc4001_b.aggregate', '--protocol', a.protocol,
                        '--a-per-seed', str(Path(a.a_winners).with_name('per_seed.csv'))], check=True)
        return
    if a.phase is None or not a.seeds:
        p.error('--phase and --seeds are required unless --finish')
    seeds, budgets = a.seeds, a.budgets or spec['budgets']
    if not set(seeds) <= set(spec['seeds']) or len(set(seeds)) != len(seeds):
        raise SystemExit('Invalid/duplicate seed list')
    if not set(budgets) <= set(spec['budgets']):
        raise SystemExit('Invalid budget list')
    if a.dry_run:
        per_cell = 3 + (1 if a.phase == 'M4' else 0)
        print(json.dumps({'phase': a.phase, 'cells': len(seeds) * len(budgets),
                          'training_runs': len(seeds) * len(budgets) * per_cell}))
        return
    check_frozen_code(spec, a_spec)
    data = Path(spec['data'])
    if not (data / 'READY').is_file():
        raise SystemExit('Data not ready')
    check_data_matches_a(data, a.a_manifest)
    manifest = sha256(data / 'manifest_hashes.json')
    a_winners = json.loads(Path(a.a_winners).read_text())['winners']
    root.mkdir(parents=True, exist_ok=True)
    Path('logs').mkdir(exist_ok=True)
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    for seed in seeds:
        for budget in budgets:
            run_cell(a.phase, budget, seed, spec, a_winners, data, root, manifest)


if __name__ == '__main__':
    main()
