"""Recompute the C submission from archived predictions, without Git or weights.

Run after installing requirements-final.txt. This wrapper preserves the frozen
scientific implementations and writes into a new directory, with one log per
stage and an explicit failure record. It does not repeat GPU training/inference.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def compare_outputs(repo, output):
    import pandas as pd

    comparisons = []
    for reference in sorted((repo / 'member-c/results/C-final').rglob('*.csv')):
        relative = reference.relative_to(repo / 'member-c/results/C-final')
        actual = output / 'C-final' / relative
        left, right = pd.read_csv(reference), pd.read_csv(actual)
        # Platform rounding only; strings, shapes, dtypes and missingness match.
        pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=0, atol=1e-12)
        differences = []
        for col in left.select_dtypes(include='number').columns:
            valid = left[col].notna() & right[col].notna()
            delta = (left.loc[valid, col] - right.loc[valid, col]).abs()
            if len(delta) and delta.max() > 0:
                differences.append({'column': col, 'max_absolute_difference': float(delta.max()),
                                    'cells_differing': int((delta > 0).sum())})
        comparisons.append({'table': str(relative), 'rows': len(left),
                            'byte_identical': reference.read_bytes() == actual.read_bytes(),
                            'numeric_roundoff': differences})
    if len(comparisons) != 22:
        raise ValueError('Expected the complete 22-table seven-method synthesis')
    figures = []
    for directory in ('figures', 'figures-grayscale'):
        for extension in ('pdf', 'svg', 'png'):
            paths = sorted((output / directory).glob('*.' + extension))
            if len(paths) != 6 or any(p.stat().st_size < 1000 for p in paths):
                raise ValueError('Missing or empty scientific figure exports')
            figures.extend(str(p.relative_to(output)) for p in paths)
    original = json.loads((output / 'C-v1/reproduction_audit.json').read_text())
    m5 = json.loads((output / 'M5-local-audit/local_audit.json').read_text())
    if original['status'] != 'PASS' or m5['status'] != 'PASS':
        raise ValueError('Both independent prediction audits must pass')
    return {'A_B_prediction_rows': original['prediction_rows'],
            'M5_prediction_rows': m5['prediction_rows'],
            'synthesis_tables': comparisons, 'generated_figure_files': figures,
            'roundoff_policy': 'Synthesis CSVs: absolute 1e-12, no relative tolerance; every difference logged. '
                               'Original seven tables retain their narrower approximate-t-only policy.',
            'figure_boundary': 'Scientific data/styles reproduced. Font availability and PDF/SVG timestamps '
                               'can change rendered bytes; this is not a pixel-identity claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('reproduced/C-complete'))
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output = args.out.resolve()
    if output.exists() or output == repo or any(
            output.is_relative_to(repo / name) for name in ('member-a', 'member-b', 'member-c', '.git')):
        parser.error('Choose a new output directory outside the archived member/Git directories')
    required = ('numpy', 'pandas', 'scipy', 'matplotlib', 'torch', 'transformers', 'datasets', 'accelerate')
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        parser.error('Missing dependencies: ' + ', '.join(missing) +
                     '. Install member-c/requirements-final.txt in the active Python environment first.')
    output.mkdir(parents=True)
    logs = output / 'logs'
    logs.mkdir()
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join(str(repo / p) for p in (
        'member-c/src', 'member-a/protocol/locked_src', 'member-b/protocol/locked_src'))
    environment['MPLCONFIGDIR'] = str(output / 'matplotlib-cache')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        environment[name] = '4'
    environment['TOKENIZERS_PARALLELISM'] = 'false'
    raw = 'member-c/formal-results/M5-return-v1'
    steps = [
        ('tests', ['-m', 'unittest', 'discover', '-s', 'member-c/tests', '-v']),
        ('original-predictions', ['-m', 'sc4001_c.reproduce', '--out', str(output / 'C-v1')]),
        ('m5-predictions', ['-m', 'sc4001_c.m5_import', '--raw', raw, '--out', str(output / 'M5-local-audit')]),
        ('synthesis', ['-m', 'sc4001_c.final_analysis', '--m5', raw + '/member-c/results/M5-v1/per_seed.csv',
                       '--m5-audit', str(output / 'M5-local-audit/local_audit.json'), '--out', str(output / 'C-final')]),
        ('color-figures', ['-m', 'sc4001_c.final_plots', '--data', str(output / 'C-final'), '--out', str(output / 'figures')]),
        ('grayscale-figures', ['-m', 'sc4001_c.final_plots', '--data', str(output / 'C-final'),
                              '--grayscale', '--out', str(output / 'figures-grayscale')]),
    ]
    package = repo / 'PACKAGE_MANIFEST.json'
    record = {'status': 'RUNNING', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'no_git': not (repo / '.git').exists(),
              'package_commit': json.loads(package.read_text())['commit'] if package.exists() else None,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'python': sys.version, 'packages': {p: importlib.metadata.version(p) for p in required},
              'steps': [], 'scope': 'Offline evidence re-analysis and CPU tests; no new GPU experiments or human review.'}
    manifest = output / 'verification.json'
    try:
        for name, arguments in steps:
            print('Running:', name, flush=True)
            with (logs / (name + '.out')).open('w') as stdout, (logs / (name + '.err')).open('w') as stderr:
                result = subprocess.run([sys.executable, *arguments], cwd=repo, env=environment,
                                        stdout=stdout, stderr=stderr)
            record['steps'].append({'name': name, 'exit_code': result.returncode, 'arguments': arguments})
            if result.returncode:
                raise RuntimeError(f'{name} failed; see {logs / (name + ".err")}')
        record.update(compare_outputs(repo, output))
        record['status'] = 'PASS'
    except Exception as error:
        record['status'] = 'FAILED'
        record['error'] = str(error)
        raise
    finally:
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        manifest.write_text(json.dumps(record, indent=2) + '\n')
    print('COMPLETE_REPRODUCTION_PASS: 5,775,000 predictions; 22 synthesis tables; 36 figures.')
    print('Evidence:', manifest)


if __name__ == '__main__':
    main()
