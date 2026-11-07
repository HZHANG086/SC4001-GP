"""Selection, dependency, evaluation-order and cleanup logic of sc4001_b.sweep, plus aggregation.

Training and test evaluation are replaced by fakes that write Member A-format run folders,
so this runs on CPU in seconds. Real end-to-end runs are covered by test_train and the
cluster smoke/pilot jobs.
"""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import fixtures
from sc4001_b import aggregate, sweep

SPEC = json.loads((fixtures.ROOT / 'member-b/protocol/B-v1.json').read_text())
A_PER_SEED = fixtures.ROOT / 'member-a/formal-results/results/A-v1/per_seed.csv'
A_WINNERS = json.loads((fixtures.ROOT / 'member-a/formal-results/results/A-v1/winners.json').read_text())['winners']
BEST = {'M3': ('lr', 2e-05), 'M4': ('lr', 1e-05), 'M6': ('lam', 1.0), 'L2SP': ('lam', 0.1)}


def arg(command, flag):
    return command[command.index(flag) + 1] if flag in command else None


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'runs'
        self.root.mkdir()
        self.trained, self.evaluated = [], []
        self._orig = sweep.ensure_trained, sweep.evaluate
        sweep.ensure_trained, sweep.evaluate = self.fake_train, self.fake_evaluate

    def tearDown(self):
        sweep.ensure_trained, sweep.evaluate = self._orig
        self.tmp.cleanup()

    def fake_train(self, command, run, root, expected):
        method, lr, lam = arg(command, '--method'), float(arg(command, '--lr')), arg(command, '--lam')
        lam = None if lam is None else float(lam)
        if not (run / 'COMPLETE').exists():
            self.trained.append(run.name)
            (run / 'best').mkdir(parents=True)
            (run / 'best/model.safetensors').write_text('weights')
            (run / 'checkpoints').mkdir()
            key, value = BEST.get(method, ('lr', lr))
            acc = 0.9 if (lr if key == 'lr' else lam) == value else 0.8
            cfg = {'method': method, 'budget': expected['budget'], 'seed': expected['seed'], 'lr': lr,
                   'lam': lam, 'stage1': arg(command, '--stage1')}
            (run / 'run_config.json').write_text(json.dumps(cfg))
            (run / 'validation.json').write_text(json.dumps({
                'eval_accuracy': acc, 'train_wall_seconds_this_invocation': 60.0,
                'peak_cuda_allocated_bytes': 2**30, 'val_feature_distance': 0.0 if method == 'M4S1' else lam or 0.05}))
            (run / 'parameters.json').write_text(json.dumps({'trainable': 1 if method == 'M4S1' else 100}))
            (run / 'COMPLETE').write_text('x')
        self.assertEqual(expected['lr'], lr)
        self.assertEqual(expected.get('lam'), lam)
        return json.loads((run / 'validation.json').read_text())

    def fake_evaluate(self, run, data):
        if (run / 'test/metrics.json').exists():                             # as sweep.evaluate
            return
        assert (run / 'best').exists(), f'evaluated after pruning: {run}'
        sel = list(self.root.glob('selection-*'))
        assert sel, 'test access before any selection file'
        self.evaluated.append(run.name)
        (run / 'test').mkdir()
        (run / 'test/metrics.json').write_text(json.dumps({
            'imdb': {'accuracy': 0.9}, 'yelp': {'accuracy': 0.88}, 'amazon': {'accuracy': 0.86},
            'ood_mean_accuracy': 0.87}))

    def cell(self, phase):
        sweep.run_cell(phase, 400, 13, SPEC, A_WINNERS, Path('data'), self.root, 'manifest-sha')

    def test_m6_requires_m3_selection(self):
        with self.assertRaises(SystemExit):
            self.cell('M6')

    def test_full_cell_flow(self):
        self.cell('M3')
        sel = json.loads((self.root / 'selection-M3-b400-s13.json').read_text())
        self.assertEqual(sel['lr'], 2e-05)
        self.assertEqual(self.evaluated, ['M3-b400-s13-lr2e-05'])           # winner only
        for name in ('M3-b400-s13-lr1e-05', 'M3-b400-s13-lr3e-05'):
            self.assertFalse((self.root / name / 'best').exists())
            self.assertTrue((self.root / name / 'WEIGHTS_PRUNED').exists())
        self.assertTrue((self.root / 'M3-b400-s13-lr2e-05/best').exists())
        self.assertFalse(any(self.root.glob('*/checkpoints')))

        self.cell('M6')                                                      # inherits M3 LR 2e-05
        m6 = [sweep.run_name('M6', 400, 13, lam=lam) for lam in SPEC['lambdas']['M6']]
        self.assertEqual(self.evaluated[1:], m6)                             # all candidates (ablation)
        self.assertTrue((self.root / 'M6-b400-s13-lam1/best').exists())
        for name in m6:
            self.assertEqual((self.root / name / 'best').exists(), name == 'M6-b400-s13-lam1')

        self.cell('L2SP')
        self.cell('M4')
        lr1 = sweep.a_m1_lr(A_WINNERS, 400, 13)
        stage1 = self.root / sweep.run_name('M4S1', 400, 13, lr=lr1)
        self.assertEqual(stage1.name, 'M4S1-b400-s13-lr1e-03')
        self.assertTrue((stage1 / 'WEIGHTS_PRUNED').exists())
        m4 = json.loads((self.root / 'selection-M4-b400-s13.json').read_text())
        self.assertEqual(m4['stage1_run'], str(stage1))

        before = list(self.trained)
        for phase in ('M3', 'M4', 'M6', 'L2SP'):                             # rerun: idempotent
            self.cell(phase)
        self.assertEqual(self.trained, before)

        out = Path(self.tmp.name) / 'results'
        aggregate.main(['--protocol', str(fixtures.ROOT / 'member-b/protocol/B-v1.json'),
                        '--a-per-seed', str(A_PER_SEED), '--runs', str(self.root), '--out', str(out)])
        with (out / 'per_seed.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(sum(r['source'] == 'A-v1' for r in rows), 30)
        b = {r['method']: r for r in rows if r['source'] == 'B-v1'}
        self.assertEqual(set(b), {'M3', 'M4', 'M6', 'L2SP'})
        self.assertEqual(float(b['M4']['stage1_train_seconds']), 60.0)
        self.assertEqual(float(b['M4']['search_train_seconds_recorded']), 4 * 60.0)
        with (out / 'ablation.csv').open() as stream:
            ablation = list(csv.DictReader(stream))
        lams = sorted(float(r['lam']) for r in ablation if r['method'] == 'M6')
        self.assertEqual(lams, [0.0] + SPEC['lambdas']['M6'])
        self.assertFalse((out / 'COMPLETE').exists())
        self.assertIn('INCOMPLETE', (out / 'summary.md').read_text())

    def test_data_check_against_member_a(self):
        data = Path(self.tmp.name) / 'data'
        data.mkdir()
        reference = fixtures.ROOT / 'member-a/remote-records/data-v1/manifest_hashes.json'
        manifest = json.loads(reference.read_text())
        manifest['imdb_train/state.json'] = 'differs-by-cache-path'          # tolerated
        (data / 'manifest_hashes.json').write_text(json.dumps(manifest))
        sweep.check_data_matches_a(data, reference)
        manifest['splits/b400_s13.json'] = 'different-split'
        (data / 'manifest_hashes.json').write_text(json.dumps(manifest))
        with self.assertRaises(SystemExit):
            sweep.check_data_matches_a(data, reference)

    def test_frozen_selection_is_never_silently_changed(self):
        self.cell('M3')
        path = self.root / 'selection-M3-b400-s13.json'
        sel = json.loads(path.read_text())
        sel['winner'] = str(self.root / 'M3-b400-s13-lr1e-05')
        path.write_text(json.dumps(sel))
        with self.assertRaises(SystemExit):
            self.cell('M3')


if __name__ == '__main__':
    unittest.main()
