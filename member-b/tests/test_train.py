"""End-to-end CPU tests of sc4001_b.train on a tiny random model and fake data-v1."""
import json
import tempfile
import unittest
from pathlib import Path

import fixtures
import torch
from safetensors.torch import load_file

from sc4001_b.train import main


class TrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.model = fixtures.make_model(root / 'model')
        cls.data = fixtures.make_data(root / 'data')
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_method(self, name, *extra, lr='1e-3'):
        out = self.root / 'runs' / name
        if not (out / 'COMPLETE').exists():
            main(['--data', str(self.data), '--out', str(out), '--model-path', str(self.model),
                  '--budget', '400', '--seed', '13', '--lr', lr, '--length', '256',
                  '--epochs', '2', '--cpu', *extra])
        return out

    @staticmethod
    def weights(run):
        return load_file(str(run / 'best' / 'model.safetensors'))

    def test_m6_with_zero_lambda_reproduces_m3_exactly(self):
        m3 = self.weights(self.run_method('M3', '--method', 'M3'))
        m6 = self.weights(self.run_method('M6-lam0', '--method', 'M6', '--lam', '0'))
        self.assertEqual(m3.keys(), m6.keys())
        for name in m3:
            torch.testing.assert_close(m3[name], m6[name], rtol=0, atol=0, msg=name)

    def test_regularisers_are_logged_and_change_training(self):
        m3 = self.weights(self.run_method('M3', '--method', 'M3'))
        for method, logged in (('M6', 'train_feature_preservation'), ('L2SP', 'train_l2sp')):
            run = self.run_method(f'{method}-lam1', '--method', method, '--lam', '1')
            history = json.loads((run / 'history.json').read_text())
            self.assertTrue(any(logged in entry for entry in history), method)
            weights = self.weights(run)
            self.assertTrue(any(not torch.equal(m3[n], weights[n]) for n in m3), method)
            val = json.loads((run / 'validation.json').read_text())
            self.assertGreaterEqual(val['val_feature_distance'], 0.0)

    def test_m4_stage1_trains_only_the_head_and_stage2_starts_from_it(self):
        initial = load_file(str(self.model / 'model.safetensors'))
        stage1 = self.run_method('M4S1', '--method', 'M4S1', lr='1e-2')
        head_only = self.weights(stage1)
        for name, value in initial.items():
            if name.startswith('roberta.'):
                torch.testing.assert_close(head_only[name], value, rtol=0, atol=0, msg=name)
        self.assertFalse(torch.equal(head_only['classifier.weight'], initial['classifier.weight']))
        stage2 = self.run_method('M4', '--method', 'M4', '--stage1', str(stage1))
        cfg = json.loads((stage2 / 'run_config.json').read_text())
        self.assertEqual(cfg['stage1_record']['lr'], 1e-2)
        params = json.loads((stage2 / 'parameters.json').read_text())
        self.assertEqual(params['trainable'], params['total'])
        s1_params = json.loads((stage1 / 'parameters.json').read_text())
        self.assertEqual(s1_params['trainable'], 16 * 2 + 2)

    def test_run_folder_matches_member_a_evaluation_contract(self):
        run = self.run_method('M3', '--method', 'M3')
        for name in ('COMPLETE', 'run_config.json', 'parameters.json', 'validation.json',
                     'history.json', 'best/model.safetensors', 'source/train.py',
                     'source/objectives.py', 'source/a_model.py'):
            self.assertTrue((run / name).exists(), name)
        cfg = json.loads((run / 'run_config.json').read_text())
        for key in ('data_manifest_sha256', 'smoke', 'stage', 'length', 'split_sha256', 'model_revision'):
            self.assertIn(key, cfg)
        self.assertTrue(cfg['smoke'])            # test-model runs can never become formal results
        self.assertEqual(cfg['stage'], 'smoke')
        val = json.loads((run / 'validation.json').read_text())
        for key in ('eval_accuracy', 'train_wall_seconds_this_invocation', 'best_checkpoint'):
            self.assertIn(key, val)

    def test_argument_rules(self):
        base = ['--data', str(self.data), '--model-path', str(self.model), '--lr', '1e-3', '--cpu',
                '--length', '256']
        bad = [['--method', 'M6'], ['--method', 'M3', '--lam', '1'], ['--method', 'M4'],
               ['--method', 'M6', '--lam', '-1'], ['--method', 'M3', '--stage1', 'x']]
        for i, extra in enumerate(bad):
            with self.assertRaises(SystemExit, msg=extra):
                main(base + ['--out', str(self.root / f'bad{i}')] + extra)
        stage1 = self.run_method('M4S1', '--method', 'M4S1', lr='1e-2')
        with self.assertRaises(SystemExit):
            main(base + ['--out', str(self.root / 'bad-stage1'), '--method', 'M4', '--stage1',
                         str(stage1), '--length', '512'])


if __name__ == '__main__':
    unittest.main()
