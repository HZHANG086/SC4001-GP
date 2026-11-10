"""CPU tests exercise actual backward/optimizer paths and reject incomplete returns."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from sc4001_c.common import dump,write_csv,read_csv,BUDGETS,SEEDS,DOMAINS
from sc4001_c.collect_cka import collect,METHODS,COMPARISONS
from sc4001_c.benchmark import run_probe,PROFILES


class CostKernelTests(unittest.TestCase):
    def test_all_methods_and_teacher_profiles_backward(self):
        import torch
        from transformers import RobertaConfig,RobertaModel
        from sc4001_a.model import LinearRoberta
        torch.set_num_threads(1)
        config=RobertaConfig(vocab_size=20,hidden_size=8,num_hidden_layers=3,num_attention_heads=2,
            intermediate_size=16,max_position_embeddings=24,pad_token_id=1)
        for method,profile in PROFILES:
            with self.subTest(method=method,profile=profile):
                torch.manual_seed(3)
                model=LinearRoberta(config)
                frozen_before=model.roberta.encoder.layer[0].output.dense.weight.detach().clone()
                head_before=model.classifier.weight.detach().clone()
                teacher=RobertaModel(config,add_pooling_layer=False) if method=='M6' or profile=='diagnostic_teacher' else None
                inputs={'input_ids':torch.randint(3,20,(4,8)),'attention_mask':torch.ones(4,8,dtype=torch.long)}
                result=run_probe(model,inputs,torch.tensor([0,1,0,1]),method,teacher,lam=.01,warmup=1,steps=2,accumulation=2)
                self.assertEqual(len(result['training_seconds_per_optimizer_step']),2)
                self.assertGreater(result['optimizer_state_tensors'],0)
                self.assertTrue(np.isfinite(result['training_loss_diagnostic_only']).all())
                self.assertFalse(torch.equal(head_before,model.classifier.weight))
                self.assertEqual(result['teacher_used_in_loss'],method=='M6')
                self.assertEqual(result['teacher_present_training'],teacher is not None)
                if teacher is not None:
                    self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))
                if method in ('M1','M2'):
                    self.assertTrue(torch.equal(frozen_before,model.roberta.encoder.layer[0].output.dense.weight))
                else:
                    self.assertFalse(torch.equal(frozen_before,model.roberta.encoder.layer[0].output.dense.weight))
                self.assertIsNone(result['training_peak']['allocated_bytes']) # CPU is never claimed as CUDA


class CollectionTests(unittest.TestCase):
    def fixture(self,root):
        for s in SEEDS:
            for b in BUDGETS:
                f=root/f'b{b}-s{s}'
                dump(f/'manifest.json',{'fixture':False,'models':[{'method':m,'seed':s,'budget':b} for m in METHODS],
                    **{k:'test-only' for k in ('version','samples','base_model','base_revision','batch','max_length',
                          'script_sha256','common_sha256','model_code_sha256','versions')}})
                dump(f/'run.json',{'fixture':False,'models_completed':4,'job_id':'fixture','gpu':'fixture'})
                (f/'COMPLETE').write_text('C-CKA-v2 completed\n')
                write_csv(f/'cka.csv',[{'method':m,'reference':ref,'budget':b,'seed':s,'domain':d,
                    'pooling':p,'layer':l,'n_examples':256,'linear_cka':.8,'debiased_linear_cka':.7,
                    'shuffled_linear_cka':.1,'cosine_distance':.01,'status':'ok'}
                    for m,ref in COMPARISONS for d in DOMAINS for p in ('cls','mean') for l in range(13)])
                write_csv(f/'prediction_replay.csv',[{'method':m,'budget':b,'seed':s,'domain':d,
                    'max_probability_difference':0,'n':256} for m in METHODS for d in DOMAINS])

    def test_full_grid_and_corrupt_replay_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'input';out=Path(t)/'out';self.fixture(root);collect(root,out)
            self.assertEqual(len(read_csv(out/'cka_all_seeds.csv')),8190)
            self.assertEqual(len(read_csv(out/'prediction_replay_all.csv')),180)
            file=root/'b400-s13/prediction_replay.csv';rows=read_csv(file)
            rows[0]['max_probability_difference']='nan';write_csv(file,rows)
            with self.assertRaises(ValueError): collect(root,out)

    def test_missing_comparison_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'input';self.fixture(root)
            file=root/'b400-s13/cka.csv';rows=read_csv(file);write_csv(file,rows[:-1])
            with self.assertRaises(ValueError): collect(root,Path(t)/'out')

    def test_synthetic_result_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'input';self.fixture(root)
            (root/'b400-s13/COMPLETE').write_text('SYNTHETIC_TEST_ONLY\n')
            with self.assertRaises(ValueError): collect(root,Path(t)/'out')



class QueueScriptTests(unittest.TestCase):
    def setup_mock(self,root):
        import os
        b=root/'member-b';(b/'scripts').mkdir(parents=True);(root/'member-c/src').mkdir(parents=True)
        (b/'scripts/activate.sh').write_text('export PYTHONPATH=""\n')
        bin_dir=root/'bin';bin_dir.mkdir()
        for name,body in {
            'python':'echo "python $*" >> "$CALLS"; if [ "${FAIL_PYTHON:-0}" = 1 ]; then exit 9; fi',
            'sbatch':'echo "sbatch $*" >> "$CALLS"; echo 456'
        }.items():
            f=bin_dir/name;f.write_text('#!/bin/bash\n'+body+'\n');f.chmod(0o755)
        env={**os.environ,'PATH':str(bin_dir)+':'+os.environ['PATH'],'SLURM_SUBMIT_DIR':str(b),
             'SLURM_JOB_ID':'123','CALLS':str(root/'calls')}
        return b,env

    def test_success_submits_exactly_one_successor(self):
        import subprocess
        script=Path(__file__).resolve().parents[1]/'scripts/run_bundle.sbatch'
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);b,env=self.setup_mock(root)
            done=subprocess.run(['bash',str(script),'0'],cwd=b,env=env,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            calls=(root/'calls').read_text()
            self.assertEqual(sum(line.startswith('sbatch ') for line in calls.splitlines()),1)
            self.assertIn('--dependency=afterok:123',calls)
            self.assertIn('run_bundle.sbatch 1',calls)

    def test_failure_never_submits_next_job(self):
        import subprocess
        script=Path(__file__).resolve().parents[1]/'scripts/run_bundle.sbatch'
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);b,env=self.setup_mock(root);env['FAIL_PYTHON']='1'
            done=subprocess.run(['bash',str(script),'0'],cwd=b,env=env,capture_output=True,text=True)
            self.assertNotEqual(done.returncode,0)
            self.assertNotIn('sbatch ',(root/'calls').read_text())

    def test_last_stage_runs_nine_fresh_probes_then_packages(self):
        import subprocess
        script=Path(__file__).resolve().parents[1]/'scripts/run_bundle.sbatch'
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);b,env=self.setup_mock(root)
            done=subprocess.run(['bash',str(script),'17'],cwd=b,env=env,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            calls=(root/'calls').read_text()
            self.assertEqual(calls.count('sc4001_c.benchmark'),9)
            self.assertIn('sc4001_c.handoff package',calls)
            self.assertNotIn('sbatch ',calls)


class CostCollectionTests(unittest.TestCase):
    def fixture(self,root):
        for repeat in (1,2,3):
            for method,profile in PROFILES:
                folder=root/f'r{repeat}-{method}-{profile}'
                dump(folder/'manifest.json',{'method':method,'profile':profile,'repeat':repeat,'smoke':False,
                    'batch':8,'length':512,'accumulation':4,'warmup_steps':2,'measured_steps':5,
                    'lambda':.01 if method=='M6' else None})
                dump(folder/'result.json',{'fixture':False,'gpu':'fixture-only','optimizer_state_tensors':10,
                    'training_seconds_per_optimizer_step':[1.]*5,'teacher_used_in_loss':method=='M6',
                    'teacher_present_training':method=='M6' or profile=='diagnostic_teacher',
                    'training_peak':{'allocated_bytes':100,'reserved_bytes':200},
                    'warmup_peak':{'allocated_bytes':110,'reserved_bytes':220},
                    'inference_peak':{'allocated_bytes':50},'training_examples_per_second':32.,
                    'inference_examples_per_second':64.,'trainable_parameters':20,'job_id':'fixture-only'})
                (folder/'COMPLETE').write_text('C-cost-v2 completed\n')

    def test_all_processes_and_teacher_tamper(self):
        from sc4001_c.handoff import collect_cost
        from sc4001_c.common import load
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'input';out=Path(t)/'out';self.fixture(root);collect_cost(root,out)
            self.assertEqual(len(read_csv(out/'cost_all_repeats.csv')),27)
            self.assertEqual(len(read_csv(out/'cost_summary.csv')),9)
            file=root/'r1-M6-clean/result.json';data=load(file);data['teacher_used_in_loss']=False;dump(file,data)
            with self.assertRaises(ValueError): collect_cost(root,out)

    def test_gpu_mismatch_rejected(self):
        from sc4001_c.handoff import collect_cost
        from sc4001_c.common import load
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'input';self.fixture(root)
            file=root/'r2-M3-clean/result.json';data=load(file);data['gpu']='other-GPU';dump(file,data)
            with self.assertRaises(ValueError): collect_cost(root,Path(t)/'out')


if __name__=='__main__': unittest.main()
