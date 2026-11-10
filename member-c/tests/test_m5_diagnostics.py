import copy
import json
import tempfile
import unittest
from pathlib import Path
import torch
from transformers import RobertaConfig
from sc4001_c.lora import attach
from sc4001_c.m5_benchmark import LoRAProbe
from sc4001_c.benchmark import run_probe
from sc4001_c.m5_audit import audit_predictions


class DiagnosticTests(unittest.TestCase):
    def test_collector_rejects_mixed_hardware_and_changed_inputs(self):
        # Entirely synthetic, temporary fixture: never retained as experiment data.
        import itertools
        import pandas as pd
        from sc4001_c.common import dump,load
        from sc4001_c.m5_collect import main as collect
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);dump(root/'M5-audit/audit.json',{'status':'PASS'})
            cka=[];replay=[]
            for b,s,d in itertools.product((400,1600,6400),(13,21,42,87,100),('imdb','yelp','amazon')):
                replay.append({'budget':b,'seed':s,'domain':d,'n':256,'max_probability_difference':0.})
                for pool,l in itertools.product(('cls','mean'),range(13)):
                    value=float('nan') if pool=='cls' and l==0 else .5
                    cka.append({'method':'M5','reference':'pretrained','budget':b,'seed':s,'domain':d,
                        'pooling':pool,'layer':l,'n_examples':256,'linear_cka':value,
                        'debiased_linear_cka':value,'shuffled_linear_cka':value})
            folder=root/'M5-CKA';folder.mkdir()
            pd.DataFrame(cka).to_csv(folder/'cka.csv',index=False)
            pd.DataFrame(replay).to_csv(folder/'prediction_replay.csv',index=False)
            (folder/'COMPLETE').write_text('M5-CKA-v1 completed\n')
            shared={'base_model':'SYNTHETIC','base_revision':'fixture','split_sha256':'fixture',
                'example_ids':['fixture'],'text_sha256':['fixture'],'token_ids_sha256':'fixture',
                'length':512,'batch':8,'accumulation':4,'precision':'FP32; TF32 disabled',
                'warmup_steps':2,'measured_steps':5,'learning_rate':2e-5}
            for m,r in itertools.product(('M1','M2','M3','M4','M5','M6','L2SP'),(1,2,3)):
                folder=root/'Ada-cost'/m/'clean'/f'repeat-{r}'
                dump(folder/'manifest.json',{**shared,'method':m,'repeat':r,'profile':'clean'})
                dump(folder/'result.json',{'gpu':'RTX 6000 Ada','training_seconds_per_optimizer_step':[1.]*5,
                    'training_peak':{'allocated_bytes':1024},'trainable_parameters':2})
                (folder/'COMPLETE').write_text('SYNTHETIC LOCAL TEST')
            collect(['--results',str(root)])
            self.assertEqual(load(root/'M5-diagnostics/audit.json')['cost_processes'],21)
            target=root/'Ada-cost/M5/clean/repeat-3'
            original=load(target/'result.json');dump(target/'result.json',{**original,'gpu':'V100'})
            with self.assertRaisesRegex(ValueError,'Mixed cost hardware'):collect(['--results',str(root)])
            dump(target/'result.json',original)
            original=load(target/'manifest.json');dump(target/'manifest.json',{**original,'token_ids_sha256':'different'})
            with self.assertRaisesRegex(ValueError,'Cost input/protocol mismatch'):collect(['--results',str(root)])

    def test_shared_benchmark_never_unfreezes_lora_base(self):
        torch.set_num_threads(1);torch.manual_seed(17)
        cfg=RobertaConfig(vocab_size=20,hidden_size=16,num_hidden_layers=2,num_attention_heads=2,
            intermediate_size=32,max_position_embeddings=40)
        model=LoRAProbe(cfg);attach(model,rank=2,alpha=4)
        frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        inputs={'input_ids':torch.randint(3,20,(4,8)),'attention_mask':torch.ones(4,8,dtype=torch.long)}
        result=run_probe(model,inputs,torch.tensor([0,1,0,1]),'M5',warmup=1,steps=2,accumulation=2)
        self.assertEqual(result['trainable_parameters'],trainable)
        for n,p in model.named_parameters():
            if n in frozen:
                self.assertFalse(p.requires_grad);self.assertIsNone(p.grad)
                torch.testing.assert_close(p,frozen[n],atol=0,rtol=0)
        next(model.roberta.parameters()).requires_grad=True
        with self.assertRaisesRegex(ValueError,'unexpectedly trainable'):model.configure('full')

    def test_prediction_audit_rejects_duplicate_label_and_probability_corruption(self):
        rows=[{'example_id':'a','label':0,'prediction':0,'p_positive':.1},
              {'example_id':'b','label':1,'prediction':1,'p_positive':.8}]
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'predictions.jsonl'
            def save(x):path.write_text(''.join(json.dumps(r)+'\n' for r in x))
            save(rows);metrics,n=audit_predictions(path,{'a':0,'b':1})
            self.assertEqual(n,2);self.assertEqual(metrics['accuracy'],1.)
            save(rows+[rows[0]])
            with self.assertRaises(ValueError):audit_predictions(path,{'a':0,'b':1})
            bad=copy.deepcopy(rows);bad[0]['label']=1;save(bad)
            with self.assertRaises(ValueError):audit_predictions(path,{'a':0,'b':1})
            bad=copy.deepcopy(rows);bad[0]['p_positive']=float('nan');save(bad)
            with self.assertRaises(ValueError):audit_predictions(path,{'a':0,'b':1})
            bad=copy.deepcopy(rows);bad[0]['prediction']=1;save(bad)
            with self.assertRaises(ValueError):audit_predictions(path,{'a':0,'b':1})


if __name__=='__main__':unittest.main()
