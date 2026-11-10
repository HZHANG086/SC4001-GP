import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from transformers import RobertaConfig,Trainer,TrainerCallback
from sc4001_a.model import LinearRoberta
from sc4001_c.lora import attach,merge,LoRALinear
from sc4001_c.lora_train import main
from sc4001_c.common import load


class LoRATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(13)
        self.config=RobertaConfig(vocab_size=20,hidden_size=16,num_hidden_layers=2,num_attention_heads=2,intermediate_size=32,max_position_embeddings=40)

    def test_identity_freezing_gradient_merge_and_plain_reload(self):
        model=LinearRoberta(self.config).eval();reference=copy.deepcopy(model)
        x={'input_ids':torch.randint(3,20,(4,8)),'attention_mask':torch.ones(4,8,dtype=torch.long)}
        attach(model,rank=2,alpha=4)
        model.eval();torch.testing.assert_close(model(**x).logits,reference(**x).logits,rtol=0,atol=0)
        trainable=[n for n,p in model.named_parameters() if p.requires_grad]
        self.assertTrue(all('lora_' in n or n.startswith('classifier.') for n in trainable))
        optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        model.train()
        for _ in range(2):
            optim.zero_grad();model(**x,labels=torch.tensor([0,1,0,1])).loss.backward();optim.step()
        for i,layer in enumerate(model.roberta.encoder.layer):
            for name in ('query','value'):
                module=getattr(layer.attention.self,name)
                torch.testing.assert_close(module.base.weight,getattr(reference.roberta.encoder.layer[i].attention.self,name).weight,rtol=0,atol=0)
                self.assertIsNone(module.base.weight.grad);self.assertGreater(module.lora_A.grad.abs().sum(),0)
        model.eval();before=model(**x).logits.detach();merge(model)
        torch.testing.assert_close(before,model(**x).logits,rtol=1e-5,atol=1e-6)
        self.assertFalse(any(isinstance(m,LoRALinear) for m in model.modules()))
        with tempfile.TemporaryDirectory() as t:
            model.save_pretrained(t);reloaded=LinearRoberta.from_pretrained(t).eval()
            torch.testing.assert_close(model(**x).logits,reloaded(**x).logits,rtol=0,atol=0)

    def test_double_attachment_and_merge_rejected(self):
        model=LinearRoberta(self.config);attach(model,rank=2,alpha=4)
        with self.assertRaises(ValueError):attach(model)
        merge(model)
        with self.assertRaises(ValueError):merge(model)

    def test_trainer_export_and_interrupted_resume_match(self):
        # Reuse A/B's offline synthetic fixture builder, never any real test data.
        fixture_dir=Path(__file__).resolve().parents[2]/'member-b/tests'
        sys.path.insert(0,str(fixture_dir))
        from fixtures import make_model,make_data
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);model=make_model(root/'base');data=make_data(root/'data')
            protocol=Path(__file__).resolve().parents[1]/'protocol/M5-v1.json'
            common=['--data',str(data),'--data-reference',str(data),'--protocol',str(protocol),
                '--budget','400','--seed','13','--lr','0.0003','--smoke','--fixture-model',str(model)]
            main(common+['--out',str(root/'continuous')])
            class StopAfterSaved(TrainerCallback):
                def on_save(self,args,state,control,**kwargs):raise RuntimeError('SIMULATED_INTERRUPT_AFTER_CHECKPOINT')
            class Interrupted(Trainer):
                def __init__(self,*args,**kwargs):
                    super().__init__(*args,**kwargs);self.add_callback(StopAfterSaved())
            with patch('transformers.Trainer',Interrupted):
                with self.assertRaisesRegex(RuntimeError,'SIMULATED_INTERRUPT'):
                    main(common+['--out',str(root/'resumed')])
            self.assertFalse((root/'resumed/COMPLETE').exists())
            main(common+['--out',str(root/'resumed'),'--resume'])
            from safetensors.torch import load_file
            a=load_file(root/'continuous/best/model.safetensors');b=load_file(root/'resumed/best/model.safetensors')
            for name in a:torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
            self.assertEqual((root/'resumed/COMPLETE').read_text().strip(),'SYNTHETIC_TEST_ONLY')
            self.assertTrue(load(root/'resumed/run_config.json')['smoke'])
            self.assertFalse((root/'resumed/test').exists())
            self.assertLess(load(root/'resumed/validation.json')['merge_max_abs_logit_difference'],1e-5)


if __name__=='__main__':unittest.main()

class SelectionTests(unittest.TestCase):
    def test_source_only_tie_order_and_immutable_decision(self):
        from sc4001_c.lora_sweep import select_cell,run_name
        from sc4001_c.common import dump
        spec={'learning_rates':[.0001,.0003,.001],'lora':{'rank':8,'alpha':16,'dropout':0.}}
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            for lr,score in zip(spec['learning_rates'],[.7,.8,.8]):
                run=root/run_name(400,13,lr)
                dump(run/'run_config.json',{'method':'M5','budget':400,'seed':13,'lr':lr,'epochs':10,
                    'length':512,'batch':8,'accumulation':4,'stage':'formal','smoke':False,'fixture':False,'lora':spec['lora']})
                dump(run/'validation.json',{'eval_accuracy':score})
                dump(run/'history.json',[{'epoch':1,'eval_accuracy':score},{'epoch':10,'eval_accuracy':score-.01}])
                dump(run/'parameters.json',{'trainable':296450})
                (run/'COMPLETE').write_text('M5 source-validation training complete; no tests evaluated\n')
                # If selection reads target results, this invalid file should fail parsing.
                (run/'test').mkdir();(run/'test/metrics.json').write_text('DO NOT READ TEST DATA DURING SELECTION')
            decision=select_cell(root,400,13,spec)
            self.assertEqual(decision['lr'],.0003)
            self.assertEqual(decision,select_cell(root,400,13,spec))
            run=root/run_name(400,13,.0001)
            dump(run/'validation.json',{'eval_accuracy':.9});dump(run/'history.json',[{'epoch':10,'eval_accuracy':.9}])
            with self.assertRaisesRegex(ValueError,'frozen selection'):select_cell(root,400,13,spec)

    def test_prune_guard(self):
        from sc4001_c.lora_sweep import prune
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'runs';root.mkdir();other=Path(t)/'outside';other.mkdir()
            with self.assertRaises(ValueError):prune(other,root)
            self.assertTrue(other.exists())
