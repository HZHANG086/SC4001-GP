"""Offline end-to-end test with saved tiny LinearRoberta models and replay checks."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from sc4001_c.common import dump,digest,read_csv
from sc4001_c.cka import main,extract


class CKAIntegration(unittest.TestCase):
    def test_saved_models_replay_and_resumption(self):
        import torch
        from datasets import Dataset
        from transformers import RobertaConfig,PreTrainedTokenizerFast
        from tokenizers import Tokenizer,models,pre_tokenizers,processors
        from sc4001_a.model import LinearRoberta
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data'; data.mkdir(); archive=root/'archive'
            special=['<s>','<pad>','</s>','<unk>']; words=['good','bad','film','plot','very','not','the','is']
            core=Tokenizer(models.WordLevel(vocab={t:i for i,t in enumerate(special+words)},unk_token='<unk>'))
            core.pre_tokenizer=pre_tokenizers.Whitespace()
            core.post_processor=processors.TemplateProcessing(single='<s> $A </s>',special_tokens=[('<s>',0),('</s>',2)])
            tok=PreTrainedTokenizerFast(tokenizer_object=core,bos_token='<s>',eos_token='</s>',pad_token='<pad>',
                    unk_token='<unk>',model_input_names=['input_ids','attention_mask'])
            cfg=RobertaConfig(vocab_size=12,hidden_size=16,num_hidden_layers=2,num_attention_heads=2,
                             intermediate_size=32,max_position_embeddings=40,pad_token_id=1,bos_token_id=0,eos_token_id=2)
            torch.manual_seed(9); base=LinearRoberta(cfg).eval(); ref=root/'reference'; base.save_pretrained(ref); tok.save_pretrained(ref)
            data_rows={}
            rng=np.random.default_rng(1)
            for domain in ('imdb','yelp','amazon'):
                rs=[{'example_id':f'{domain}:test:{i}','label':i%2,
                     'text':' '.join(rng.choice(words,size=3+i%7))} for i in range(24)]
                data_rows[domain]=rs; Dataset.from_list(rs).save_to_disk(str(data/f'{domain}_test'))
            dump(data/'sources.json',{'model_id':'synthetic','model_revision':'synthetic'})
            dump(data/'manifest_hashes.json',{'sources.json':digest(data/'sources.json')})
            (data/'READY').write_text('fixture')
            entries=[]
            for method in ('M3','M6'):
                run=Path('runs/B-v1')/f'{method}-b400-s13-test'; folder=root/run/'best'
                model=LinearRoberta.from_pretrained(ref).eval()
                if method=='M6':
                    with torch.no_grad(): model.roberta.encoder.layer[1].output.dense.weight.add_(.003*torch.randn_like(model.roberta.encoder.layer[1].output.dense.weight))
                model.save_pretrained(folder); tok.save_pretrained(folder)
                dump(folder.parent/'run_config.json',{'method':method,'budget':400,'seed':13})
                dump(archive/'runs/B-v1'/f'selection-{method}-b400-s13.json',{'winner':str(run)})
                for domain,rs in data_rows.items():
                    _,prob=extract(model.roberta,tok,rs,'cpu',32,4,model.classifier)
                    target=archive/run/'test'/f'{domain}_predictions.jsonl'; target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_text('\n'.join(json.dumps({'example_id':r['example_id'],'label':r['label'],
                        'p_positive':float(q),'prediction':int(q>.5)}) for r,q in zip(rs,prob)))
                entries.append({'method':method,'budget':400,'seed':13,'model_dir':str(run/'best'),'cka_subset':'yes'})
            manifest=root/'models.tsv'
            with manifest.open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(entries[0]),delimiter='\t');w.writeheader();w.writerows(entries)
            args=['--models',str(manifest),'--model-base',str(root),'--archive',str(archive),
                  '--data',str(data),'--data-reference',str(data),'--out',str(root/'out'),
                  '--methods','M3','M6','--seeds','13','--budgets','400','--samples','8','--batch','4',
                  '--cpu-fixture','--reference-fixture',str(ref)]
            main(args)
            records=read_csv(root/'out/cka.csv')
            self.assertEqual(len(records),54) # 3 comparisons * 3 domains * 2 pools * 3 layers
            selected=[r for r in records if r['method']=='M3' and r['status']=='ok']
            self.assertTrue(all(abs(float(r['linear_cka'])-1)<1e-8 for r in selected))
            self.assertTrue((root/'out/COMPLETE').exists())
            before=digest(root/'out/cka.csv'); main(args); self.assertEqual(before,digest(root/'out/cka.csv'))
            # Changing a checkpoint must fail rather than reuse stale completed output.
            weight=root/entries[0]['model_dir']/'model.safetensors'
            weight.write_bytes(weight.read_bytes()+b'changed')
            with self.assertRaises(ValueError): main(args)


if __name__=='__main__': unittest.main()
