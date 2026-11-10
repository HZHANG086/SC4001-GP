"""LoRA training microbenchmark on the same inputs/kernel as C-cost-v2.

The inherited full-tuning configure() must never unfreeze the LoRA backbone.
Inference here is UNMERGED and is not a deployment claim about saved winners.
"""
import argparse
import hashlib
import os
from pathlib import Path
from sc4001_a.model import LinearRoberta
from .lora import attach
from .common import load, dump, digest
from .benchmark import run_probe
from .cka import verify_data


class LoRAProbe(LinearRoberta):
    def configure(self, method):
        if method != 'full': raise ValueError('LoRA probe accepts only the shared full-mode kernel')
        if not any('lora_A' in n for n,_ in self.named_parameters()):
            raise ValueError('Attach frozen-protocol adapters before measuring')
        if any(p.requires_grad for n,p in self.named_parameters() if 'lora_' not in n and not n.startswith('classifier.')):
            raise ValueError('LoRA backbone unexpectedly trainable')
        self.method='full'


def main(argv=None):
    import torch
    import transformers
    from datasets import load_from_disk
    from transformers import AutoTokenizer, set_seed
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('member-c/artifacts/data-v1'))
    p.add_argument('--reference',type=Path,default=Path('member-a/remote-records/data-v1'))
    p.add_argument('--repeat',type=int,choices=(1,2,3),required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
    if not torch.cuda.is_available():raise ValueError('Allocated GPU required')
    verify_data(a.data,a.reference)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    set_seed(20261110+a.repeat)
    sources=load(a.data/'sources.json');split=a.data/'splits/b400_s13.json'
    rows=load_from_disk(str(a.data/'imdb_train')).select(load(split)['train_indices'][:32])
    tok=AutoTokenizer.from_pretrained(sources['model_id'],revision=sources['model_revision'],local_files_only=True)
    encoded=tok(rows['text'],truncation=True,padding='max_length',max_length=512,return_tensors='pt')
    signature={'version':'M5-cost-v1','method':'M5','profile':'clean','repeat':a.repeat,
        'base_model':sources['model_id'],'base_revision':sources['model_revision'],'split_sha256':digest(split),
        'example_ids':rows['example_id'],'text_sha256':[hashlib.sha256(t.encode()).hexdigest() for t in rows['text']],
        'token_ids_sha256':hashlib.sha256(encoded['input_ids'].numpy().tobytes()).hexdigest(),
        'script_sha256':digest(Path(__file__)),'shared_kernel_sha256':digest(Path(__file__).with_name('benchmark.py')),
        'lora_sha256':digest(Path(__file__).with_name('lora.py')),'lora':{'rank':8,'alpha':16,'dropout':0.},
        'length':512,'batch':8,'accumulation':4,'warmup_steps':2,'measured_steps':5,
        'learning_rate':2e-5,'precision':'FP32; TF32 disabled','versions':{'torch':torch.__version__,'transformers':transformers.__version__},
        'interpretation':'Shared fixed-shape computational probe; benchmark LR is not the training search. Inference is unmerged, not deployment timing.'}
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'manifest.json').exists() and load(a.out/'manifest.json')!=signature:raise ValueError('Existing probe differs')
    dump(a.out/'manifest.json',signature)
    if (a.out/'COMPLETE').exists():return
    model=LoRAProbe.from_pretrained(sources['model_id'],revision=sources['model_revision'],local_files_only=True,use_safetensors=True,num_labels=2)
    attach(model,**signature['lora']);model.float().cuda()
    inputs={k:v.cuda() for k,v in encoded.items() if k in ('input_ids','attention_mask')}
    labels=torch.tensor(rows['label'],device='cuda')
    set_seed(20261110+a.repeat)
    result=run_probe(model,inputs,labels,'M5',warmup=2,steps=5)
    if result['trainable_parameters']!=296450:raise ValueError('Incorrect LoRA trainable count')
    result.update({'method':'M5','gpu':torch.cuda.get_device_name(),'job_id':os.getenv('SLURM_JOB_ID'),
        'saved_model':False,'inference_state':'unmerged adapters; excluded from deployment comparison'})
    dump(a.out/'result.json',result);(a.out/'COMPLETE').write_text('M5-cost-v1 completed\n')


if __name__=='__main__':main()
