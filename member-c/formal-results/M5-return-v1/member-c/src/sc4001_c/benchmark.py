"""Fresh-process, fixed-shape cost probes; NEVER save or select trained models.

A computational microbenchmark, not a reproduction of epoch-level training.
M4 stage 2 uses the same pretrained initialization as M3 as a compute proxy;
M1 represents the computation of M4 stage 1. Do not add their memory peaks.
"""
import argparse
import gc
import hashlib
import os
import platform
import time
from pathlib import Path
import numpy as np
from .common import load,dump,digest
from .cka import verify_data

METHODS=('M1','M2','M3','M4','M6','L2SP')
PROFILES=tuple((m,'clean') for m in METHODS)+tuple((m,'diagnostic_teacher') for m in ('M3','M4','L2SP'))


def run_probe(model,inputs,labels,method,teacher=None,lam=.01,warmup=2,steps=5,accumulation=4):
    """CPU-compatible kernel for regression tests; formal entry requires CUDA."""
    import torch
    from torch.nn import functional as F
    from sc4001_b.objectives import FeaturePreservation,WeightPreservation
    device=next(model.parameters()).device
    cuda=device.type=='cuda'
    def sync():
        if cuda: torch.cuda.synchronize()
    def reset():
        if cuda: torch.cuda.reset_peak_memory_stats()
    def peaks():
        return {'allocated_bytes':torch.cuda.max_memory_allocated() if cuda else None,
                'reserved_bytes':torch.cuda.max_memory_reserved() if cuda else None}
    model.configure(method if method in ('M1','M2') else 'full')
    batch=len(labels)//accumulation
    if batch*accumulation!=len(labels) or batch<1: raise ValueError('Invalid accumulation batch')
    chunks=[{k:v[i:i+batch] for k,v in inputs.items()} for i in range(0,len(labels),batch)]
    # Pure student-only inference: any teacher remains on CPU until afterwards.
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup): model(**chunks[0])
        sync();reset();start=time.perf_counter()
        for _ in range(steps):
            for chunk in chunks: model(**chunk)
        sync();inference_seconds=time.perf_counter()-start
        inference_peak=peaks()
    if teacher is not None:
        teacher.to(device).eval()
        for p in teacher.parameters(): p.requires_grad=False
    objective=FeaturePreservation(teacher,lam) if method=='M6' else (
        WeightPreservation(model.roberta,lam) if method=='L2SP' else None)
    trainable=[p for p in model.parameters() if p.requires_grad]
    # Match Trainer's AdamW decay split (bias and LayerNorm have no weight decay).
    decay=[];no_decay=[]
    for name,p in model.named_parameters():
        if p.requires_grad: (no_decay if 'bias' in name or 'LayerNorm.weight' in name else decay).append(p)
    optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':.01},
                                {'params':no_decay,'weight_decay':0}],lr=2e-5)
    model.train();losses=[]
    def step():
        optimizer.zero_grad(set_to_none=True)
        total=0.
        for i,chunk in enumerate(chunks):
            outputs=model.roberta(**chunk,return_dict=True)
            logits=model.classifier(outputs.last_hidden_state[:,0])
            loss=F.cross_entropy(logits,labels[i*batch:(i+1)*batch])
            if objective is not None: loss=loss+objective(model,chunk,outputs)[0]
            if not torch.isfinite(loss): raise ValueError('Non-finite benchmark loss')
            (loss/accumulation).backward();total+=float(loss.detach())/accumulation
        torch.nn.utils.clip_grad_norm_(trainable,1.0)
        optimizer.step()
        return total
    # Record cold-start peak, then warmed timing/peak with populated optimizer states.
    sync();reset()
    for _ in range(warmup): step()
    sync();warmup_peak=peaks();reset();durations=[]
    for _ in range(steps):
        sync();start=time.perf_counter();losses.append(step());sync()
        durations.append(time.perf_counter()-start)
    train_peak=peaks()
    return {'training_seconds_per_optimizer_step':durations,'training_loss_diagnostic_only':losses,
            'training_examples_per_second':steps*len(labels)/sum(durations),
            'warmup_peak':warmup_peak,'training_peak':train_peak,'inference_peak':inference_peak,
            'inference_seconds':inference_seconds,'inference_examples_per_second':steps*len(labels)/inference_seconds,
            'trainable_parameters':sum(p.numel() for p in trainable),
            'total_parameters':sum(p.numel() for p in model.parameters()),
            'optimizer_state_tensors':sum(len(s) for s in optimizer.state.values()),
            'intrinsic_reference':method if objective else None,'teacher_present_training':teacher is not None,
            'teacher_used_in_loss':method=='M6','microbatch':batch,'accumulation':accumulation}


def main(argv=None):
    import torch
    import transformers
    import datasets
    from transformers import AutoTokenizer,RobertaModel,set_seed
    from datasets import load_from_disk
    from sc4001_a.model import LinearRoberta
    import sc4001_a.model as model_code
    import sc4001_b.objectives as objective_code
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('artifacts/data-v1'))
    p.add_argument('--data-reference',type=Path,default=Path('../member-a/remote-records/data-v1'))
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--profile',choices=('clean','diagnostic_teacher'),default='clean')
    p.add_argument('--repeat',type=int,choices=(1,2,3),required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    if (a.method,a.profile) not in PROFILES: raise ValueError('Unsupported profile')
    if not torch.cuda.is_available(): raise SystemExit('CUDA required: submit a GPU job')
    verify_data(a.data,a.data_reference)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    set_seed(20261110+a.repeat)
    sources=load(a.data/'sources.json');split_path=a.data/'splits/b400_s13.json'
    indices=load(split_path)['train_indices'][:32]
    if len(indices)!=32 or len(set(indices))!=32: raise ValueError('Need 32 distinct source TRAIN examples')
    rows=load_from_disk(str(a.data/'imdb_train')).select(indices)
    tok=AutoTokenizer.from_pretrained(sources['model_id'],revision=sources['model_revision'],local_files_only=True)
    encoded=tok(rows['text'],truncation=True,padding='max_length',max_length=512,return_tensors='pt')
    inputs={k:v.to('cuda') for k,v in encoded.items() if k in ('input_ids','attention_mask')}
    labels=torch.tensor(rows['label'],device='cuda')
    versions={'torch':torch.__version__,'transformers':transformers.__version__,
              'datasets':datasets.__version__,'numpy':np.__version__,'python':platform.python_version()}
    signature={'version':'C-cost-v2','method':a.method,'profile':a.profile,'repeat':a.repeat,'smoke':a.smoke,
        'base_model':sources['model_id'],'base_revision':sources['model_revision'],
        'split_sha256':digest(split_path),'example_ids':rows['example_id'],
        'text_sha256':[hashlib.sha256(t.encode()).hexdigest() for t in rows['text']],
        'token_ids_sha256':hashlib.sha256(encoded['input_ids'].numpy().tobytes()).hexdigest(),
        'script_sha256':digest(Path(__file__)),'model_sha256':digest(Path(model_code.__file__)),
        'objective_sha256':digest(Path(objective_code.__file__)),
        'versions':versions,'length':512,'batch':8,'accumulation':4,'precision':'FP32; TF32 disabled',
        'warmup_steps':2,'measured_steps':1 if a.smoke else 5,'learning_rate':2e-5,
        'lambda':.01 if a.method=='M6' else .1 if a.method=='L2SP' else None,
        'input_residency':'32 source examples pretokenized and resident on GPU; no data-loader or transfer timing',
        'initialization':'same pretrained encoder and seeded fresh head; M4 phase-2 compute proxy'}
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'manifest.json').exists() and load(a.out/'manifest.json')!=signature:
        raise ValueError('Existing output differs; preserve it and choose a new folder')
    dump(a.out/'manifest.json',signature)
    if (a.out/'COMPLETE').exists():
        print('Matching completed benchmark skipped');return
    model=LinearRoberta.from_pretrained(sources['model_id'],revision=sources['model_revision'],
            use_safetensors=True,local_files_only=True,num_labels=2).float().to('cuda')
    teacher=None
    if a.method=='M6' or a.profile=='diagnostic_teacher':
        teacher=RobertaModel.from_pretrained(sources['model_id'],revision=sources['model_revision'],
                use_safetensors=True,local_files_only=True,add_pooling_layer=False).float()
    gc.collect();torch.cuda.empty_cache()
    # Teacher construction must not change dropout RNG relative to the clean profile.
    set_seed(20261110+a.repeat)
    started=time.perf_counter()
    result=run_probe(model,inputs,labels,a.method,teacher,signature['lambda'] or 0,
                     warmup=2,steps=signature['measured_steps'])
    result.update({'fixture':False,'gpu':torch.cuda.get_device_name(),
        'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,
        'job_id':os.getenv('SLURM_JOB_ID'),'elapsed_seconds':time.perf_counter()-started,
        'interpretation':'Short fixed-shape microbenchmark; not full-run cost or model accuracy',
        'saved_model':False})
    dump(a.out/'result.json',result)
    (a.out/'COMPLETE').write_text('C-cost-v2 smoke completed\n' if a.smoke else 'C-cost-v2 completed\n')
    print(f'Completed {a.method}/{a.profile}/repeat-{a.repeat}',flush=True)


if __name__=='__main__': main()
