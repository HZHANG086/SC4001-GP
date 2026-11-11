"""M5: source-only LoRA training, recoverable Trainer checkpoints, merged A-compatible export."""
import argparse
from contextlib import nullcontext
import json,os,platform,subprocess,time,shutil
from pathlib import Path
from .common import load,dump,digest
from .lora import attach,merge
from .cka import verify_data


def main(argv=None):
    import numpy as np
    import torch,transformers,datasets
    from torch.nn import functional as F
    from transformers import AutoTokenizer,DataCollatorWithPadding,Trainer,TrainingArguments,set_seed
    from datasets import load_from_disk
    from sc4001_a.model import LinearRoberta
    from sc4001_a.checkpoint import rng_serialization_context
    import sc4001_a.model as model_module
    import sc4001_a.checkpoint as checkpoint_module
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--data-reference',type=Path,required=True)
    p.add_argument('--budget',type=int,choices=(400,1600,6400),required=True)
    p.add_argument('--seed',type=int,choices=(13,21,42,87,100),required=True)
    p.add_argument('--lr',type=float,required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke',action='store_true');p.add_argument('--fixture-model',type=Path)
    p.add_argument('--fixture-epochs',type=int,default=2)
    a=p.parse_args(argv);fixture=a.fixture_model is not None
    if fixture and not a.smoke: raise ValueError('CPU fixture must be smoke; never formal')
    if not fixture and not torch.cuda.is_available(): raise SystemExit('CUDA required in an allocated GPU job')
    spec=load(a.protocol)
    if a.lr not in spec['learning_rates']: raise ValueError('Learning rate outside frozen grid')
    if not fixture:
        verify_data(a.data,a.data_reference)
        for name,sha in spec['code_hashes'].items():
            if digest(Path(__file__).parent/name)!=sha: raise ValueError(f'Frozen M5 code changed: {name}')
    sources=load(a.data/'sources.json');split_path=a.data/'splits'/f'b{a.budget}_s{a.seed}.json'
    split=load(split_path)
    if not fixture and (sources['model_id'],sources['model_revision'])!=(spec['model_id'],spec['model_revision']):
        raise ValueError('Pretrained source mismatch')
    length=32 if fixture else spec['max_length']
    epochs=a.fixture_epochs if fixture else spec['epochs']
    cfg={'method':'M5','protocol':spec['version'],'budget':a.budget,'seed':a.seed,'lr':a.lr,
         'length':length,'epochs':epochs,'batch':8,'accumulation':4,'smoke':a.smoke,
         'stage':'smoke' if a.smoke else 'formal','data_manifest_sha256':digest(a.data/'manifest_hashes.json'),
         'split_sha256':digest(split_path),'model_id':sources['model_id'],'model_revision':sources['model_revision'],
         'lora':spec['lora'],'protocol_sha256':digest(a.protocol),'fixture':fixture,
         'code_sha256':{n:digest(Path(__file__).parent/n) for n in ('lora.py','lora_train.py','common.py')},
         'a_model_sha256':digest(Path(model_module.__file__)),'a_checkpoint_sha256':digest(Path(checkpoint_module.__file__))}
    if a.out.exists():
        if not a.resume: raise ValueError('Output exists; use explicit resume after checking the job')
        if load(a.out/'run_config.json')!=cfg: raise ValueError('Resume configuration changed')
        if (a.out/'COMPLETE').exists(): print('Matching complete run skipped');return
    else: a.out.mkdir(parents=True)
    dump(a.out/'run_config.json',cfg)
    (a.out/'source').mkdir(exist_ok=True)
    for name in ('lora.py','lora_train.py','common.py'):
        shutil.copy2(Path(__file__).parent/name,a.out/'source'/name)
    shutil.copy2(a.protocol,a.out/'source/protocol.json')
    with (a.out/'invocations.jsonl').open('a') as f:
        f.write(json.dumps({'event':'start','unix_time':time.time(),'job_id':os.getenv('SLURM_JOB_ID'),'resume':a.resume})+'\n')
    from importlib.metadata import distributions
    (a.out/'environment.txt').write_text('\n'.join(sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()))+'\n')
    set_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cpu' if fixture else 'cuda'
    base=str(a.fixture_model) if fixture else sources['model_id']
    opts={'revision':None if fixture else sources['model_revision'],'local_files_only':True}
    tok=AutoTokenizer.from_pretrained(base,**opts)
    model=LinearRoberta.from_pretrained(base,use_safetensors=True,num_labels=2,**opts)
    targets=attach(model,**spec['lora'])
    parameters={'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total':sum(p.numel() for p in model.parameters()),'merged_total':124056578 if not fixture else None,
        'trainable_names':[n for n,p in model.named_parameters() if p.requires_grad],'targets':targets}
    if not fixture and parameters['trainable']!=296450: raise ValueError('Unexpected LoRA parameter count')
    dump(a.out/'parameters.json',parameters)
    pool=load_from_disk(str(a.data/'imdb_train'))
    def tokenize(ds):return ds.map(lambda b:tok(b['text'],truncation=True,max_length=length),batched=True,remove_columns=['text','example_id'])
    train=tokenize(pool.select(split['train_indices']));val=tokenize(pool.select(split['validation_indices']))
    if fixture:
        train=train.select(range(32));val=val.select(range(16))
    args=TrainingArguments(output_dir=str(a.out/'checkpoints'),learning_rate=a.lr,num_train_epochs=epochs,
        max_steps=2 if a.smoke and not fixture else -1,per_device_train_batch_size=8,per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,weight_decay=.01,warmup_ratio=.1,
        eval_strategy='epoch',save_strategy='epoch',logging_strategy='steps',logging_steps=10,
        load_best_model_at_end=True,metric_for_best_model='accuracy',greater_is_better=True,save_total_limit=2,
        save_safetensors=True,seed=a.seed,data_seed=a.seed,report_to=[],use_cpu=fixture,fp16=False,bf16=False,
        dataloader_num_workers=0,optim='adamw_torch')
    trainer=Trainer(model=model,args=args,train_dataset=train,eval_dataset=val,processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),compute_metrics=lambda pred:{'accuracy':float(np.mean(pred.predictions.argmax(-1)==pred.label_ids))})
    if not fixture: torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    with rng_serialization_context() if a.resume else nullcontext():trainer.train(resume_from_checkpoint=True if a.resume else None)
    if not fixture: torch.cuda.synchronize()
    train_seconds=time.perf_counter()-started
    train_peak=None if fixture else torch.cuda.max_memory_allocated()
    reserve_peak=None if fixture else torch.cuda.max_memory_reserved()
    metrics=trainer.evaluate()
    trainer.model.eval()
    probe=tok(pool.select(split['validation_indices'][:8])['text'],padding=True,truncation=True,max_length=length,return_tensors='pt').to(device)
    with torch.inference_mode():before=trainer.model(**probe).logits.cpu()
    # Recoverable adapter checkpoint remains in checkpoints/. Only best/ is merged.
    merge(trainer.model)
    with torch.inference_mode():after=trainer.model(**probe).logits.cpu()
    merge_delta=float((before-after).abs().max())
    if merge_delta>1e-4: raise ValueError(f'Adapter merge mismatch: {merge_delta}')
    trainer.model.save_pretrained(a.out/'best',safe_serialization=True);tok.save_pretrained(a.out/'best')
    metrics.update({'train_wall_seconds_this_invocation':train_seconds,'peak_cuda_allocated_bytes':train_peak,
        'peak_cuda_reserved_bytes':reserve_peak,'gpu':None if fixture else torch.cuda.get_device_name(),
        'best_checkpoint':trainer.state.best_model_checkpoint,'global_step':trainer.state.global_step,
        'merge_max_abs_logit_difference':merge_delta,'checkpoint_weight_sha256':digest(a.out/'best/model.safetensors'),
        'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'datasets':datasets.__version__,'python':platform.python_version()},
        'job_id':os.getenv('SLURM_JOB_ID'),'runtime_includes_epoch_validation_and_saving':True,
        'timing_note':'This invocation only; see invocations and Slurm ledger for interruptions. Training peak captured before post-training validation and merge.'})
    dump(a.out/'validation.json',metrics);dump(a.out/'history.json',trainer.state.log_history)
    with (a.out/'invocations.jsonl').open('a') as f:
        f.write(json.dumps({'event':'success','unix_time':time.time(),'job_id':os.getenv('SLURM_JOB_ID'),'train_seconds':train_seconds})+'\n')
    (a.out/'COMPLETE').write_text('SYNTHETIC_TEST_ONLY\n' if fixture else 'M5 smoke complete\n' if a.smoke else 'M5 source-validation training complete; no tests evaluated\n')
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':main()
