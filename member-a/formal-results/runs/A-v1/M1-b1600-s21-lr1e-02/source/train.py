"""Single source-validation-selected run. Test evaluation is a separate command."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import subprocess
import shutil
import time
from pathlib import Path


def main():
    import numpy as np
    import torch
    from datasets import load_from_disk
    from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
    from .model import LinearRoberta
    from .checkpoint import rng_serialization_context

    p = argparse.ArgumentParser()
    p.add_argument('--data', default='artifacts/data-v1')
    p.add_argument('--out', required=True)
    p.add_argument('--method', choices=['M1', 'M2'], required=True)
    p.add_argument('--budget', type=int, choices=[400, 1600, 6400], default=400)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--length', type=int, choices=[256, 512], default=256)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--max-steps', type=int, default=-1, help='Positive value marks a smoke run')
    p.add_argument('--cpu', action='store_true', help='Local diagnostic only; not on TC1 head node')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stage', choices=['pilot', 'formal'], default='formal')
    a = p.parse_args()
    if not a.cpu and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Submit through SLURM; do not train on the head node.')
    data, out = Path(a.data), Path(a.out)
    if not (data / 'READY').exists():
        raise SystemExit('Data preparation incomplete: missing READY')
    if out.exists() and not a.resume:
        raise SystemExit('Output exists. Use a new candidate path or --resume.')
    out.mkdir(parents=True, exist_ok=True)
    sources = json.loads((data / 'sources.json').read_text())
    split_path = data / 'splits' / f'b{a.budget}_s{a.seed}.json'
    split = json.loads(split_path.read_text())
    config = vars(a) | {'model_id': sources['model_id'], 'model_revision': sources['model_revision'],
                       'split_sha256': hashlib.sha256(split_path.read_bytes()).hexdigest(),
                       'data_manifest_sha256': hashlib.sha256((data / 'manifest_hashes.json').read_bytes()).hexdigest(),
                       'dataset_revisions': sources['revisions'], 'smoke': a.max_steps > 0,
                       'stage': 'smoke' if a.max_steps > 0 else a.stage,
                       'code_sha256': {f: hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()
                                       for f in ('train.py', 'model.py', 'checkpoint.py')}}
    if a.resume:
        previous = json.loads((out / 'run_config.json').read_text())
        if {k:v for k,v in previous.items() if k != 'resume'} != {k:v for k,v in config.items() if k != 'resume'}:
            raise SystemExit('Resume config does not match original run')
    else:
        (out / 'run_config.json').write_text(json.dumps(config, indent=2))
        (out / 'source').mkdir()
        for filename in config['code_sha256']:
            shutil.copy2(Path(__file__).with_name(filename), out / 'source' / filename)
    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(sources['model_id'], revision=sources['model_revision'])
    model = LinearRoberta.from_pretrained(sources['model_id'], revision=sources['model_revision'],
                                         use_safetensors=True, num_labels=2)
    model.configure(a.method)
    parameters = {'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'total': sum(p.numel() for p in model.parameters()),
                  'trainable_names': [n for n,p in model.named_parameters() if p.requires_grad]}
    (out / 'parameters.json').write_text(json.dumps(parameters, indent=2))
    pool = load_from_disk(str(data / 'imdb_train'))
    def tokenize(ds):
        return ds.map(lambda b: tok(b['text'], truncation=True, max_length=a.length),
                      batched=True, remove_columns=['text', 'example_id'])
    train_ds = tokenize(pool.select(split['train_indices']))
    val_ds = tokenize(pool.select(split['validation_indices']))
    args = TrainingArguments(
        output_dir=str(out / 'checkpoints'), learning_rate=a.lr,
        num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch * 2,
        gradient_accumulation_steps=a.accumulation, weight_decay=.01, warmup_ratio=.1,
        eval_strategy='epoch', save_strategy='epoch', logging_strategy='steps', logging_steps=10,
        load_best_model_at_end=True, metric_for_best_model='accuracy', greater_is_better=True,
        save_total_limit=2, save_safetensors=True, seed=a.seed, data_seed=a.seed,
        report_to=[], use_cpu=a.cpu, fp16=False, bf16=False,
        dataloader_num_workers=0, optim='adamw_torch')
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=lambda pred: {'accuracy': float(np.mean(pred.predictions.argmax(-1) == pred.label_ids))})
    started = time.perf_counter()
    if not a.cpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    with rng_serialization_context() if a.resume else nullcontext():
        trainer.train(resume_from_checkpoint=True if a.resume else None)
    if not a.cpu:
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    metrics = trainer.evaluate()
    trainer.save_model(str(out / 'best'))
    tok.save_pretrained(out / 'best')
    metrics.update({'train_wall_seconds_this_invocation': train_seconds,
                    'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if not a.cpu else None,
                    'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved() if not a.cpu else None,
                    'gpu': torch.cuda.get_device_name() if not a.cpu else None,
                    'best_checkpoint': trainer.state.best_model_checkpoint})
    (out / 'validation.json').write_text(json.dumps(metrics, indent=2))
    (out / 'history.json').write_text(json.dumps(trainer.state.log_history, indent=2))
    (out / 'environment.txt').write_text(subprocess.check_output(
        ['python', '-m', 'pip', 'freeze'], text=True))
    (out / 'COMPLETE').write_text('Source validation complete. No tests evaluated.\n')
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
