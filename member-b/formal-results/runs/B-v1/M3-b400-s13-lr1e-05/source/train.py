"""Single source-validation-selected Member B run. Test evaluation is a separate command.

Methods (all start from FacebookAI/roberta-base with Member A's LinearRoberta CLS head):
  M3     full fine-tuning
  M4S1   M4 stage 1: frozen encoder, train the linear head (identical to Member A's M1)
  M4     M4 stage 2: full fine-tuning initialised from an M4S1 run (--stage1)
  M6     full fine-tuning + feature preservation, L = CE + lam * (1 - cos)   (--lam)
  L2SP   full fine-tuning + weight preservation,  L = CE + lam * L2-SP       (--lam)

Training arguments, data handling, checkpoint selection and output files mirror
member-a/src/sc4001_a/train.py so that Member A's locked evaluate.py and the shared
aggregation apply unchanged.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import subprocess
import shutil
import sys
import time
from pathlib import Path

METHODS = ('M3', 'M4S1', 'M4', 'M6', 'L2SP')
NEEDS_LAMBDA = ('M6', 'L2SP')
B_SOURCES = ('train.py', 'objectives.py')
A_SOURCES = ('model.py', 'checkpoint.py')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='artifacts/data-v1')
    p.add_argument('--out', required=True)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--budget', type=int, choices=[400, 1600, 6400], default=400)
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--lam', type=float, default=None, help='Regulariser weight (M6, L2SP)')
    p.add_argument('--stage1', default=None, help='Completed M4S1 run directory (M4 only)')
    p.add_argument('--length', type=int, choices=[256, 512], default=512)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--max-steps', type=int, default=-1, help='Positive value marks a smoke run')
    p.add_argument('--cpu', action='store_true', help='Local diagnostic only; not on TC1 head node')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stage', choices=['pilot', 'formal'], default='formal')
    p.add_argument('--model-path', default=None,
                   help='Local test model instead of the Hub revision; marks the run as a test')
    a = p.parse_args(argv)
    if (a.lam is None) == (a.method in NEEDS_LAMBDA):
        p.error(f'--lam is required for {NEEDS_LAMBDA} and not allowed otherwise')
    if a.lam is not None and a.lam < 0:
        p.error('--lam must be non-negative')
    if (a.stage1 is None) == (a.method == 'M4'):
        p.error('--stage1 is required for M4 and not allowed otherwise')
    return a


def stage1_record(stage1, budget, seed, length, manifest):
    """Validate an M4S1 run used to initialise M4 and summarise it for the run config."""
    stage1 = Path(stage1)
    if not (stage1 / 'COMPLETE').exists() or not (stage1 / 'best').is_dir():
        raise SystemExit(f'Stage 1 run incomplete or pruned: {stage1}')
    cfg = json.loads((stage1 / 'run_config.json').read_text())
    expected = {'method': 'M4S1', 'budget': budget, 'seed': seed, 'length': length,
                'data_manifest_sha256': manifest}
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise SystemExit(f'Stage 1 run does not match this cell: {stage1}')
    val = json.loads((stage1 / 'validation.json').read_text())
    return {'run': str(stage1), 'lr': cfg['lr'], 'source_val_accuracy': val['eval_accuracy'],
            'train_wall_seconds': val['train_wall_seconds_this_invocation'],
            'weights_sha256': sha256(stage1 / 'best' / 'model.safetensors')}


def validation_feature_distance(model, teacher, dataset, collator, batch, device):
    """Mean pooled-representation distance from the pretrained encoder on validation data."""
    import torch
    from .objectives import feature_distance
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(dataset), batch):
            rows = dataset.select(range(start, min(start + batch, len(dataset))))
            features = collator([{k: rows[i][k] for k in ('input_ids', 'attention_mask')}
                                 for i in range(len(rows))])
            ids = features['input_ids'].to(device)
            mask = features['attention_mask'].to(device)
            student = model.roberta(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
            reference = teacher(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
            distance = feature_distance(student, reference, mask)
            total += float(distance.sum())
            count += len(distance)
    return total / count


def main(argv=None):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from datasets import load_from_disk
    from transformers import (AutoTokenizer, DataCollatorWithPadding, RobertaModel, Trainer,
                              TrainingArguments, set_seed)
    from transformers.modeling_outputs import SequenceClassifierOutput
    import sc4001_a.model as a_model
    from sc4001_a.checkpoint import rng_serialization_context
    from sc4001_a.model import LinearRoberta
    from .objectives import FeaturePreservation, WeightPreservation

    a = parse_args(argv)
    if not a.cpu and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Submit through SLURM; do not train on the head node.')
    data, out = Path(a.data), Path(a.out)
    if not (data / 'READY').exists():
        raise SystemExit('Data preparation incomplete: missing READY')
    if out.exists() and not a.resume:
        raise SystemExit('Output exists. Use a new candidate path or --resume.')
    sources = json.loads((data / 'sources.json').read_text())
    split_path = data / 'splits' / f'b{a.budget}_s{a.seed}.json'
    split = json.loads(split_path.read_text())
    manifest = sha256(data / 'manifest_hashes.json')
    stage1 = stage1_record(a.stage1, a.budget, a.seed, a.length, manifest) if a.stage1 else None
    a_dir = Path(a_model.__file__).parent
    smoke = a.max_steps > 0 or a.model_path is not None
    config = vars(a) | {'protocol': 'B-v1', 'model_id': sources['model_id'],
                        'model_revision': sources['model_revision'],
                        'split_sha256': sha256(split_path), 'data_manifest_sha256': manifest,
                        'dataset_revisions': sources['revisions'], 'smoke': smoke,
                        'stage': 'smoke' if smoke else a.stage, 'stage1_record': stage1,
                        'code_sha256': {**{f: sha256(Path(__file__).with_name(f)) for f in B_SOURCES},
                                        **{f'sc4001_a/{f}': sha256(a_dir / f) for f in A_SOURCES}}}
    out.mkdir(parents=True, exist_ok=True)
    if a.resume:
        previous = json.loads((out / 'run_config.json').read_text())
        if {k: v for k, v in previous.items() if k != 'resume'} != {k: v for k, v in config.items() if k != 'resume'}:
            raise SystemExit('Resume config does not match original run')
    else:
        (out / 'run_config.json').write_text(json.dumps(config, indent=2))
        (out / 'source').mkdir()
        for f in B_SOURCES:
            shutil.copy2(Path(__file__).with_name(f), out / 'source' / f)
        for f in A_SOURCES:
            shutil.copy2(a_dir / f, out / 'source' / f'a_{f}')

    set_seed(a.seed)
    base = a.model_path or sources['model_id']
    revision = None if a.model_path else sources['model_revision']
    tok = AutoTokenizer.from_pretrained(base, revision=revision)
    if a.method == 'M4':
        model = LinearRoberta.from_pretrained(Path(a.stage1) / 'best', use_safetensors=True)
    else:
        model = LinearRoberta.from_pretrained(base, revision=revision, use_safetensors=True, num_labels=2)
    model.configure('M1' if a.method == 'M4S1' else 'full')
    device = torch.device('cpu' if a.cpu else 'cuda')
    teacher = None
    if a.method != 'M4S1':
        # Frozen pretrained encoder: M6 anchor target and the drift diagnostic for every method.
        teacher = RobertaModel.from_pretrained(base, revision=revision, use_safetensors=True,
                                               add_pooling_layer=False).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
    objective = None
    if a.method == 'M6':
        objective = FeaturePreservation(teacher, a.lam)
    elif a.method == 'L2SP':
        objective = WeightPreservation(model.roberta, a.lam)
    parameters = {'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'total': sum(p.numel() for p in model.parameters()),
                  'trainable_names': [n for n, p in model.named_parameters() if p.requires_grad]}
    (out / 'parameters.json').write_text(json.dumps(parameters, indent=2))

    pool = load_from_disk(str(data / 'imdb_train'))
    def tokenize(ds):
        return ds.map(lambda b: tok(b['text'], truncation=True, max_length=a.length),
                      batched=True, remove_columns=['text', 'example_id'])
    train_ds = tokenize(pool.select(split['train_indices']))
    val_ds = tokenize(pool.select(split['validation_indices']))

    class ObjectiveTrainer(Trainer):
        """Trainer whose loss is CE on the CLS logits plus the method's regulariser."""
        def __init__(self, *args, objective=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.objective = objective
            self.reg_sum, self.reg_count = 0.0, 0

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            net = self.accelerator.unwrap_model(model)
            outputs = net.roberta(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                                  token_type_ids=inputs.get('token_type_ids'), return_dict=True)
            logits = net.classifier(outputs.last_hidden_state[:, 0])
            loss = F.cross_entropy(logits, inputs['labels'])
            if self.objective is not None and model.training:
                penalty, raw = self.objective(net, inputs, outputs)
                loss = loss + penalty
                self.reg_sum += float(raw)
                self.reg_count += 1
            result = SequenceClassifierOutput(loss=loss, logits=logits)
            return (loss, result) if return_outputs else loss

        def log(self, logs, *args, **kwargs):
            if self.reg_count and 'loss' in logs:
                logs[f'train_{self.objective.name}'] = self.reg_sum / self.reg_count
                self.reg_sum, self.reg_count = 0.0, 0
            super().log(logs, *args, **kwargs)

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
    trainer = ObjectiveTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                               processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                               compute_metrics=lambda pred: {'accuracy': float(np.mean(pred.predictions.argmax(-1) == pred.label_ids))},
                               objective=objective)
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
    if teacher is not None:
        metrics['val_feature_distance'] = validation_feature_distance(
            trainer.model, teacher, val_ds, DataCollatorWithPadding(tok), a.batch * 2, device)
    (out / 'validation.json').write_text(json.dumps(metrics, indent=2))
    (out / 'history.json').write_text(json.dumps(trainer.state.log_history, indent=2))
    (out / 'environment.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    (out / 'COMPLETE').write_text('Source validation complete. No tests evaluated.\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
