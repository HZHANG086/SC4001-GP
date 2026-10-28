"""Checkpoint/RNG integration check on a random tiny model, compute node only."""
import argparse
import tempfile
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset
from transformers import RobertaConfig, Trainer, TrainerCallback, TrainingArguments, set_seed
from sc4001_a.model import LinearRoberta
from sc4001_a.checkpoint import rng_serialization_context

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true')
cli = parser.parse_args()
if cli.cuda:
    assert torch.cuda.is_available(), 'Allocated CUDA device required'

class StopAfterFirstSave(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_training_stop = True
        return control

with tempfile.TemporaryDirectory(prefix='resume-check-', dir='work') as tmp:
    root = Path(tmp)
    ds = Dataset.from_dict({'input_ids': [[4+i, 7, 8, 2] for i in range(16)],
                            'attention_mask': [[1]*4 for _ in range(16)],
                            'labels': [i % 2 for i in range(16)]})
    def build(folder, interrupted=False):
        set_seed(13)
        cfg = RobertaConfig(vocab_size=64, hidden_size=16, num_hidden_layers=3,
                            num_attention_heads=2, intermediate_size=32,
                            max_position_embeddings=16)
        model = LinearRoberta(cfg)
        model.configure('M2')
        args = TrainingArguments(output_dir=str(folder), max_steps=4,
                                 per_device_train_batch_size=4,
                                 learning_rate=1e-3, use_cpu=not cli.cuda, seed=13,
                                 save_strategy='steps', save_steps=2,
                                 report_to=[], disable_tqdm=True,
                                 dataloader_num_workers=0, optim='adamw_torch')
        return Trainer(model=model, args=args, train_dataset=ds,
                       callbacks=[StopAfterFirstSave()] if interrupted else [])
    full = build(root / 'full')
    full.train()
    interrupted = build(root / 'split', True)
    interrupted.train()
    assert interrupted.state.global_step == 2
    restored = build(root / 'split')
    with rng_serialization_context():
        restored.train(resume_from_checkpoint=str(root / 'split/checkpoint-2'))
    assert restored.state.global_step == 4
    for name, value in full.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[name], rtol=0, atol=1e-7, msg=name)
    print(('CUDA' if cli.cuda else 'CPU') + ' RESUME_CHECK_PASSED: interrupted at step 2, restored to step 4; weights match uninterrupted training.')
