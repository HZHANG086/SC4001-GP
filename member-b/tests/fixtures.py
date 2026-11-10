"""Offline fixtures: tiny tokenizer, tiny LinearRoberta and a fake data-v1 directory.

The fake data directory has Member A's layout (READY, sources.json, manifest_hashes.json,
splits/b400_s13.json, imdb_train saved with datasets), so sc4001_b.train runs unchanged.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / 'member-a' / 'protocol' / 'locked_src', ROOT / 'member-b' / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

WORDS = ['good', 'great', 'fine', 'bad', 'awful', 'dull', 'the', 'film', 'plot', 'was', 'and']


def make_tokenizer(folder):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    vocab = {t: i for i, t in enumerate(['<s>', '<pad>', '</s>', '<unk>'] + WORDS)}
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token='<unk>'))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    core.post_processor = processors.TemplateProcessing(
        single='<s> $A </s>', special_tokens=[('<s>', 0), ('</s>', 2)])
    tok = PreTrainedTokenizerFast(tokenizer_object=core, bos_token='<s>', eos_token='</s>',
                                  pad_token='<pad>', unk_token='<unk>', cls_token='<s>',
                                  sep_token='</s>', model_input_names=['input_ids', 'attention_mask'])
    tok.save_pretrained(folder)
    return tok


def make_model(folder, seed=0):
    import torch
    from transformers import RobertaConfig
    from sc4001_a.model import LinearRoberta
    torch.manual_seed(seed)
    cfg = RobertaConfig(vocab_size=4 + len(WORDS), hidden_size=16, num_hidden_layers=3,
                        num_attention_heads=2, intermediate_size=32, max_position_embeddings=40,
                        pad_token_id=1, bos_token_id=0, eos_token_id=2)
    LinearRoberta(cfg).save_pretrained(folder, safe_serialization=True)
    make_tokenizer(folder)
    return Path(folder)


def make_data(folder, n=500, seed=0):
    """Fake artifacts/data-v1 with one b400_s13 split produced by Member A's paired_splits."""
    from datasets import Dataset
    from sc4001_a.protocol import paired_splits
    folder = Path(folder)
    rng = random.Random(seed)
    texts, labels = [], []
    for i in range(n):
        y = i % 2
        polar = ['good', 'great', 'fine'] if y else ['bad', 'awful', 'dull']
        words = [rng.choice(WORDS[6:]) for _ in range(rng.randint(3, 12))]
        words.insert(rng.randrange(len(words)), rng.choice(polar))
        texts.append(' '.join(words))
        labels.append(y)
    ds = Dataset.from_dict({'text': texts, 'label': labels,
                            'example_id': [f'imdb:train:{i}' for i in range(n)]})
    ds.save_to_disk(str(folder / 'imdb_train'))
    (folder / 'splits').mkdir(parents=True)
    split = paired_splits(labels, list(range(n)), 13, budgets=(400,))[400]
    (folder / 'splits' / 'b400_s13.json').write_text(json.dumps({'budget': 400, 'seed': 13, **split}))
    (folder / 'sources.json').write_text(json.dumps({
        'datasets': {}, 'revisions': {'imdb': 'test'}, 'model_id': 'test-model',
        'model_revision': 'test', 'label_mapping': {'0': 'negative', '1': 'positive'}}))
    hashes = {str(f.relative_to(folder)): hashlib.sha256(f.read_bytes()).hexdigest()
              for f in sorted(folder.rglob('*.json'))}
    (folder / 'manifest_hashes.json').write_text(json.dumps(hashes))
    (folder / 'READY').write_text('test\n')
    return folder
