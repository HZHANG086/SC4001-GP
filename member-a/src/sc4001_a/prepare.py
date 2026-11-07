"""Run on allocated compute node; create immutable data artifacts."""
import argparse
import hashlib
import json
import math
import statistics
import random
from collections import Counter, defaultdict
from pathlib import Path

from .protocol import BUDGETS, SEEDS, clean_source, paired_splits, text_hash


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument('--out', default='artifacts/data-v1')
    p.add_argument('--source-lock', help='Existing sources.json to reproduce exact Hub revisions')
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'Refusing to overwrite {out}. Reuse it or choose a new version.')
    out.mkdir(parents=True)
    api = HfApi()
    sources = {'imdb': 'stanfordnlp/imdb', 'yelp': 'fancyzhx/yelp_polarity',
               'amazon': 'fancyzhx/amazon_polarity'}
    model_id = 'FacebookAI/roberta-base'
    if args.source_lock:
        locked = json.loads(Path(args.source_lock).read_text())
        if locked['datasets'] != sources or locked['model_id'] != model_id:
            raise ValueError('Source lock does not match this protocol')
        revisions = locked['revisions']
        model_sha = locked['model_revision']
    else:
        revisions = {k: api.dataset_info(v).sha for k, v in sources.items()}
        model_sha = api.model_info(model_id).sha
    dump(out / 'sources.json', {'datasets': sources, 'revisions': revisions,
                              'model_id': model_id, 'model_revision': model_sha,
                              'label_mapping': {'0': 'negative', '1': 'positive'}})
    tok = AutoTokenizer.from_pretrained(model_id, revision=model_sha)
    # Download only needed inference weights; never execute repository code.
    snapshot_download(model_id, revision=model_sha,
                      allow_patterns=['*.json', '*.txt', '*.safetensors'])
    train = load_dataset(sources['imdb'], revision=revisions['imdb'], split='train')
    test = load_dataset(sources['imdb'], revision=revisions['imdb'], split='test')
    for ds in (train, test):
        assert ds.features['label'].names == ['neg', 'pos']
    label_sets = defaultdict(set)
    for row in train:
        label_sets[text_hash(row['text'])].add(row['label'])
    conflicting = {h for h, ys in label_sets.items() if len(ys) > 1}
    eligible, removal = clean_source(train['text'], test['text'])
    eligible = [i for i in eligible if text_hash(train[i]['text']) not in conflicting]
    removal['conflicting_normalized_text_groups_removed'] = len(conflicting)
    # Keep raw row order: manifests refer to original HF split row indices.
    train = train.add_column('example_id', [f'imdb:train:{i}' for i in range(len(train))])
    test = test.add_column('example_id', [f'imdb:test:{i}' for i in range(len(test))])
    train.save_to_disk(str(out / 'imdb_train'))
    test.save_to_disk(str(out / 'imdb_test'))
    split_dir = out / 'splits'
    split_dir.mkdir()
    for seed in SEEDS:
        for budget, split in paired_splits(train['label'], eligible, seed).items():
            dump(split_dir / f'b{budget}_s{seed}.json',
                 {'budget': budget, 'seed': seed, 'dataset_revision': revisions['imdb'], **split})
    # Target labels only serve fixed balanced evaluation sampling, never selection.
    excluded = {text_hash(t) for t in train['text']} | {text_hash(t) for t in test['text']}
    tests = {'imdb': test}
    audit = {'source_removals': removal, 'eligible_source_rows': len(eligible),
             'test_protocol': 'IMDb official test unchanged; target tests normalized-deduplicated',
             'seed_protocol': 'nested train/validation within each seed; disjoint across budgets',
             'datasets': {}}
    for domain in ('yelp', 'amazon'):
        folder = 'plain_text' if domain == 'yelp' else 'amazon_polarity'
        ds = load_dataset(sources[domain], revision=revisions[domain],
                          data_files={'test': f'{folder}/test-*'}, split='test',
                          verification_mode='no_checks')
        # Hub metadata lists train+test; a test-only override needs explicit
        # local checks instead of the builder's all-splits expectation.
        assert len(ds) == {'yelp': 38000, 'amazon': 400000}[domain], len(ds)
        expected_names = ['1', '2'] if domain == 'yelp' else ['negative', 'positive']
        assert ds.features['label'].names == expected_names, ds.features
        order = list(range(len(ds)))
        random.Random(20261027).shuffle(order)
        seen = set(excluded)
        picked, counts = [], Counter()
        for i in order:
            row = ds[i]
            text = row['text'] if domain == 'yelp' else row['title'] + '\n' + row['content']
            h, y = text_hash(text), row['label']
            if counts[y] >= 2500 or h in seen:
                continue
            seen.add(h)
            picked.append({'text': text, 'label': y, 'example_id': f'{domain}:test:{i}'})
            counts[y] += 1
            if len(picked) == 5000:
                break
        assert dict(counts) == {0: 2500, 1: 2500}, counts
        # Also avoid duplicate selected target examples across Yelp and Amazon.
        excluded.update(text_hash(r['text']) for r in picked)
        tests[domain] = Dataset.from_list(picked)
        tests[domain].save_to_disk(str(out / f'{domain}_test'))
        dump(out / f'{domain}_test_indices.json', [int(r['example_id'].split(':')[-1]) for r in picked])
    for name, ds in {'imdb_train_pool': train.select(eligible), **tests}.items():
        lengths = []
        for start in range(0, len(ds), 128):
            batch = ds[start:start + 128]['text']
            lengths.extend(len(ids) for ids in tok(batch, truncation=False, verbose=False)['input_ids'])
        ordered = sorted(lengths)
        hashes = [text_hash(t) for t in ds['text']]
        audit['datasets'][name] = {
            'n': len(ds), 'labels': dict(Counter(ds['label'])),
            'normalized_duplicate_rows': len(hashes) - len(set(hashes)),
            'token_length_median': statistics.median(ordered),
            'token_length_p95': ordered[max(0, math.ceil(.95 * len(ordered)) - 1)],
            'truncation_rate_256': sum(n > 256 for n in lengths) / len(lengths),
            'truncation_rate_512': sum(n > 512 for n in lengths) / len(lengths)}
    # Diagnostic only: keep target audit closed until experiment protocol is locked.
    dump(out / 'audit.json', audit)
    payload = {str(f.relative_to(out)): hashlib.sha256(f.read_bytes()).hexdigest()
               for f in sorted(out.rglob('*.json'))}
    dump(out / 'manifest_hashes.json', payload)
    (out / 'READY').write_text('Complete. Do not edit data artifacts in place.\n')
    print(f'Prepared {out}; 15 paired split manifests. Check audit.json.')

if __name__ == '__main__':
    main()
