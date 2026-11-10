import csv
import hashlib
import json
from pathlib import Path

METHODS = ('M1', 'M2', 'M3', 'M4', 'M6', 'L2SP')
BUDGETS = (400, 1600, 6400)
SEEDS = (13, 21, 42, 87, 100)
DOMAINS = ('imdb', 'yelp', 'amazon')
SIZES = {'imdb': 25000, 'yelp': 5000, 'amazon': 5000}
PAIRS = (('M6', 'M3'), ('M4', 'M3'), ('L2SP', 'M3'), ('M3', 'M2'), ('M6', 'M4'), ('M6', 'L2SP'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def read_csv(path, delimiter=','):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'Empty table: {path}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_rows(repo):
    rows = read_csv(Path(repo) / 'member-b/formal-results/results/B-v1/per_seed.csv')
    for r in rows:
        for k in ('budget', 'seed', 'trainable_parameters'):
            r[k] = int(r[k])
        for k in ('lr', 'source_val_accuracy', 'selected_train_seconds',
                  'selected_peak_allocated_gib', 'search_train_seconds_recorded',
                  'stage1_train_seconds', 'imdb_accuracy', 'yelp_accuracy',
                  'amazon_accuracy', 'ood_accuracy', 'val_feature_distance'):
            r[k] = float(r[k]) if r.get(k) not in ('', None) else None
    return rows


def run_path(repo, row):
    member = 'member-a' if row['method'] in ('M1', 'M2') else 'member-b'
    return Path(repo) / member / 'formal-results' / row['run']


def read_predictions(path, expected_ids=None, expected_labels=None):
    import numpy as np
    with Path(path).open() as stream:
        records = [json.loads(line) for line in stream]
    ids = [r['example_id'] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate example IDs: {path}')
    if expected_ids is not None:
        if set(ids) != set(expected_ids):
            raise ValueError(f'Example ID set mismatch: {path}')
        lookup = {r['example_id']: r for r in records}
        records = [lookup[i] for i in expected_ids]
        ids = list(expected_ids)
    labels = np.array([r['label'] for r in records], dtype=np.int64)
    preds = np.array([r['prediction'] for r in records], dtype=np.int64)
    probs = np.array([r['p_positive'] for r in records], dtype=np.float64)
    if not (np.isin(labels, [0, 1]).all() and np.isin(preds, [0, 1]).all()):
        raise ValueError(f'Nonbinary labels/predictions: {path}')
    if not (np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()):
        raise ValueError(f'Invalid probability: {path}')
    # Two float32 softmax entries need not sum to exactly one. Ignore only ties
    # within 1e-7 here; evaluate.py compares the two entries rather than p > .5.
    wrong = (preds != (probs > .5)) & (np.abs(probs - .5) > 1e-7)
    if wrong.any():
        raise ValueError(f'Probability/prediction mismatch: {path}')
    if expected_labels is not None and not np.array_equal(labels, expected_labels):
        raise ValueError(f'Label mismatch: {path}')
    return ids, labels, preds, probs
