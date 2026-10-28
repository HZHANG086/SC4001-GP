"""Pure Python protocol primitives, shared by data preparation and tests."""
import hashlib
import html
import random
import re
import unicodedata
from collections import Counter

BUDGETS = (400, 1600, 6400)
SEEDS = (13, 21, 42, 87, 100)

def text_hash(text):
    normalized = unicodedata.normalize('NFKC', html.unescape(text))
    normalized = re.sub(r'<br\s*/?>', ' ', normalized, flags=re.I)
    normalized = ' '.join(normalized.casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()

def paired_splits(labels, eligible, seed, budgets=BUDGETS):
    """Nested train and val sets, disjoint even ACROSS budgets within a seed."""
    if any(b <= 0 or b % 20 for b in budgets):
        raise ValueError('Budgets must be positive multiples of 20')
    rng = random.Random(seed)
    by_class = {y: [i for i in eligible if labels[i] == y] for y in (0, 1)}
    max_train = max(budgets) * 4 // 5 // 2
    max_val = max(budgets) // 5 // 2
    for pool in by_class.values():
        rng.shuffle(pool)
        if len(pool) < max_train + max_val:
            raise ValueError('Insufficient unique source examples')
    result = {}
    for b in budgets:
        tr, va = [], []
        for pool in by_class.values():
            tr.extend(pool[:b * 4 // 5 // 2])
            va.extend(pool[max_train:max_train + b // 5 // 2])
        rng.shuffle(tr)
        rng.shuffle(va)
        assert not set(tr) & set(va)
        result[b] = {'train_indices': tr, 'validation_indices': va}
    return result

def clean_source(train_texts, test_texts):
    """Deduplicate source train; remove normalized train/test overlap.

    Test set stays intact. No test labels or predictions are used here.
    Conflicting duplicate source labels are handled by caller before this step.
    """
    test_hashes = {text_hash(t) for t in test_texts}
    seen, eligible = set(), []
    counts = Counter()
    for i, t in enumerate(train_texts):
        h = text_hash(t)
        if h in test_hashes:
            counts['train_test_overlap_rows_removed'] += 1
        elif h in seen:
            counts['duplicate_train_rows_removed'] += 1
        else:
            eligible.append(i)
        seen.add(h)
    return eligible, dict(counts)
