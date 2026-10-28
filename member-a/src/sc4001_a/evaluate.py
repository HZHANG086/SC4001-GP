"""Evaluate a locked winner once; do not run on all hyperparameter candidates."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    import torch
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    from .model import LinearRoberta
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--data', default='artifacts/data-v1')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--protocol-locked', action='store_true')
    a = p.parse_args()
    if not a.protocol_locked:
        raise SystemExit('Lock length, search space and source-selected winner before test evaluation.')
    run = Path(a.run)
    if not (run / 'COMPLETE').exists():
        raise SystemExit('Training incomplete')
    cfg = json.loads((run / 'run_config.json').read_text())
    data = Path(a.data)
    if not (data / 'READY').exists():
        raise SystemExit('Evaluation data preparation incomplete')
    if hashlib.sha256((data / 'manifest_hashes.json').read_bytes()).hexdigest() != cfg['data_manifest_sha256']:
        raise SystemExit('Evaluation data differs from training data version')
    if cfg['smoke'] or cfg.get('stage', 'formal') != 'formal':
        raise SystemExit('Smoke/pilot runs cannot produce formal test results')
    if not torch.cuda.is_available():
        raise SystemExit('Evaluation requires allocated GPU on TC1')
    target = run / 'test'
    target.mkdir(exist_ok=False)
    model = LinearRoberta.from_pretrained(run / 'best', use_safetensors=True).cuda().eval()
    tok = AutoTokenizer.from_pretrained(run / 'best')
    metrics = {}
    for domain in ('imdb', 'yelp', 'amazon'):
        ds = load_from_disk(str(Path(a.data) / f'{domain}_test'))
        correct = 0
        with (target / f'{domain}_predictions.jsonl').open('w') as f, torch.inference_mode():
            for start in range(0, len(ds), a.batch):
                rows = ds[start:start + a.batch]
                batch = tok(rows['text'], padding=True, truncation=True,
                            max_length=cfg['length'], return_tensors='pt').to('cuda')
                probs = model(**batch).logits.softmax(-1).cpu().tolist()
                for idx, y, prob in zip(rows['example_id'], rows['label'], probs):
                    pred = int(prob[1] > prob[0])
                    correct += pred == y
                    f.write(json.dumps({'example_id': idx, 'label': y,
                                        'prediction': pred, 'p_positive': prob[1]}) + '\n')
        metrics[domain] = {'accuracy': correct / len(ds), 'n': len(ds)}
    metrics['ood_mean_accuracy'] = (metrics['yelp']['accuracy'] + metrics['amazon']['accuracy']) / 2
    (target / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
