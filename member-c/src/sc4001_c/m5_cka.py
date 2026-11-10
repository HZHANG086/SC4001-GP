"""M5-only representation diagnostics using the unchanged C-v2 estimator and inputs.

Runs only AFTER all source-only decisions and formal predictions exist. No B
weights are needed. Cross-method comparisons use the shared pretrained reference.
"""
import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from .common import load, dump, digest, write_csv
from .cka import DOMAINS, verify_data, select_samples, extract, compare_features


def main(argv=None):
    import torch
    from datasets import load_from_disk
    from transformers import AutoTokenizer, RobertaModel
    from sc4001_a.model import LinearRoberta
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('member-c/runs/M5-v1'))
    p.add_argument('--data',type=Path,default=Path('member-c/artifacts/data-v1'))
    p.add_argument('--reference',type=Path,default=Path('member-a/remote-records/data-v1'))
    p.add_argument('--out',type=Path,default=Path('member-c/results/M5-CKA'))
    a=p.parse_args(argv)
    if not torch.cuda.is_available():raise ValueError('Allocated GPU required')
    verify_data(a.data,a.reference)
    lock=load(a.root/'ALL_SELECTIONS_LOCKED.json')
    if len(lock['selections'])!=15:raise ValueError('Incomplete source selection')
    for sel in lock['selections']:
        if not (Path(sel['winner'])/'test/metrics.json').exists():raise ValueError('Complete all formal predictions first')
    sources=load(a.data/'sources.json'); device=torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(20261110);np.random.seed(20261110)
    samples={d:select_samples(load_from_disk(str(a.data/f'{d}_test')),256) for d in DOMAINS}
    sample_manifest={d:[{'example_id':r['example_id'],'label':r['label'],
        'text_sha256':hashlib.sha256(r['text'].encode()).hexdigest()} for r in rows] for d,rows in samples.items()}
    signature={'version':'M5-CKA-v1','selection_sha256':digest(a.root/'ALL_SELECTIONS_LOCKED.json'),
        'samples':sample_manifest,'script_sha256':digest(Path(__file__)),
        'estimator_sha256':digest(Path(__file__).with_name('cka.py')),
        'base_revision':sources['model_revision'],'batch':8,'max_length':512,
        'hardware':torch.cuda.get_device_name(),
        'scope':'15 M5 winners vs pretrained only; same sampling and estimators as C-v2; no direct paired M3 weights'}
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'manifest.json').exists() and load(a.out/'manifest.json')!=signature:raise ValueError('Diagnostic signature changed')
    dump(a.out/'manifest.json',signature)
    if (a.out/'COMPLETE').exists():print('Matching M5 CKA already complete');return
    opts={'revision':sources['model_revision'],'local_files_only':True}
    tok=AutoTokenizer.from_pretrained(sources['model_id'],**opts)
    ref=RobertaModel.from_pretrained(sources['model_id'],use_safetensors=True,add_pooling_layer=False,**opts).float().cuda().eval()
    references={d:extract(ref,tok,rows,device,512,8)[0] for d,rows in samples.items()}
    del ref;gc.collect();torch.cuda.empty_cache()
    collected=[];replays=[];started=time.perf_counter()
    for sel in lock['selections']:
        run=Path(sel['winner']); val=load(run/'validation.json')
        if digest(run/'best/model.safetensors')!=val['checkpoint_weight_sha256']:raise ValueError('Weight checksum mismatch')
        model=LinearRoberta.from_pretrained(run/'best',use_safetensors=True).float().cuda().eval()
        saved_tok=AutoTokenizer.from_pretrained(run/'best')
        probe=[x['text'] for x in samples['imdb'][:4]]
        if saved_tok(probe)['input_ids']!=tok(probe)['input_ids']:raise ValueError('Tokenizer mismatch')
        for d,rows in samples.items():
            features,probs=extract(model.roberta,tok,rows,device,512,8,model.classifier)
            path=run/'test'/f'{d}_predictions.jsonl';ids={x['example_id'] for x in rows}
            with path.open() as f:
                archived={x['example_id']:x for x in (json.loads(line) for line in f) if x['example_id'] in ids}
            if set(archived)!=ids or any(archived[x['example_id']]['label']!=x['label'] for x in rows):raise ValueError('Prediction label/ID mismatch')
            delta=float(np.max(abs(probs-np.array([archived[x['example_id']]['p_positive'] for x in rows]))))
            if not np.isfinite(delta) or delta>1e-4:raise ValueError('Prediction replay mismatch')
            key={'method':'M5','budget':sel['budget'],'seed':sel['seed'],'domain':d}
            replays.append({**key,'max_probability_difference':delta,'n':256,'prediction_file_sha256':digest(path)})
            collected.extend(compare_features(features,references[d],{**key,'reference':'pretrained'}))
        del model;gc.collect();torch.cuda.empty_cache()
        write_csv(a.out/'cka.partial.csv',collected)
        print('M5 CKA complete',sel['budget'],sel['seed'],flush=True)
    if len(collected)!=1170 or len(replays)!=45:raise ValueError('Incomplete CKA grid')
    write_csv(a.out/'cka.csv',collected);write_csv(a.out/'prediction_replay.csv',replays)
    dump(a.out/'run.json',{'status':'COMPLETE','models':15,'rows':1170,'replays':45,
        'gpu':torch.cuda.get_device_name(),'job_id':os.getenv('SLURM_JOB_ID'),'elapsed_seconds':time.perf_counter()-started})
    (a.out/'COMPLETE').write_text('M5-CKA-v1 completed\n')


if __name__=='__main__':main()
