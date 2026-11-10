"""Memory-bounded, frozen-model representation analysis for TC1.

No training, no model selection, no saved weights/activations. The same manifest
of inputs is used for every model. Only NumPy, torch, transformers and datasets
are required on the server. Use one seed per SLURM job.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
from .common import digest, load, dump, read_csv, write_csv, DOMAINS


def centered_gram(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or len(x)<4 or not np.isfinite(x).all():
        raise ValueError('CKA requires >= 4 finite sample vectors')
    x=x-x.mean(axis=0,keepdims=True)
    return x@x.T


def hsic_unbiased(k,l):
    """Unbiased HSIC up to the common positive factor n*(n-3)."""
    k=k.copy(); l=l.copy(); n=len(k)
    np.fill_diagonal(k,0); np.fill_diagonal(l,0)
    return float(np.sum(k*l)+k.sum()*l.sum()/((n-1)*(n-2))-
                 2*np.dot(k.sum(0),l.sum(0))/(n-2))


def gram_cka(k,l,debiased=False):
    if k.shape!=l.shape or len(k)<4:
        raise ValueError('Gram shapes differ or insufficient samples')
    if debiased:
        cross,kk,ll=hsic_unbiased(k,l),hsic_unbiased(k,k),hsic_unbiased(l,l)
    else:
        cross,kk,ll=float(np.sum(k*l)),float(np.sum(k*k)),float(np.sum(l*l))
    if kk<=1e-24 or ll<=1e-24:
        return None
    return cross/np.sqrt(kk*ll)


def linear_cka(x,y,debiased=False):
    return gram_cka(centered_gram(x),centered_gram(y),debiased)


def cosine_distance(x,y):
    x=np.asarray(x,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    den=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
    if (den<=1e-20).any():
        return None
    return float(np.mean(1-(x*y).sum(1)/den))


def select_samples(dataset,n,salt='SC4001-C-v1-20261110'):
    if n<4 or n%2:
        raise ValueError('Sample size must be even and >=4')
    candidates={0:[],1:[]}
    ids=dataset['example_id']; labels=dataset['label']
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate IDs in prepared data')
    for i,(ident,y) in enumerate(zip(ids,labels)):
        if y not in candidates:
            raise ValueError('Nonbinary label')
        candidates[y].append((hashlib.sha256((salt+'|'+ident).encode()).hexdigest(),i))
    if min(map(len,candidates.values()))<n//2:
        raise ValueError('Insufficient examples for balanced subset')
    indices=[i for y in (0,1) for _,i in sorted(candidates[y])[:n//2]]
    indices.sort(key=lambda i:ids[i])
    return [dataset[i] for i in indices]


def extract(encoder,tokenizer,rows,device,max_length=512,batch=8,classifier=None):
    import torch
    encoder.eval()
    states={'cls':None,'mean':None}
    probabilities=[]
    with torch.inference_mode():
        for start in range(0,len(rows),batch):
            inp=tokenizer([r['text'] for r in rows[start:start+batch]],padding=True,
                          truncation=True,max_length=max_length,return_tensors='pt').to(device)
            outputs=encoder(input_ids=inp['input_ids'],attention_mask=inp['attention_mask'],
                            output_hidden_states=True,return_dict=True)
            if classifier is not None:
                probabilities.extend(classifier(outputs.last_hidden_state[:,0]).softmax(-1)[:,1].cpu().tolist())
            mask=inp['attention_mask'].unsqueeze(-1)
            for pool in states:
                if states[pool] is None:
                    states[pool]=[[] for _ in outputs.hidden_states]
            for layer,h in enumerate(outputs.hidden_states):
                # Mean includes every non-padding token, matching B's regularizer.
                states['cls'][layer].append(h[:,0].float().cpu().numpy())
                states['mean'][layer].append(((h*mask).sum(1)/mask.sum(1)).float().cpu().numpy())
    result={pool:[np.concatenate(v,axis=0) for v in layers] for pool,layers in states.items()}
    return result,np.asarray(probabilities)


def compare_features(current,reference,info):
    rows=[]
    for pool in ('cls','mean'):
        if len(current[pool])!=len(reference[pool]):
            raise ValueError('Layer count differs')
        for layer,(x,y) in enumerate(zip(current[pool],reference[pool])):
            k,l=centered_gram(x),centered_gram(y)
            perm=np.random.default_rng(20261110).permutation(len(y))
            shuffled=l[perm][:,perm]
            biased=gram_cka(k,l)
            rows.append({**info,'pooling':pool,'layer':layer,'n_examples':len(x),
                         'linear_cka':biased,'debiased_linear_cka':gram_cka(k,l,True),
                         'shuffled_linear_cka':gram_cka(k,shuffled),
                         'cosine_distance':cosine_distance(x,y),
                         'status':'degenerate_zero_variance' if biased is None else 'ok'})
    return rows


def verify_data(data,reference):
    if not (data/'READY').exists():
        raise ValueError('Prepared data READY missing')
    recorded=load(reference/'manifest_hashes.json')
    checked=[]
    for name,sha in recorded.items():
        if name.endswith(('state.json','dataset_info.json')):
            continue
        if digest(data/name)!=sha:
            raise ValueError(f'Data metadata mismatch: {name}')
        checked.append(name)
    return checked


def main(argv=None):
    import torch
    import transformers
    import datasets
    from transformers import AutoTokenizer,RobertaModel
    from datasets import load_from_disk
    from sc4001_a.model import LinearRoberta
    p=argparse.ArgumentParser()
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--model-base',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--data-reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--methods',nargs='+',default=['M3','M4','M6','L2SP'])
    p.add_argument('--seeds',nargs='+',type=int,default=[13])
    p.add_argument('--budgets',nargs='+',type=int,default=[400,1600,6400])
    p.add_argument('--samples',type=int,default=256)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--cpu-fixture',action='store_true',help='Synthetic offline testing ONLY')
    p.add_argument('--reference-fixture',type=Path,help='Synthetic reference ONLY')
    a=p.parse_args(argv)
    fixture=a.cpu_fixture
    if bool(a.reference_fixture)!=fixture:
        raise ValueError('Fixture reference and cpu-fixture must be used together')
    if not fixture and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable; use an allocated GPU, not the login node')
    if not fixture and a.samples!=256:
        raise ValueError('Formal C-v2 fixes 256 inputs/domain; change protocol version to change sample size')
    device=torch.device('cpu' if fixture else 'cuda')
    torch.manual_seed(20261110); np.random.seed(20261110)
    if not fixture:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter()
    a.out.mkdir(parents=True,exist_ok=True)
    checked=verify_data(a.data,a.data_reference)
    sources=load(a.data/'sources.json')
    model_id=sources['model_id']; revision=sources['model_revision']
    chosen=[r for r in read_csv(a.models,'\t') if r['method'] in a.methods and
            int(r['seed']) in a.seeds and int(r['budget']) in a.budgets]
    keys={(r['method'],int(r['budget']),int(r['seed'])) for r in chosen}
    expected={(m,b,s) for m in a.methods for b in a.budgets for s in a.seeds}
    if len(keys)!=len(chosen) or keys!=expected:
        raise ValueError('Incomplete or duplicate requested model grid')
    chosen.sort(key=lambda r:(r['method']!='M3',r['method'],int(r['budget']),int(r['seed'])))
    samples={d:select_samples(load_from_disk(str(a.data/f'{d}_test')),a.samples) for d in DOMAINS}
    sample_manifest={d:[{'example_id':r['example_id'],'label':r['label'],
                         'text_sha256':hashlib.sha256(r['text'].encode()).hexdigest()} for r in rows]
                     for d,rows in samples.items()}
    model_info=[]
    for r in chosen:
        folder=a.model_base/r['model_dir']; cfg=load(folder.parent/'run_config.json')
        if (cfg['method'],cfg['budget'],cfg['seed'])!=(r['method'],int(r['budget']),int(r['seed'])):
            raise ValueError(f'Model/config mismatch: {folder}')
        if not fixture and (cfg['model_revision']!=revision or cfg['length']!=512):
            raise ValueError('Unexpected model revision/length')
        sel=load(a.archive/'runs/B-v1'/f"selection-{r['method']}-b{r['budget']}-s{r['seed']}.json")
        if r['model_dir']!=sel['winner']+'/best':
            raise ValueError('Manifest does not identify frozen winner')
        model_info.append({**r,'weight_sha256':digest(folder/'model.safetensors'),
                           'config_sha256':digest(folder/'config.json')})
    signature={'version':'C-CKA-v2','fixture':fixture,'samples':sample_manifest,'models':model_info,
               'base_model':model_id,'base_revision':revision,'batch':a.batch,'max_length':32 if fixture else 512,
               'script_sha256':digest(Path(__file__)),'common_sha256':digest(Path(__file__).with_name('common.py')),
               'model_code_sha256':digest(Path(__import__('sc4001_a.model',fromlist=['x']).__file__)),
               'versions':{'torch':torch.__version__,'transformers':transformers.__version__,
                           'datasets':datasets.__version__,'numpy':np.__version__}}
    if (a.out/'manifest.json').exists():
        if load(a.out/'manifest.json')!=signature:
            raise ValueError('Existing output uses different inputs/code; choose a new output folder')
    else:
        dump(a.out/'manifest.json',signature)
    if (a.out/'COMPLETE').exists():
        print('Matching complete output already exists; skipped.',flush=True)
        return
    ref_path=a.reference_fixture or model_id
    tokenizer=AutoTokenizer.from_pretrained(ref_path,revision=None if fixture else revision,local_files_only=not fixture)
    if fixture:
        ref=LinearRoberta.from_pretrained(ref_path,use_safetensors=True).roberta
    else:
        ref=RobertaModel.from_pretrained(ref_path,revision=revision,use_safetensors=True,
                                        add_pooling_layer=False,local_files_only=True)
    ref=ref.float().to(device).eval()
    max_length=32 if fixture else 512
    references={d:extract(ref,tokenizer,r,device,max_length,a.batch)[0] for d,r in samples.items()}
    del ref; gc.collect()
    if not fixture: torch.cuda.empty_cache()
    collected=[]; verification=[]; m3_features={}
    for r in chosen:
        m,b,s=r['method'],int(r['budget']),int(r['seed'])
        folder=a.model_base/r['model_dir']
        model=LinearRoberta.from_pretrained(folder,use_safetensors=True).float().to(device).eval()
        # Use the SAME frozen tokenizer for every model; validate saved tokenizer too.
        saved_tok=AutoTokenizer.from_pretrained(folder)
        probe=[x['text'] for x in samples['imdb'][:4]]
        if saved_tok(probe)['input_ids']!=tokenizer(probe)['input_ids']:
            raise ValueError('Saved tokenizer differs from frozen base tokenizer')
        for d,input_rows in samples.items():
            features,probs=extract(model.roberta,tokenizer,input_rows,device,max_length,a.batch,model.classifier)
            pred_path=a.archive/Path(r['model_dir']).parent/'test'/f'{d}_predictions.jsonl'
            selected_ids={x['example_id'] for x in input_rows}
            with pred_path.open() as stream:
                archived={x['example_id']:x for x in (json.loads(line) for line in stream)
                          if x['example_id'] in selected_ids}
            if len(archived)!=len(selected_ids):
                raise ValueError('Missing archived prediction for CKA subset')
            expected_probs=np.array([archived[x['example_id']]['p_positive'] for x in input_rows])
            if any(archived[x['example_id']]['label']!=x['label'] for x in input_rows):
                raise ValueError('CKA inputs/archived labels differ')
            delta=float(abs(probs-expected_probs).max())
            if not np.isfinite(delta) or delta>1e-4:
                raise ValueError(f'Checkpoint replay probability mismatch {m}/{b}/{s}/{d}: {delta}')
            verification.append({'method':m,'budget':b,'seed':s,'domain':d,'max_probability_difference':delta,
                                 'prediction_file_sha256':digest(pred_path),'n':len(probs)})
            collected.extend(compare_features(features,references[d],
                {'method':m,'reference':'pretrained','budget':b,'seed':s,'domain':d}))
            if m=='M3':
                m3_features[b,s,d]=features
            elif m in ('M4','M6','L2SP') and (b,s,d) in m3_features:
                collected.extend(compare_features(features,m3_features[b,s,d],
                    {'method':m,'reference':'M3','budget':b,'seed':s,'domain':d}))
        del model; gc.collect()
        if not fixture: torch.cuda.empty_cache()
        # Interim snapshots allow inspection after timeout; rerun recomputes this
        # seed job rather than mixing partially computed features from disk.
        write_csv(a.out/'cka.partial.csv',collected)
        print(f'Completed {m} budget={b} seed={s}',flush=True)
    write_csv(a.out/'cka.csv',collected)
    write_csv(a.out/'prediction_replay.csv',verification)
    dump(a.out/'run.json',{'fixture':fixture,'elapsed_seconds':time.perf_counter()-started,
         'gpu':None if fixture else torch.cuda.get_device_name(), 'job_id':os.getenv('SLURM_JOB_ID'),
         'versions':{'torch':torch.__version__,'transformers':transformers.__version__,
                     'datasets':datasets.__version__,'numpy':np.__version__,'python':platform.python_version()},
         'data_metadata_files_checked':checked,'models_completed':len(chosen),
         'inference_only':True,'saved_activations':False})
    (a.out/'COMPLETE').write_text('SYNTHETIC_TEST_ONLY\n' if fixture else 'C-CKA-v2 completed\n')


if __name__=='__main__':
    main()
