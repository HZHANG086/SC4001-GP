"""Reconstruct ONLY the fixed test texts, then join predictions for error analysis.

Downloads public dataset test parquet files at A's locked commits, not model weights.
Text characteristics are descriptive strata, not manually verified linguistic labels.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from .common import (BUDGETS,SEEDS,DOMAINS,digest,load,dump,read_csv,write_csv,
                     selected_rows,run_path,read_predictions)

NEGATION = re.compile(r"\b(?:no|not|never|neither|nor|hardly|without)\b|n['’]t\b", re.I)


def reconstruct(repo, cache, out):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    sources=load(repo/'member-a/remote-records/data-v1/sources.json')
    tokenizer=AutoTokenizer.from_pretrained(sources['model_id'],revision=sources['model_revision'],
                                           cache_dir=str(cache/'hub'))
    files={}
    for domain in DOMAINS:
        dest=cache/f'{domain}_texts.jsonl'
        folder='amazon_polarity' if domain=='amazon' else 'plain_text'
        ds=load_dataset(sources['datasets'][domain],revision=sources['revisions'][domain],
                        data_files={'test':f'{folder}/test-00000-of-00001.parquet'},split='test',
                        verification_mode='no_checks',cache_dir=str(cache/'datasets'))
        expected={'imdb':25000,'yelp':38000,'amazon':400000}[domain]
        if len(ds)!=expected:
            raise ValueError(f'{domain}: unexpected dataset size')
        expected_names={'imdb':['neg','pos'],'yelp':['1','2'],'amazon':['negative','positive']}[domain]
        if ds.features['label'].names!=expected_names:
            raise ValueError('Unexpected source label mapping')
        indices=list(range(25000)) if domain=='imdb' else load(
            repo/f'member-a/remote-records/data-v1/{domain}_test_indices.json')
        selected=ds.select(indices)
        with dest.open('w') as stream:
            for start in range(0,len(selected),128):
                batch=selected[start:start+128]
                texts=batch['text'] if domain!='amazon' else [t+'\n'+c for t,c in zip(batch['title'],batch['content'])]
                tokens=tokenizer(texts,truncation=False,verbose=False)['input_ids']
                for i,text,y,tok in zip(indices[start:start+128],texts,batch['label'],tokens):
                    record={'example_id':f'{domain}:test:{i}','label':y,'text':text,
                            'token_length':len(tok),'truncated_at_512':len(tok)>512,
                            'negation_marker':bool(NEGATION.search(text))}
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        files[domain]={'path':str(dest),'sha256':digest(dest),'n':len(indices),
                       'dataset_revision':sources['revisions'][domain]}
        print(f'Reconstructed {domain}: {len(indices)} texts',flush=True)
    dump(out/'test_text_sources.json',{'files':files,'sources':sources,
                                      'definition':'IMDb unchanged; Yelp/Amazon selected by A fixed indices; Amazon title + newline + content'})


def analyze(repo, cache, out):
    rows=selected_rows(repo); bykey={(r['method'],r['budget'],r['seed']):r for r in rows}
    texts={}
    for d in DOMAINS:
        with (cache/f'{d}_texts.jsonl').open() as stream:
            texts[d]={x['example_id']:x for x in (json.loads(line) for line in stream)}
    transitions=[]; audit={}; classrows=[]
    for domain in DOMAINS:
        ds=texts[domain]
        base=run_path(repo,bykey['M3',400,13])/'test'/f'{domain}_predictions.jsonl'
        ids,y,_,_=read_predictions(base)
        if set(ids)!=set(ds) or any(ds[i]['label']!=int(label) for i,label in zip(ids,y)):
            raise ValueError(f'Text/prediction ID or label mismatch: {domain}')
        lengths=np.array([ds[i]['token_length'] for i in ids])
        neg=np.array([ds[i]['negation_marker'] for i in ids])
        strata={'all':np.ones(len(ids),bool),'length_le_128':lengths<=128,
                'length_129_512':(lengths>128)&(lengths<=512),'length_gt_512':lengths>512,
                'negation_marker':neg,'no_negation_marker':~neg,
                'negative_label':y==0,'positive_label':y==1}
        audit[domain]={'n':len(ids),'labels_match':True,'positive_n':int(y.sum()),
                       'truncated_n':int((lengths>512).sum()),'text_file_sha256':digest(cache/f'{domain}_texts.jsonl')}
        for b in BUDGETS:
            for s in SEEDS:
                _,_,p3,q3=read_predictions(run_path(repo,bykey['M3',b,s])/'test'/f'{domain}_predictions.jsonl',ids,y)
                _,_,p6,q6=read_predictions(run_path(repo,bykey['M6',b,s])/'test'/f'{domain}_predictions.jsonl',ids,y)
                for label,mask in strata.items():
                    if not mask.any():
                        continue
                    c3,c6=p3[mask]==y[mask],p6[mask]==y[mask]
                    transitions.append({'budget':b,'seed':s,'domain':domain,'stratum':label,'n':int(mask.sum()),
                                        'M3_accuracy':float(c3.mean()),'M6_accuracy':float(c6.mean()),
                                        'difference_pp':float(100*(c6.mean()-c3.mean())),
                                        'fixed':int((c6&~c3).sum()),'broken':int((c3&~c6).sum())})
    cases=read_csv(out/'case_sample_ids.csv')
    for r in cases:
        t=texts[r['domain']][r['example_id']]
        import hashlib
        r.update(text_excerpt=' '.join(t['text'].split()[:40]),
                 text_sha256=hashlib.sha256(t['text'].encode()).hexdigest(),
                 token_length=t['token_length'],truncated_at_512=t['truncated_at_512'],
                 negation_marker=t['negation_marker'],human_review='PENDING_TEAM_REVIEW')
    write_csv(out/'error_cases_with_excerpts.csv',cases)
    write_csv(out/'error_strata.csv',transitions)
    dump(out/'text_join_audit.json',audit)
    print('All 35,000 original test texts joined; labels agree with archived predictions.',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,default=Path('.'))
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('member-c/results/C-v1'))
    p.add_argument('--join-only',action='store_true')
    a=p.parse_args(); a.cache.mkdir(parents=True,exist_ok=True); a.out.mkdir(parents=True,exist_ok=True)
    if not a.join_only:
        reconstruct(a.repo,a.cache,a.out)
    analyze(a.repo,a.cache,a.out)


if __name__=='__main__':
    main()
