"""Portable, descriptive synthesis of frozen accuracy, CKA, ablation and cost.

Report five paired seeds; never treat layers or domains as independent model runs.
This script does not train, select parameters, or overwrite raw return files.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .common import load,dump,digest
from .collect_cka import collect
from .handoff import collect_cost


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path,default=Path('member-c/results/C-final'))
    p.add_argument('--m5',type=Path);a=p.parse_args(argv)
    a.repo=a.repo.resolve();a.out.mkdir(parents=True,exist_ok=True)
    raw=a.repo/'member-c/formal-results/return-v2';root=raw/'analysis/C-handoff-v2'
    for name,sha in load(root/'RETURN_MANIFEST.json')['files'].items():
        if digest(raw/name)!=sha:raise ValueError(f'Returned file changed: {name}')
    collect(root/'cka',a.out/'recomputed');collect_cost(root/'cost',a.out/'recomputed')
    cka=pd.read_csv(a.out/'recomputed/cka_all_seeds.csv')
    selected=pd.read_csv(a.repo/'member-b/formal-results/results/B-v1/per_seed.csv')
    if a.m5:
        extra=pd.read_csv(a.m5)
        if len(extra)!=15 or set(extra.method)!={'M5'}:raise ValueError('Incomplete M5 selected grid')
        selected=pd.concat([selected,extra],ignore_index=True)
    selected.to_csv(a.out/'selected_models.csv',index=False)
    def stats(df,keys,metric):
        return df.groupby(keys,dropna=False)[metric].agg(['mean','std','count']).reset_index()
    # Equal domain mean within seed first; individual-domain summaries retained separately.
    ood=cka[cka.domain.isin(['yelp','amazon'])].groupby(['method','reference','budget','seed','pooling','layer'],as_index=False)[
        ['linear_cka','debiased_linear_cka','shuffled_linear_cka','cosine_distance']].mean()
    ood.to_csv(a.out/'cka_ood_per_seed.csv',index=False)
    layer=stats(ood,['method','reference','budget','pooling','layer'],'linear_cka')
    layer.to_csv(a.out/'cka_layer_summary.csv',index=False)
    stats(cka,['method','reference','budget','domain','pooling','layer'],'linear_cka').to_csv(a.out/'cka_by_domain.csv',index=False)
    final=ood[(ood.layer==12)&(ood.reference=='pretrained')].merge(
        selected[['method','budget','seed','lam','ood_accuracy','imdb_accuracy']],on=['method','budget','seed'],validate='many_to_one')
    final.to_csv(a.out/'cka_accuracy_per_seed.csv',index=False)
    correlations=[]
    for (method,budget,pool),group in final.groupby(['method','budget','pooling']):
        rho=spearmanr(group.linear_cka,group.ood_accuracy).statistic
        correlations.append({'method':method,'budget':budget,'pooling':pool,'n_seeds':len(group),
            'spearman_rho_descriptive':rho,'interpretation':'Within fixed method and budget, n=5; exploratory, no causal or significance claim'})
    pd.DataFrame(correlations).to_csv(a.out/'cka_accuracy_associations.csv',index=False)
    effects=[]
    for (budget,pool),group in final.groupby(['budget','pooling']):
        reference=group[group.method=='M3'].set_index('seed')
        for method in ('M4','M6','L2SP'):
            target=group[group.method==method].set_index('seed').loc[reference.index]
            for metric in ('linear_cka','debiased_linear_cka','cosine_distance','ood_accuracy'):
                delta=(target[metric]-reference[metric]).to_numpy()
                if metric=='ood_accuracy':delta=100*delta
                effects.append({'budget':budget,'pooling':pool,'method':method,'reference':'M3','metric':metric,
                    'mean_paired_difference':delta.mean(),'sample_sd':delta.std(ddof=1),'n_seeds':len(delta),
                    'positive_seeds':int((delta>0).sum()),'negative_seeds':int((delta<0).sum()),
                    'min_difference':delta.min(),'max_difference':delta.max()})
    pd.DataFrame(effects).to_csv(a.out/'cka_paired_effects.csv',index=False)
    ablation=pd.read_csv(a.repo/'member-b/formal-results/results/B-v1/ablation.csv')
    for metric in ('ood_accuracy','imdb_accuracy','source_val_accuracy','val_feature_distance'):
        summary=stats(ablation,['method','budget','lam'],metric)
        if 'accuracy' in metric:summary[['mean','std']]*=100
        summary.to_csv(a.out/f'ablation_{metric}.csv',index=False)
    chosen=selected[selected.method.isin(['M6','L2SP'])]
    chosen.groupby(['method','budget','lam']).size().rename('selected_seeds').reset_index().to_csv(a.out/'lambda_selection_counts.csv',index=False)
    # Search/selected costs remain source-recorded; do not conflate their hardware with microbenchmarks.
    for metric in ('imdb_accuracy','yelp_accuracy','amazon_accuracy','ood_accuracy'):
        summary=stats(selected,['method','budget'],metric);summary[['mean','std']]*=100
        summary.to_csv(a.out/f'{metric}_summary.csv',index=False)
    parameters=selected.groupby('method').trainable_parameters.agg(['min','max']).reset_index()
    parameters.to_csv(a.out/'trainable_parameters.csv',index=False)
    # Comparisons of existing six methods use the earlier audited paired inference tables.
    inputs=[a.repo/'member-b/formal-results/results/B-v1/per_seed.csv',a.repo/'member-b/formal-results/results/B-v1/ablation.csv',
            root/'RETURN_MANIFEST.json',a.repo/'member-c/results/C-v1/paired_comparisons.csv']
    if a.m5:inputs.append(a.m5)
    dump(a.out/'analysis_manifest.json',{'version':'C-final-analysis-v1','scope':'Descriptive synthesis plus existing paired statistical audit',
        'M5_included':a.m5 is not None,'status':'ANALYSIS_READY_FOR_WRITING' if a.m5 else 'PARTIAL_M5_REAL_RUNS_PENDING',
        'source_hashes':{str(x):digest(x) for x in inputs},'script_sha256':digest(Path(__file__)),
        'counts':{'selected_cells':len(selected),'cka_rows':len(cka),'cka_checkpoint_cells':60,'cost_probes':27},
        'inference_units':'Five paired training seeds; two OOD domains averaged within seed, layers never independent repeats',
        'limit':'CKA is B-only for now; M5 representation/cost extension must be labeled separately if added. No test-guided selection.'})
    print('Synthesis complete; M5 included:',a.m5 is not None)


if __name__=='__main__':main()
