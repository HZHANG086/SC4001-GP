"""Build the integrated report from unchanged, audited A/B/C evidence.

This script generates tables/macros and their value-to-source ledger; it does not
train a model, select configurations or change the frozen member artifacts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'group-report'
TEX = REPORT / 'latex'
METHODS = ('M1','M2','M3','M4','M5','M6','L2SP')
BUDGETS = (400,1600,6400)
claims = []
inputs = {}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def csv(relative):
    p = ROOT / relative
    inputs[relative] = digest(p)
    return pd.read_csv(p)


def record(key, value, printed, source):
    claims.append({'key': key, 'value': float(value), 'printed': printed, 'source': source})
    return printed


def tab(name, columns, header, rows):
    (TEX/name).write_text('\\begin{tabular}{'+columns+'}\\toprule\n'+header+r'\\\midrule'+'\n'+'\n'.join(row+r'\\' for row in rows)+'\n\\bottomrule\\end{tabular}\n')


def escape(s):
    table={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(table.get(c,c) for c in str(s))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tectonic',default='tectonic')
    p.add_argument('--out',type=Path,default=ROOT.parent/'output/ABC-report')
    p.add_argument('--tables-only',action='store_true')
    a=p.parse_args();TEX.mkdir(parents=True,exist_ok=True)
    base='member-c/results/C-final/'
    metrics={d:csv(base+d+'_accuracy_summary.csv').set_index(['method','budget']) for d in ('imdb','yelp','amazon','ood')}
    assert all(set(x.index.get_level_values(0))==set(METHODS) for x in metrics.values())
    def cell(d,m,b):
        r=metrics[d].loc[m,b];assert int(r['count'])==5
        mean=record(f'{d}/{m}/{b}/mean',r['mean'],f"{r['mean']:.2f}",base+d+'_accuracy_summary.csv')
        sd=record(f'{d}/{m}/{b}/SD',r['std'],f"{r['std']:.2f}",base+d+'_accuracy_summary.csv')
        return mean+r' $\pm$ '+sd
    rows=[m.replace('L2SP','L2-SP')+' & '+' & '.join(cell(d,m,b) for b in BUDGETS for d in ('imdb','ood')) for m in METHODS]
    header=r'& \multicolumn{2}{c}{Budget 400} & \multicolumn{2}{c}{Budget 1,600} & \multicolumn{2}{c}{Budget 6,400}\\ Method & IMDb & OOD & IMDb & OOD & IMDb & OOD'
    tab('accuracy_table.tex','lrrrrrr',header,rows)
    rows=[f'{b:,} & '+m.replace('L2SP','L2-SP')+' & '+' & '.join(cell(d,m,b) for d in ('imdb','yelp','amazon','ood')) for b in BUDGETS for m in METHODS]
    tab('full_domain_table.tex','rlrrrr','Budget & Method & IMDb & Yelp & Amazon & OOD',rows)
    params=csv(base+'trainable_parameters.csv').set_index('method')
    descriptions={
        'M1':'Frozen encoder; train linear head only.',
        'M2':'Tune final two encoder blocks and linear head.',
        'M3':'Tune the complete encoder and linear head.',
        'M4':'Head-only first stage, then full fine-tuning (LP-FT).',
        'M5':'Query/value LoRA in all layers, plus linear head.',
        'M6':'Full tuning with pooled cosine feature preservation.',
        'L2SP':'Full tuning with encoder weight anchoring.'}
    rows=[]
    for m in METHODS:
        v=int(params.loc[m,'min']);assert v==int(params.loc[m,'max'])
        text=record(m+'/trainable',v,f'{v:,}',base+'trainable_parameters.csv')
        rows.append(m.replace('L2SP','L2-SP')+' & '+text+' & '+descriptions[m])
    tab('methods_table.tex',r'lrp{105mm}','Method & Trainable parameters & Adaptation',rows)
    source='member-a/remote-records/data-v1/audit.json';audit=json.loads((ROOT/source).read_text());inputs[source]=digest(ROOT/source)
    rows=[]
    for key,label in [('imdb_train_pool','IMDb source pool'),('imdb','IMDb test'),('yelp','Yelp test'),('amazon','Amazon test')]:
        r=audit['datasets'][key];values=[record(key+'/n',r['n'],f"{r['n']:,}",source),record(key+'/median',r['token_length_median'],str(int(r['token_length_median'])),source),record(key+'/p95',r['token_length_p95'],str(int(r['token_length_p95'])),source),record(key+'/trunc512',100*r['truncation_rate_512'],f"{100*r['truncation_rate_512']:.2f}",source)+r'\%']
        rows.append(label+' & '+' & '.join(values))
    tab('data_table.tex','lrrrr',r'Dataset & Examples & Median tokens & 95th percentile & $>512$ tokens',rows)
    tab('search_table.tex','lp{61mm}p{63mm}','Method & Three candidates per cell & Additional selection dependency',[
        r'M1 & Learning rate: $10^{-4},10^{-3},10^{-2}$ & None',
        r'M2 & Learning rate: $5\!\times\!10^{-6},2\!\times\!10^{-5},5\!\times\!10^{-5}$ & None',
        r'M3 / M4 & Learning rate: $10^{-5},2\!\times\!10^{-5},3\!\times\!10^{-5}$ & M4 retrains a selected-rate head first',
        r'M5 & Learning rate: $10^{-4},3\!\times\!10^{-4},10^{-3}$ & Rank 8; alpha 16; query and value',
        r'M6 & $\lambda=0.01,0.1,1$ & M3 selected learning rate',
        r'L2-SP & $\lambda=0.1,1,10$ & M3 selected learning rate'])
    original=csv('member-c/results/C-v1/paired_comparisons.csv')
    lora=csv('member-c/results/M5-local-audit/paired_comparisons_recomputed.csv')
    rows=[]
    for label,table,source in [('M6--M3',original,'member-c/results/C-v1/paired_comparisons.csv'),('M5--M3',lora,'member-c/results/M5-local-audit/paired_comparisons_recomputed.csv')]:
        for b in BUDGETS:
            r=table[(table.comparison==label.replace('--','-'))&(table.budget==b)&(table.metric=='ood_accuracy')].iloc[0]
            values=[record(f'{label}/{b}/{c}',r[c],f'{r[c]:+.3f}',source) for c in ('mean','ci95_low','ci95_high')]
            rows.append(f'{label} & {b:,} & {values[0]} & [{values[1]}, {values[2]}] & {int(r.wins)}/5')
    tab('paired_table.tex','lrrcr','Comparison & Budget & Mean difference & 95\\% paired interval & Wins',rows)
    macros={}
    def macro(name,value,fmt,source):
        macros[name]=record(name,value,format(value,fmt),source)
    for d,m,b,name,fmt in [('ood','M5',6400,'OODLoRAHigh','.3f'),('ood','M2',6400,'OODPartialHigh','.3f'),('ood','M3',6400,'OODFullHigh','.3f')]:
        macro(name,metrics[d].loc[(m,b),'mean'],fmt,base+d+'_accuracy_summary.csv')
    for b,suffix in zip(BUDGETS,('Low','Mid','High')):
        macro('MtwoMinusMone'+suffix,metrics['ood'].loc[('M2',b),'mean']-metrics['ood'].loc[('M1',b),'mean'],'.3f',base+'ood_accuracy_summary.csv')
    macro('LoRAvsLPMid',metrics['ood'].loc[('M5',1600),'mean']-metrics['ood'].loc[('M4',1600),'mean'],'.3f',base+'ood_accuracy_summary.csv')
    macro('LoRASDLow',metrics['ood'].loc[('M5',400),'std'],'.3f',base+'ood_accuracy_summary.csv')
    selected=csv(base+'selected_models.csv')
    failure=selected[(selected.method=='M5')&(selected.budget==400)&(selected.seed==42)].iloc[0]
    macro('LoRAFailure',100*failure.ood_accuracy,'.2f',base+'selected_models.csv')
    macro('LoRATrainingHours',selected[selected.method=='M5'].search_train_seconds_recorded.sum()/3600,'.3f',base+'selected_models.csv')
    # Report-only sensitivity analysis of previously audited predictions.
    f1_sources=['member-c/results/C-v1/prediction_metrics.csv',
                'member-c/results/M5-local-audit/prediction_metrics_recomputed.csv']
    per_domain=pd.concat([csv(path) for path in f1_sources],ignore_index=True)
    assert len(per_domain)==315 and not per_domain.duplicated(['method','budget','seed','domain']).any()
    per_seed=per_domain[per_domain.domain.isin(['yelp','amazon'])].groupby(['method','budget','seed'])[['accuracy','macro_f1']].mean()*100
    sensitivity=per_seed.groupby(['method','budget']).agg(['mean','std'])
    sensitivity.columns=['_'.join(c) for c in sensitivity.columns]
    assert len(sensitivity)==21
    for b in BUDGETS:
        cell=sensitivity.xs(b,level='budget')
        assert cell.accuracy_mean.sort_values().index.tolist()==cell.macro_f1_mean.sort_values().index.tolist()
    evidence=REPORT/'evidence';evidence.mkdir(exist_ok=True)
    sensitivity.to_csv(evidence/'metric_sensitivity.csv')
    macro('FoneMaxGap',(sensitivity.accuracy_mean-sensitivity.macro_f1_mean).abs().max(),'.3f','; '.join(f1_sources))
    cost=csv(base+'ada_cost_summary.csv').set_index('method')
    assert (cost.peak_allocated_gib_sd==0).all()
    macro('CostSDMin',cost.seconds_per_step_sd.min(),'.5f',base+'ada_cost_summary.csv')
    macro('CostSDMax',cost.seconds_per_step_sd.max(),'.5f',base+'ada_cost_summary.csv')
    for m,prefix in [('M1','Probe'),('M2','Partial'),('M3','Full'),('M5','LoRA'),('M6','Feature'),('L2SP','Weight')]:
        macro(prefix+'Memory',cost.loc[m,'peak_allocated_gib_mean'],'.3f',base+'ada_cost_summary.csv')
        macro(prefix+'Step',cost.loc[m,'seconds_per_step_mean'],'.3f',base+'ada_cost_summary.csv')
    for field,saving,vs in [('peak_allocated_gib_mean','MemorySaving','MemoryVsPartial'),('seconds_per_step_mean','TimeSaving','TimeVsPartial')]:
        macro(saving,100*(1-cost.loc['M5',field]/cost.loc['M3',field]),'.1f',base+'ada_cost_summary.csv')
        macro(vs,100*(cost.loc['M5',field]/cost.loc['M2',field]-1),'.1f',base+'ada_cost_summary.csv')
    (TEX/'numbers.tex').write_text('\n'.join('\\newcommand{\\'+name+'}{'+value+'}' for name,value in macros.items())+'\n')
    c=csv(base+'recomputed/cka_all_seeds.csv');c=c[(c.reference=='pretrained')&(c.layer==12)].groupby(['method','budget','domain','pooling'])[['linear_cka','cosine_distance']].mean()
    prediction=csv('member-c/results/C-v1/prediction_metrics.csv').groupby(['method','budget','domain'])[['accuracy','brier','nll','ece15']].mean()
    rows=[];cal=[]
    for b in BUDGETS:
        for d in ('imdb','yelp','amazon'):
            source=base+'recomputed/cka_all_seeds.csv';r=c.loc['M3',b,d,'mean'];t=c.loc['M6',b,d,'mean']
            values=[r.linear_cka,t.linear_cka,c.loc[('M6',b,d,'cls'),'linear_cka']-c.loc[('M3',b,d,'cls'),'linear_cka'],t.cosine_distance-r.cosine_distance,100*(prediction.loc[('M6',b,d),'accuracy']-prediction.loc[('M3',b,d),'accuracy'])]
            texts=[record(f'domain/{b}/{d}/{i}',v,format(v,'.3f' if i<2 else '+.3f'),source if i<4 else 'member-c/results/C-v1/prediction_metrics.csv') for i,v in enumerate(values)]
            rows.append(f"{b:,} & { {'imdb':'IMDb','yelp':'Yelp','amazon':'Amazon'}[d]} & "+' & '.join(texts))
            delta=prediction.loc['M6',b,d]-prediction.loc['M3',b,d]
            texts=[record(f'calibration/{b}/{d}/{metric}',delta[metric],format(delta[metric],'+.4f'),'member-c/results/C-v1/prediction_metrics.csv') for metric in ('brier','nll','ece15')]
            cal.append(f"{b:,} & { {'imdb':'IMDb','yelp':'Yelp','amazon':'Amazon'}[d]} & "+' & '.join(texts))
    tab('domain_table.tex','rlrrrrr',r'Budget & Domain & M3 CKA & M6 CKA & $\Delta$ CLS CKA & $\Delta$ cosine & $\Delta$ acc. (pp)',rows)
    tab('calibration_table.tex','rlrrr',r'Budget & Domain & $\Delta$ Brier & $\Delta$ NLL & $\Delta$ ECE',cal)
    cases=csv('member-c/results/C-v1/error_cases_with_excerpts.csv');rows=[]
    for category in ('fixed_by_left','broken_by_left'):
        row=cases[(cases.budget==400)&(cases.seed==13)&(cases.domain=='imdb')&(cases.category==category)].iloc[0]
        snippet=' '.join(str(row.text_excerpt).split()[:28])+' ...'
        desc='Fix' if category=='fixed_by_left' else 'Regression'
        labels='/'.join('pos' if int(row[c]) else 'neg' for c in ('label','M3_prediction','M6_prediction'))
        rows.append(desc+r'\newline '+escape(row.example_id)+' & '+labels+' & '+escape(snippet))
    tab('cases_table.tex',r'p{36mm}p{25mm}p{93mm}','Record & Gold/M3/M6 & Opening excerpt',rows)
    for protocol in ('member-a/protocol/A-v1.json','member-b/protocol/B-v1.json','member-c/protocol/M5-v1.json'):
        inputs[protocol]=digest(ROOT/protocol)
    (TEX/'images').mkdir(exist_ok=True)
    for name in ('02_cka_layers.pdf','03_cka_accuracy.pdf','04_lambda_ablation.pdf','05_v100_cost.pdf','06_ada_cost.pdf'):
        source=ROOT/'member-c/report/figures'/name;shutil.copy2(source,TEX/'images'/name)
        inputs[str(source.relative_to(ROOT))]=digest(source)
    (REPORT/'GENERATED_VALUES.json').write_text(json.dumps({'status':'GENERATED_FROM_AUDITED_SOURCES','numeric_entries':len(claims),'values':claims,'source_sha256':inputs,'generator_sha256':digest(Path(__file__))},indent=2)+'\n')
    if a.tables_only:return
    a.out.mkdir(parents=True,exist_ok=True)
    environment=os.environ.copy()
    environment.setdefault('TECTONIC_CACHE_DIR',str(ROOT.parent/'.cache-c/tectonic'))
    subprocess.run([a.tectonic,'--keep-logs','--keep-intermediates','--outdir',str(a.out.resolve()),str(TEX/'main.tex')],cwd=ROOT,env=environment,check=True)
    shutil.copy2(a.out/'main.pdf',a.out/'SC4001_Group_Report.pdf')
    shutil.copy2(a.out/'SC4001_Group_Report.pdf',REPORT/'SC4001_Group_Report.pdf')
    print('Integrated report:',a.out/'SC4001_Group_Report.pdf')


if __name__=='__main__':
    main()
