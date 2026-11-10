"""Black-and-white paper figures, generated only from archived numeric results."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import METHODS,BUDGETS,SEEDS

STYLES=[('o','-'),('s','--'),('^','-.'),('D',':'),('X','-'),('v','--')]


def save(fig,out,name):
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--results',type=Path,default=Path('member-c/results/C-v1'))
    p.add_argument('--repo',type=Path,default=Path('.')); a=p.parse_args()
    out=a.results/'figures'; out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':140,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
    summary=pd.read_csv(a.results/'accuracy_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.7))
    for ax,metric,title in zip(axes,['imdb_accuracy','ood_accuracy'],['IMDb (source-domain test)','OOD (Yelp/Amazon equal mean)']):
        for i,m in enumerate(METHODS):
            g=summary[(summary.method==m)&(summary.metric==metric)].sort_values('budget')
            marker,ls=STYLES[i]
            ax.errorbar(np.arange(3)+(i-2.5)*.025,g['mean'],yerr=g.sample_sd,
                        label=m,color='black',marker=marker,linestyle=ls,ms=4,capsize=2,lw=.9)
        ax.set(xticks=range(3),xticklabels=BUDGETS,xlabel='Label budget (train + validation)',
               ylabel='Accuracy (%)',title=title)
        ax.grid(axis='y',color='.9')
    axes[1].legend(ncol=3,loc='lower right',fontsize=8)
    fig.suptitle('Mean ± sample SD across five paired seeds',fontsize=10)
    fig.tight_layout(); save(fig,out,'01_accuracy')
    paired=pd.read_csv(a.results/'paired_comparisons.csv')
    fig,axes=plt.subplots(1,3,figsize=(9.2,3.4),sharey=True)
    for ax,comparison in zip(axes,['M6-M3','M4-M3','L2SP-M3']):
        g=paired[(paired.comparison==comparison)&(paired.metric=='ood_accuracy')].sort_values('budget')
        for j,(_,r) in enumerate(g.iterrows()):
            ax.scatter(np.arange(5)*.04+j-.08,[r[f'diff_seed_{s}'] for s in SEEDS],
                       facecolors='white',edgecolors='black',s=20,zorder=3)
        ax.errorbar(range(3),g['mean'],yerr=np.array([g['mean']-g.ci95_low,g.ci95_high-g['mean']]),
                    color='black',fmt='s',capsize=4,ms=4)
        ax.axhline(0,color='.5',ls='--',lw=.8)
        ax.set(title=comparison,xticks=range(3),xticklabels=BUDGETS,xlabel='Label budget')
    axes[0].set_ylabel('Paired OOD difference (percentage points)')
    fig.suptitle('Points: seed differences; bars: approximate, unadjusted 95% paired-t CI',fontsize=9)
    fig.tight_layout(); save(fig,out,'02_paired_differences')
    costs=pd.read_csv(a.results/'cost_accounting.csv')
    costmean=costs.groupby(['method','budget']).mean(numeric_only=True).reset_index()
    fig,axes=plt.subplots(1,3,figsize=(10,3.5))
    for ax,b in zip(axes,BUDGETS):
        g=costmean[costmean.budget==b].set_index('method').loc[list(METHODS)]
        ax.bar(range(6),g.incremental_search_seconds_recorded/60,color='white',edgecolor='black',label='Own search')
        ax.bar(range(6),g.upstream_search_seconds_recorded/60,bottom=g.incremental_search_seconds_recorded/60,
               color='.8',edgecolor='black',hatch='///',label='Upstream search')
        ax.set(title=f'{b} labels',xticks=range(6),xticklabels=METHODS,ylabel='Recorded search minutes / seed')
    axes[0].legend(fontsize=8)
    fig.suptitle('Standalone recorded cost; M6 at 6,400 is a lower bound after one interrupted run',fontsize=9)
    fig.tight_layout(); save(fig,out,'03_search_cost')
    errors=pd.read_csv(a.results/'error_transitions.csv')
    errors=errors[errors.comparison=='M6-M3'].copy()
    errors['fixed_pct']=100*errors.fixed_by_left/errors.n
    errors['broken_pct']=100*errors.broken_by_left/errors.n
    fig,axes=plt.subplots(1,3,figsize=(9.2,3.4),sharey=True)
    for ax,d in zip(axes,['imdb','yelp','amazon']):
        g=errors[errors.domain==d].groupby('budget').mean(numeric_only=True).loc[list(BUDGETS)]
        ax.bar(np.arange(3)-.18,g.fixed_pct,width=.36,color='white',edgecolor='black',label='M3 wrong → M6 right')
        ax.bar(np.arange(3)+.18,g.broken_pct,width=.36,color='.7',edgecolor='black',hatch='///',label='M3 right → M6 wrong')
        ax.set(title=d.upper(),xticks=range(3),xticklabels=BUDGETS,xlabel='Label budget')
    axes[0].set_ylabel('Test examples (%) — mean over seeds')
    axes[1].legend(fontsize=7)
    fig.tight_layout(); save(fig,out,'04_error_tradeoffs')
    ablation=pd.read_csv(a.repo/'member-b/formal-results/results/B-v1/ablation.csv')
    fig,axes=plt.subplots(2,3,figsize=(9.2,5.4),sharey=True)
    for i,m in enumerate(['M6','L2SP']):
        for j,b in enumerate(BUDGETS):
            ax=axes[i,j]; g=ablation[(ablation.method==m)&(ablation.budget==b)]
            agg=g.groupby('lam').ood_accuracy.agg(['mean','std'])
            ax.errorbar(range(len(agg)),agg['mean']*100,yerr=agg['std']*100,fmt='o-',color='black',capsize=3)
            ax.set(xticks=range(len(agg)),xticklabels=[f'{x:g}' for x in agg.index],xlabel='Regularization strength λ',
                   title=f'{m}, {b} labels',ylabel='OOD accuracy (%)')
    fig.suptitle('Frozen λ ablation: mean ± SD; λ=0 uses the selected M3 run',fontsize=10)
    fig.tight_layout(); save(fig,out,'05_lambda_ablation')
    perseed=pd.read_csv(a.repo/'member-b/formal-results/results/B-v1/per_seed.csv')
    fig,axes=plt.subplots(1,3,figsize=(9.2,3.4),sharey=True)
    for ax,b in zip(axes,BUDGETS):
        for m,marker in zip(['M3','M4','M6','L2SP'],['^','D','X','v']):
            g=perseed[(perseed.method==m)&(perseed.budget==b)]
            ax.scatter(g.val_feature_distance,g.ood_accuracy*100,marker=marker,facecolors='none',edgecolors='black',label=m)
        ax.set(title=f'{b} labels',xlabel='IMDb validation pooled-feature drift')
    axes[0].set_ylabel('OOD accuracy (%)'); axes[2].legend(fontsize=7)
    fig.suptitle('Descriptive association; not causal evidence or independent 60-run inference',fontsize=9)
    fig.tight_layout(); save(fig,out,'06_drift_vs_accuracy')
    if (a.results/'error_strata.csv').exists():
        strata=pd.read_csv(a.results/'error_strata.csv')
        fig,axes=plt.subplots(1,3,figsize=(9.2,3.5),sharey=True)
        for ax,d in zip(axes,['imdb','yelp','amazon']):
            for st,marker,ls in [('length_le_128','o','-'),('length_129_512','s','--'),('length_gt_512','^',':')]:
                eligible=strata[(strata.domain==d)&(strata.stratum==st)&(strata.n>=30)]
                if eligible.empty:
                    continue
                g=eligible.groupby('budget').difference_pp.agg(['mean','std']).reindex(BUDGETS)
                label={'length_le_128':'≤128','length_129_512':'129–512','length_gt_512':'>512'}[st]
                ax.errorbar(range(3),g['mean'],yerr=g['std'],label=label,marker=marker,ls=ls,color='black',capsize=2)
            ax.axhline(0,color='.6',ls='--'); ax.set(title=d.upper(),xticks=range(3),xticklabels=BUDGETS,xlabel='Label budget')
        axes[0].set_ylabel('M6 − M3 accuracy (percentage points)'); axes[0].legend(title='Tokens incl. special',fontsize=7)
        fig.suptitle('Exploratory length strata; mean ± SD; strata with <30 examples omitted from plot',fontsize=9)
        fig.tight_layout(); save(fig,out,'07_length_strata')
    print(f'Figures written to {out}')


if __name__=='__main__':
    main()
