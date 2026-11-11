"""Conference-style figures with colorblind-friendly colors and grayscale redundancy."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import dump,digest

ORDER=['M1','M2','M3','M4','M5','M6','L2SP']
COLORS={'M1':'#777777','M2':'#0072B2','M3':'#222222','M4':'#009E73','M5':'#CC79A7','M6':'#D55E00','L2SP':'#E69F00'}
MARKERS={'M1':'o','M2':'s','M3':'^','M4':'D','M5':'v','M6':'P','L2SP':'X'}
LINES={'M1':':','M2':'--','M3':'-','M4':'--','M5':':','M6':'-.','L2SP':':'}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('member-c/results/C-final'))
    p.add_argument('--out',type=Path,default=Path('member-c/report/figures'));p.add_argument('--grayscale',action='store_true')
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    colors={m:'#222222' for m in ORDER} if a.grayscale else COLORS
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,
        'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':220,'axes.grid':True,'grid.alpha':.14})
    made=[]
    def save(fig,name):
        fig.tight_layout()
        for extension in ('pdf','svg','png'):
            target=a.out/f'{name}.{extension}';fig.savefig(target,bbox_inches='tight');made.append(target)
        plt.close(fig)
    models=pd.read_csv(a.data/'selected_models.csv');methods=[m for m in ORDER if m in set(models.method)]
    fig,axes=plt.subplots(1,2,figsize=(9,3.0))
    for ax,metric,title in zip(axes,['imdb_accuracy','ood_accuracy'],['IMDb (source test)','Mean of Yelp and Amazon (OOD)']):
        for m in methods:
            g=models[models.method==m].groupby('budget')[metric].agg(['mean','std'])
            ax.errorbar(np.arange(3),100*g['mean'],yerr=100*g['std'],label=m,color=colors[m],marker=MARKERS[m],linestyle=LINES[m],capsize=2,linewidth=1.2,markersize=4)
        ax.set_xticks([0,1,2],['400','1600','6400']);ax.set_xlabel('Label budget (train + validation)');ax.set_title(title)
    axes[0].set_ylabel('Accuracy (%)');axes[1].legend(ncol=2,frameon=False)
    save(fig,'01_accuracy')
    per=pd.read_csv(a.data/'cka_ood_per_seed.csv');per=per[per.reference=='pretrained']
    fig,axes=plt.subplots(2,3,figsize=(9,5.0),sharex=True,sharey=True)
    for row,pool in enumerate(('mean','cls')):
        for col,budget in enumerate((400,1600,6400)):
            ax=axes[row,col]
            for m in ('M3','M4','M6','L2SP'):
                subset=per[(per.method==m)&(per.budget==budget)&(per.pooling==pool)]
                # Thin curves expose each seed; heavy line shows the seed mean.
                for _,seed in subset.groupby('seed'):
                    ax.plot(seed.layer,seed.linear_cka,color=colors[m],alpha=.14,linewidth=.6)
                g=subset.groupby('layer').linear_cka.mean()
                ax.plot(g.index,g.values,color=colors[m],label=m,linestyle=LINES[m],marker=MARKERS[m],markevery=3,markersize=3,linewidth=1.3)
            ax.set_title(f'{"Mean pooling" if pool=="mean" else "CLS"}; budget {budget}')
            ax.set_ylim(-.03,1.04);ax.set_xticks([0,3,6,9,12])
            if row==1:ax.set_xlabel('Layer (0 = embeddings)')
            if col==0:ax.set_ylabel('CKA to pretrained')
    axes[0,-1].legend(ncol=2,frameon=False)
    save(fig,'02_cka_layers')
    joined=pd.read_csv(a.data/'cka_accuracy_per_seed.csv');fig,axes=plt.subplots(1,3,figsize=(9,3.1))
    for ax,budget in zip(axes,(400,1600,6400)):
        sub=joined[(joined.budget==budget)&(joined.pooling=='mean')]
        for m in ('M3','M4','M6','L2SP'):
            g=sub[sub.method==m];ax.scatter(g.linear_cka,100*g.ood_accuracy,label=m,c=colors[m],marker=MARKERS[m],s=24,alpha=.85)
        ax.set_title(f'Budget {budget}');ax.set_xlabel('Final-layer mean-pooled CKA');ax.set_xlim(-.03,1.03)
    axes[0].set_ylabel('OOD accuracy (%)');axes[-1].legend(ncol=2,frameon=False)
    save(fig,'03_cka_accuracy')
    ab=pd.read_csv(a.data/'ablation_ood_accuracy.csv');fig,axes=plt.subplots(1,2,figsize=(9,3.0))
    budget_colors=['#0072B2','#D55E00','#009E73'] if not a.grayscale else ['#222222']*3
    for ax,m in zip(axes,('M6','L2SP')):
        for i,b in enumerate((400,1600,6400)):
            g=ab[(ab.method==m)&(ab.budget==b)].sort_values('lam')
            ax.errorbar(range(len(g)),g['mean'],yerr=g['std'],color=budget_colors[i],marker=['o','s','^'][i],linestyle=['-','--',':'][i],label=f'{b} labels',capsize=2,linewidth=1.2)
            ax.set_xticks(range(len(g)),[f'{x:g}' for x in g.lam])
        ax.set_xlabel('Regularization weight (0 = M3 control)');ax.set_title(m)
    axes[0].set_ylabel('OOD accuracy (%)');axes[-1].legend(frameon=False)
    save(fig,'04_lambda_ablation')
    cost=pd.read_csv(a.data/'recomputed/cost_summary.csv');cost=cost[cost.profile=='clean'].set_index('method').loc[[m for m in ORDER if m in set(cost[cost.profile=='clean'].method)]]
    fig,axes=plt.subplots(1,2,figsize=(9,3.0))
    for ax,key,scale,label in zip(axes,['train_peak_allocated_bytes_mean','seconds_per_optimizer_step_mean_mean'],[2**30,1],['Peak allocated memory (GiB)','Seconds per optimizer step']):
        bars=ax.bar(cost.index,cost[key]/scale,color=[colors[m] for m in cost.index],edgecolor='black',linewidth=.5)
        for bar,hatch in zip(bars,['','/','..','//','xx','\\']):bar.set_hatch(hatch)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=8);ax.set_ylabel(label);ax.set_ylim(0,(cost[key]/scale).max()*1.2)
    fig.suptitle('V100 fixed-shape microbenchmark; M4 = stage-2 compute proxy',fontsize=10)
    save(fig,'05_v100_cost')
    dump(a.out/'figure_manifest.json',{'style':'Restrained colorblind-friendly conference-style; not an official ICML palette',
        'colors':colors,'grayscale':a.grayscale,'uncertainty':'Accuracy/ablation: five-seed sample SD. CKA: thin individual seed curves, bold mean. Scatter: one point per seed with OOD domains averaged.',
        'status':'Preliminary until required M5 results integrated','files':{str(p):digest(p) for p in made},'source_script_sha256':digest(Path(__file__))})
    print('Generated',len(made),'figure files')


if __name__=='__main__':main()
