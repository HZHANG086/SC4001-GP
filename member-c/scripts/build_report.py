"""Build the explicitly incomplete six-method C draft from audited source tables.

Run from the repository root in the analysis environment. Requires Tectonic (or
adapt the final subprocess to XeLaTeX) and a locally installed licensed Arial font.
Do not distribute system font files. Real M5 integration requires a prose revision.
"""
import argparse
import shutil
import subprocess
from pathlib import Path
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--tectonic',default='tectonic')
p.add_argument('--out',type=Path,default=Path('../output/C-draft'))
a=p.parse_args()
root=Path('member-c/report/latex');figures=Path('member-c/report/figures')
s=pd.read_csv('member-c/results/C-final/ood_accuracy_summary.csv')
if 'M5' in set(s.method):
    raise SystemExit('Revise the draft narrative and completion boundary for real M5 results before building a final report.')
rows=[]
for method in ('M1','M2','M3','M4','M6','L2SP'):
    g=s[s.method==method].set_index('budget')
    rows.append(method.replace('L2SP','L2-SP')+' & '+' & '.join(
        f"{g.loc[b,'mean']:.2f} $\\pm$ {g.loc[b,'std']:.2f}" for b in (400,1600,6400))+r' \\')
(root/'accuracy_table.tex').write_text(
    '\\begin{tabular}{lrrr}\\toprule\nMethod & Budget 400 & Budget 1,600 & Budget 6,400\\\\\\midrule\n'+
    '\n'.join(rows)+'\n\\bottomrule\\end{tabular}\n')
c=pd.read_csv('member-c/results/C-final/recomputed/cka_all_seeds.csv')
c=c[(c.reference=='pretrained')&(c.layer==12)]
c=c.groupby(['method','budget','domain','pooling'])[['linear_cka','cosine_distance']].mean()
metrics=pd.read_csv('member-c/results/C-v1/prediction_metrics.csv')
metrics=metrics.groupby(['method','budget','domain'])[['accuracy','brier','nll','ece15']].mean()
rows=[];calibration=[]
for b in (400,1600,6400):
    for d in ('imdb','yelp','amazon'):
        r=c.loc[('M3',b,d,'mean')];t=c.loc[('M6',b,d,'mean')]
        cls_delta=c.loc[('M6',b,d,'cls'),'linear_cka']-c.loc[('M3',b,d,'cls'),'linear_cka']
        acc_delta=100*(metrics.loc[('M6',b,d),'accuracy']-metrics.loc[('M3',b,d),'accuracy'])
        rows.append(f"{b} & { {'imdb':'IMDb','yelp':'Yelp','amazon':'Amazon'}[d] } & {r.linear_cka:.3f} & {t.linear_cka:.3f} & {cls_delta:+.3f} & {t.cosine_distance-r.cosine_distance:+.3f} & {acc_delta:+.3f}"+r' \\')
        delta=metrics.loc[('M6',b,d)]-metrics.loc[('M3',b,d)]
        calibration.append(f"{b} & { {'imdb':'IMDb','yelp':'Yelp','amazon':'Amazon'}[d] } & {delta.brier:+.4f} & {delta.nll:+.4f} & {delta.ece15:+.4f}"+r' \\')
(root/'domain_table.tex').write_text('\\begin{tabular}{rlrrrrr}\\toprule\nBudget & Domain & M3 mean CKA & M6 mean CKA & $\\Delta$ CLS CKA & $\\Delta$ cosine & $\\Delta$ acc. (pp)\\\\\\midrule\n'+'\n'.join(rows)+'\n\\bottomrule\\end{tabular}\n')
(root/'calibration_table.tex').write_text('\\begin{tabular}{rlrrr}\\toprule\nBudget & Domain & $\\Delta$ Brier & $\\Delta$ NLL & $\\Delta$ ECE\\\\\\midrule\n'+'\n'.join(calibration)+'\n\\bottomrule\\end{tabular}\n')
(root/'images').mkdir(exist_ok=True)
for name in ('02_cka_layers.pdf','03_cka_accuracy.pdf','04_lambda_ablation.pdf','05_v100_cost.pdf'):
    shutil.copy2(figures/name,root/'images'/name)
a.out.mkdir(parents=True,exist_ok=True)
subprocess.run([a.tectonic,'--keep-logs','--outdir',str(a.out),str(root/'main.tex')],check=True)
shutil.copy2(a.out/'main.pdf',a.out/'C_analysis_DRAFT.pdf')
print(a.out/'C_analysis_DRAFT.pdf')
