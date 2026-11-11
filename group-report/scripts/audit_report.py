"""Check report sources, independently recompute core statistics, and inspect PDF structure.

This supplements visual review and the existing raw-prediction acceptance run;
it does not repeat model training or claim an independent raw-data reproduction.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import xml.etree.ElementTree as ET
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'group-report'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with (ROOT / path).open() as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=ROOT.parent/'output/ABC-report')
    args = parser.parse_args()
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    ledger = json.loads((REPORT/'GENERATED_VALUES.json').read_text())
    for name, expected in ledger['source_sha256'].items():
        check('Source hash: '+name, sha(ROOT/name) == expected)
    check('Generator hash', sha(REPORT/'scripts/build_report.py') == ledger['generator_sha256'])
    surfaces = '\n'.join(p.read_text() for p in (REPORT/'latex').glob('*.tex'))
    for entry in ledger['values']:
        check('Formatted numeric entry: '+entry['key'], entry['printed'] in surfaces)
    selected = rows('member-c/results/C-final/selected_models.csv')
    check('105 unique selected cells', len(selected) == len({(r['method'],r['budget'],r['seed']) for r in selected}) == 105)
    for domain in ('imdb', 'yelp', 'amazon', 'ood'):
        for r in rows(f'member-c/results/C-final/{domain}_accuracy_summary.csv'):
            values = [100*float(v[domain+'_accuracy']) for v in selected if v['method']==r['method'] and v['budget']==r['budget']]
            check(f'{domain}/{r["method"]}/{r["budget"]}: five seeds', len(values)==5)
            for key, actual in [('mean', statistics.mean(values)), ('std', statistics.stdev(values))]:
                check(f'Independent {domain}/{r["method"]}/{r["budget"]}/{key}', math.isclose(actual,float(r[key]),abs_tol=1e-10))
    for method, path in [('M6','member-c/results/C-v1/paired_comparisons.csv'),('M5','member-c/results/M5-local-audit/paired_comparisons_recomputed.csv')]:
        for r in rows(path):
            if r['comparison'] != method+'-M3' or r['metric'] != 'ood_accuracy':
                continue
            left = {v['seed']:float(v['ood_accuracy']) for v in selected if v['method']==method and v['budget']==r['budget']}
            right = {v['seed']:float(v['ood_accuracy']) for v in selected if v['method']=='M3' and v['budget']==r['budget']}
            d = [100*(left[s]-right[s]) for s in sorted(left)]
            mean = statistics.mean(d); half = float(t.ppf(.975,4))*statistics.stdev(d)/math.sqrt(5)
            for key, actual in [('mean',mean),('ci95_low',mean-half),('ci95_high',mean+half),('wins',sum(v>0 for v in d))]:
                check(f'Independent paired {method}/{r["budget"]}/{key}',math.isclose(actual,float(r[key]),abs_tol=1e-10))
    protocols = {x:json.loads((ROOT/p).read_text()) for x,p in [('A','member-a/protocol/A-v1.json'),('B','member-b/protocol/B-v1.json'),('C','member-c/protocol/M5-v1.json')]}
    for name,p in protocols.items():
        for key,value in {'budgets':[400,1600,6400], 'seeds':[13,21,42,87,100], 'epochs':10, 'max_length':512, 'batch_size':8, 'gradient_accumulation':4, 'precision':'FP32'}.items():
            check(f'{name} shared protocol {key}', p[key]==value)
    check('A rate grids', protocols['A']['learning_rates']=={'M1':[.0001,.001,.01],'M2':[.000005,.00002,.00005]})
    check('B rate grids', protocols['B']['learning_rates']=={'M3':[.00001,.00002,.00003],'M4':[.00001,.00002,.00003]})
    check('B penalty grids', protocols['B']['lambdas']=={'M6':[.01,.1,1.],'L2SP':[.1,1.,10.]})
    check('M5 rate grid', protocols['C']['learning_rates']==[.0001,.0003,.001])
    legacy = json.loads((ROOT/'member-c/DELIVERY_MANIFEST.json').read_text())
    for name,expected in legacy['files'].items():
        check('Unchanged legacy C: '+name, sha(ROOT/'member-c'/name)==expected)
    tex = (REPORT/'latex/main.tex').read_text()
    cited = {k for group in re.findall(r'\\cite\{([^}]+)\}',tex) for k in group.split(',')}
    bib = set(re.findall(r'\\bibitem\{([^}]+)\}',tex))
    check('Eight cited and resolved bibliography entries', cited==bib and len(bib)==8)
    aux = (args.build_dir/'main.aux').read_text()
    check('Main text starts at 1', bool(re.search(r'\\newlabel\{mainstart\}\{\{[^}]*\}\{1\}',aux)))
    check('Main text ends at 8', bool(re.search(r'\\newlabel\{mainend\}\{\{[^}]*\}\{8\}',aux)))
    log = (args.build_dir/'main.log').read_text()
    check('No overflow or unresolved reference warnings', not re.search(r'Overfull|LaTeX Warning:.*(?:undefined|multiply)|undefined references',log))
    pdf = REPORT/'SC4001_Group_Report.pdf'
    check('Delivered PDF equals compiled copy', sha(pdf)==sha(args.build_dir/pdf.name))
    info = subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    check('13 A4 physical pages', 'Pages:           13' in info and '(A4)' in info)
    fonts = subprocess.check_output(['pdffonts',str(pdf)],text=True)
    font_lines = fonts.splitlines()[2:]
    check('Every font is embedded', all(re.search(r'\byes\s+yes\s+yes\b',line) for line in font_lines))
    check('Arial body font embedded', 'ArialMT' in fonts)
    xml = subprocess.check_output(['pdftotext','-bbox',str(pdf),'-'])
    tree = ET.fromstring(xml)
    pages = tree.findall('.//{*}page')
    for i,page in enumerate(pages,1):
        w,h = float(page.attrib['width']),float(page.attrib['height'])
        words = page.findall('.//{*}word')
        check(f'Page {i}: text inside page', all(0 <= float(x.attrib['xMin']) < float(x.attrib['xMax']) <= w and 0 <= float(x.attrib['yMin']) < float(x.attrib['yMax']) <= h for x in words))
    result = {'status':'PASS','scope':'Source hashes, formatted-entry presence, independent seed-level means/SDs and paired intervals, protocol grids, PDF structure. Visual inspection recorded separately; not a new raw-prediction or GPU run.','checks_passed':len(checks),'numeric_entries':ledger['numeric_entries'],'source_files':len(ledger['source_sha256']),'pdf_sha256':sha(pdf),'main_pages':8,'physical_pages':13,'checks':checks}
    (REPORT/'QA.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__ == '__main__':
    main()
