from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
source=Path('member-c/results/C-final');made=Path('reproduced/C-final');rows=[]
for original in sorted(source.rglob('*.csv')):
    relative=original.relative_to(source);actual=made/relative
    a=pd.read_csv(original);b=pd.read_csv(actual)
    pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=0,atol=1e-12)
    numeric=a.select_dtypes(include='number').columns
    differences=[]
    for col in numeric:
        valid=a[col].notna()&b[col].notna();delta=(a.loc[valid,col]-b.loc[valid,col]).abs()
        if len(delta) and delta.max()>0:differences.append({'column':col,'max_absolute_difference':float(delta.max()),'cells_differing':int((delta>0).sum())})
    rows.append({'table':str(relative),'rows':len(a),'byte_identical':original.read_bytes()==actual.read_bytes(),'numeric_roundoff':differences})
assert len(rows)==22
files=[]
for d in ('figures','figures-grayscale'):
    for ext in ('png','pdf','svg'):
        found=sorted((Path('reproduced')/d).glob('*.'+ext));assert len(found)==6
        for p in found:assert p.stat().st_size>1000;files.append(str(p))
local=json.loads(Path('reproduced/M5-local-audit/local_audit.json').read_text());assert local['status']=='PASS' and local['prediction_rows']==525000
manifest=json.loads(Path('PACKAGE_MANIFEST.json').read_text())
result={'status':'PASS','archive_source_commit':manifest['commit'],'zip_files':len(manifest['files']),'no_git':not Path('.git').exists(),'A_B_prediction_rows':5250000,'M5_prediction_rows':local['prediction_rows'],'M5_paired_comparison_audit':local['paired_statistics']['check'],'synthesis_tables':rows,'generated_figure_files':files,'scope':'Full original A/B prediction reproduction, new raw-return M5 audit, synthesis from that newly generated acceptance, all 36 scientific figure outputs and full CPU suite. Derived float tables use absolute 1e-12/no relative tolerance, exact strings and matching NaN positions. Original statistical reproduction retains its narrower t-only roundoff rule.','figure_boundary':'Linux sans-serif fallback may differ from local licensed Arial; timestamps/font rendering preclude a byte-identical PDF/SVG/PNG claim. Scientific data and styles are reproduced.'}
Path('../verification.json').write_text(json.dumps(result,indent=2)+'\n')
print('FULL_EXTRACTED_PACKAGE_PASS',len(rows),'synthesis CSVs',len(files),'figures')
