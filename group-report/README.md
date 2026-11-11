# Integrated SC4001 group report

Read **[SC4001_Group_Report.pdf](SC4001_Group_Report.pdf)** for the unified A/B/C report. It supersedes the individual draft reports as the group narrative; the original member artifacts remain unchanged.

The PDF has 13 physical pages: cover, 8 main pages, 1 reference page and 3 optional appendix pages. It follows the 10 October 2026 school handout: A4, single column, Arial 10-point body text, at most 10 main pages. Body text is black; scientific figures use restrained colors together with distinguishable markers, line styles or hatching. Complete grayscale figures remain in `member-c/report/figures/`.

**Before submission:** fill in the group number and three names (left blank at the user's request). Each member must review the interpretation, confirm their contributions and add any additional AI tools/purposes to section 11. This is a completed integrated report with those team-owned details deferred, not a claim that the team has already approved the declaration. Submit the PDF and code ZIP through NTULearn yourselves.

The grading-focused revision is documented in [评分标准复查与修改记录](RUBRIC_REVIEW_ZH.md). The [evidence map](EVIDENCE_MAP.md) links the unified report sections to A/B/C sources. Additional macro-F1 sensitivity results are in `evidence/metric_sensitivity.csv`.

## What is included

- `latex/main.tex`: editable report text, references, captions and cover.
- `latex/*_table.tex`, `latex/numbers.tex`: generated numerical tables and macros.
- `latex/images/`: five existing scientific figures copied without alteration.
- `scripts/build_report.py`: generate tables and compile the report.
- `scripts/audit_report.py`: independently recompute core seed-level summaries and paired intervals, check source hashes/protocols, and check PDF structure.
- `GENERATED_VALUES.json`: 394 formatted numeric entries and 22 input-file hashes. Repeated display of the same value is counted as another entry, not another independent claim.
- `QA.json`: automated audit result; `INTEGRATION_REVIEW.md`: visual and content review with explicit limitations.
- `CITATION_LEDGER.md`: original-source links and the claims they support.
- `INTEGRATION_PLAN.md`: source branch commits and resolved A/B/C draft conflicts.

## Rebuild the report

Run commands from the repository or extracted code root. Python 3.10–3.12 and the existing `member-c/requirements-final.txt` environment are supported. Report generation uses pandas; the independent numerical audit also uses SciPy. Install Tectonic and Poppler (`pdfinfo`, `pdffonts`, `pdftotext`) separately. For the required font, install licensed Arial on the build machine, or place your licensed `Arial.ttf`, `Arial Bold.ttf`, `Arial Italic.ttf` and `Arial Bold Italic.ttf` in `group-report/latex/fonts/`. Font binaries are deliberately excluded from Git and code exports.

```sh
python group-report/scripts/build_report.py --tectonic tectonic --out output/ABC-report
python group-report/scripts/audit_report.py --build-dir output/ABC-report
```

The build copies the compiled PDF to `group-report/SC4001_Group_Report.pdf` and the requested output directory. Tectonic may download missing TeX packages on its first run. Set `TECTONIC_CACHE_DIR` if a different writable cache is needed. To regenerate only the numerical tables, use `--tables-only`.

After editing prose or the cover, rebuild and repeat the audit and visual inspection of affected pages. Do not assume the current page count survives substantial additions.

## Reproduce the underlying study

Follow [the complete reproduction guide](../member-c/docs/REPRODUCTION.md), including dependencies, data preparation and GPU workflows. Offline analysis of archived predictions needs no model weights:

```sh
python member-c/scripts/reproduce_all.py --out reproduced/C-complete
```

Training and extracting new representations require the actual model/data and GPU workflow. The report does not imply that weights are in this repository or that another GPU run is needed to recompute archived results.

To export an associated code/evidence ZIP from a clean committed checkout:

```sh
python member-c/scripts/export_code.py --out ../SC4001_ABC_Code.zip
python member-c/scripts/export_code.py --verify ../SC4001_ABC_Code.zip
```

This exports all tracked A/B/C materials plus the report and checksums, without Git history, fonts, model weights or caches. The report integration branch leaves `main` and the member branches untouched.
