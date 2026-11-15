# SC4001 Group 111: final submission

Preserving Pretrained Representations under Limited Labels: A Controlled Study of Cross-Domain Sentiment Classification.

- [Report (PDF)](SC4001_Group_Report.pdf) — public copy with student IDs omitted; 10 main-text pages, 16 pages including the cover, references and appendices.
- [Code (ZIP)](SC4001_Code.zip) — compact training, evaluation and analysis package, frozen protocols, summary tables and CPU tests.

Sun Tingyao (A), Gao Xuran (B), and Zhang Heng (C).

## Reproduce the packaged checks

Use Python 3.10 for GPU reproduction. The CPU acceptance run used a fresh Python 3.12.5 environment with the pinned dependencies. After extracting the ZIP, keep the `member-a`, `member-b` and `member-c` directory names unchanged.

```sh
cd SC4001_Code
python -m pip install -r member-c/requirements-final.txt
python check_results.py
python run_tests.py
```

The archive README includes the data-preparation, training and evaluation commands. The CPU tests use synthetic data and tiny models, with no GPU or pretrained-model download. Full prediction archives and model checkpoints are outside this compact ZIP.

## Acceptance on 15 November 2026

A fresh extraction passed all 70 CPU tests (A: 2; B: 15; C: 40; pooling: 6; additional seeds: 6; report: 1). The archived-table check reproduced the summaries for 105 selected models, seven focal pooling contrasts and 16 A100 paired contrasts. All 149 frozen source-hash checks passed. These checks do not constitute a new formal GPU experiment.

The report clarifies the 6,400-label LoRA result, supplement contributions, AI use, and the scope of the compact package. The local course-submission PDF includes student IDs; this public PDF does not.
