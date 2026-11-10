# Reproducing C results

Run commands from the project/package root. A model download, GPU or Git history is **not needed to recompute statistics from the included predictions**. Training or repeating representation extraction is a different operation and does require the appropriate data/model weights and an allocated GPU.

## Environment

Python 3.10-3.12 is supported by the tested setup. Install `member-c/requirements-final.txt` in an isolated environment. GPU training used Python 3.10.12, PyTorch 2.5.1 and Transformers 4.51.3; local analysis used Python 3.12.5 with the pinned NumPy/SciPy versions. Select a compatible CPU/CUDA PyTorch build for your platform.

```bash
python -m pip install -r member-c/requirements-final.txt
export PYTHONPATH=member-c/src:member-a/protocol/locked_src:member-b/protocol/locked_src
```

The GPU host initially lacked SciPy. Version 1.15.3 was added with `--no-deps` for post-training statistical auditing, without changing any training-library version. Candidate environment records retain their actual package inventories; earlier candidates legitimately omit this unused training dependency.

## Offline evidence and statistics

```bash
python -m sc4001_c.reproduce --out reproduced/C-v1
python -m sc4001_c.final_analysis --out reproduced/C-final
python -m sc4001_c.final_plots --data reproduced/C-final --out reproduced/figures
python -m sc4001_c.final_plots --data reproduced/C-final --grayscale --out reproduced/figures-grayscale
```

The first command is the portable entry point for an extracted ZIP. It checks the 61-file historical C manifest and 1,879 archived input hashes, recomputes source-only choices, reads all 450 A/B prediction files (5,250,000 rows), and reconstructs seven statistical/error/bootstrap tables. It refuses to overwrite the frozen reference results. The original `sc4001_c.analyze` entry point additionally checks the Git upstream revision and is retained unchanged for a full checkout.

Cross-platform checking preserves exact input hashes and exact result fields. Only approximate Student-t probabilities (`p_t_approx`, `p_t_approx_holm`) and t-interval endpoints (`ci95_low`, `ci95_high`) allow an absolute difference of at most 1e-12, with every differing cell recorded. A real Linux test revealed two approximate p-values differing from the Mac reference by 5.55e-17; the earlier byte-only checker correctly raised a difference but was too strict for portable numerical reproduction. The failed attempt and reviewed differences are retained. Predictions, accuracies, sample counts, exact sign-flip results and other fields remain strict; no frozen result or statistical calculation was changed.

`final_analysis` verifies the 244-file GPU return manifest and re-aggregates the 60-model CKA and 27 V100 cost probes. Figure files are exported as PDF, SVG and PNG. Accuracy/ablation error bars are five-seed SDs; cost-process repetitions do not represent additional training seeds.

A separate no-Git/no-weight copy has executed all four commands: seven historical statistics tables and 21 synthesis CSVs reproduced byte-for-byte; all ten color/grayscale PNGs matched exactly. The 30 PDF/SVG/PNG exports were generated successfully. Evidence is recorded in `results/standalone_reproduction.json`. PDF/SVG timestamps or identifiers can vary between runs.

**Current M5 boundary:** M5 is still training. The commands above reproduce the completed six-method evidence, not a final seven-method report. Once the real M5 return is available, run the separate import and audit commands in `M5_RUNBOOK.md`, then pass its `per_seed.csv` with `--m5` to `final_analysis`. An independent local prediction audit is required before M5 is accepted. The final package instructions must be updated to name the actual included return location.

## Reconstructing text strata and cases

The package includes source-linked error-stratum and case tables, not the entire review corpus. To recompute text features, download the public test data at the frozen revisions:

```bash
python -m sc4001_c.texts --cache cache/test-texts --out reproduced/C-v1
```

Run the offline reproduction first, since it supplies the deterministic case-ID file. IMDb IDs, Yelp/Amazon fixed indices, label mappings and Amazon title/newline/content concatenation must remain unchanged. No new target split or model selection occurs. To reuse an already validated text cache, add `--join-only`. Negation markers are lexical heuristics, and case interpretations are AI-assisted drafts rather than human-coded gold labels.

## Repeating GPU computations

The exact A/B preparation, training and evaluation code is preserved under their `protocol/locked_src` directories. Their READMEs describe complete retraining and source-only selection. Recomputing B CKA requires the 60 selected B checkpoints (or recreating them); the small return package records their hashes and probability replays but does not include those weight files.

M5 uses the original pinned RoBERTa base, not a B checkpoint. Follow `M5_RUNBOOK.md` and the frozen M5 protocol. Do not submit duplicate jobs when the existing chain is live. M5 merged winners can be loaded by the original A inference class without adapter-specific inference code. No weights or full datasets belong in the ordinary code/report delivery archive.

## PDF build and validation

The source is `member-c/report/latex/main.tex`; the course format is A4, one column and Arial 10pt. Tectonic 0.15.0 was used locally; XeLaTeX is an alternative. Arial must be installed/licensed on the build machine. Local font copies used by the desktop preview are excluded from Git and delivery. The document embeds font subsets in the PDF.

```bash
python member-c/scripts/build_report.py --tectonic /path/to/tectonic
```

The current builder produces the explicitly incomplete six-method draft and refuses a table containing M5 until the prose has been revised. PDF export alone is not acceptance: render all pages, verify tables/labels/references and confirm the whole group's main content stays within ten pages. The course also requires the group's own identification, integration and AI-use review; these cannot be inferred from code execution.

## Exporting and checking the code ZIP

From a clean, committed checkout, export all tracked code and experiment evidence:

```bash
python member-c/scripts/export_code.py --out ../output/SC4001-code-snapshot.zip
```

The exporter includes the archived per-example predictions needed for offline statistics. It rejects tracked weights, recovery tensors, caches and font binaries, records file sizes/SHA256 hashes and the source commit, then re-reads every archive member to verify its bytes. It does not overwrite an existing delivery. Keep generated archives outside the checkout.

The ZIP can be checked without Git using the included script after extraction:

```bash
python member-c/scripts/export_code.py --verify /path/to/SC4001-code-snapshot.zip
```

Compare the external `.zip.sha256` value with the delivered archive before extraction. `PACKAGE_MANIFEST.json` describes the archived version and points to the current-status and reproduction instructions. An export is a versioned snapshot, not a certificate that M5, the report, the later public-coursework review or the overall goal has finished. The final deliverable must be rebuilt from the completed, audited version; do not submit the explicitly incomplete preflight snapshot.

Actual preflight evidence: `results/zip_preflight/PREFLIGHT.json`. Archive version `05b37e3` contains 4,132 files (79,739,964 bytes); Linux CPU job 12371 fully recomputed 450 prediction files/5,250,000 rows and all seven tables after extraction. It used Python 3.10.12, NumPy 1.26.4, pandas 2.2.3 and SciPy 1.15.3. Both the first failed byte-only attempt and the successful rerun are retained. This check excludes unfinished M5 results and is not the final submission package.

## CPU verification tests

With the PYTHONPATH above, run `python -m unittest discover -s member-c/tests -v`. The updated 37-test suite passed in an allocated CPU-only Linux job; `results/validation-37-tests` records raw logs, source hashes and scope. Tests use small synthetic fixtures for training/resume, CKA, costs, archive integrity, roundoff handling and paired-statistics re-execution. Fixture output must never be presented as a formal GPU experiment.
