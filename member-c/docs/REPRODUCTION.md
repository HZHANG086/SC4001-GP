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
python -m sc4001_c.m5_import \
  --raw member-c/formal-results/M5-return-v1 \
  --out reproduced/M5-local-audit
python -m sc4001_c.final_analysis \
  --m5 member-c/formal-results/M5-return-v1/member-c/results/M5-v1/per_seed.csv \
  --m5-audit reproduced/M5-local-audit/local_audit.json \
  --out reproduced/C-final
python -m sc4001_c.final_plots --data reproduced/C-final --out reproduced/figures
python -m sc4001_c.final_plots --data reproduced/C-final --grayscale --out reproduced/figures-grayscale
```

The first command is the portable entry point for an extracted ZIP. It checks the 61-file historical C manifest and 1,879 archived input hashes, recomputes source-only choices, reads all 450 A/B prediction files (5,250,000 rows), and reconstructs seven statistical/error/bootstrap tables. It refuses to overwrite the frozen reference results. The original `sc4001_c.analyze` entry point additionally checks the Git upstream revision and is retained unchanged for a full checkout.

Cross-platform checking preserves exact input hashes and exact result fields. Only approximate Student-t probabilities (`p_t_approx`, `p_t_approx_holm`) and t-interval endpoints (`ci95_low`, `ci95_high`) allow an absolute difference of at most 1e-12, with every differing cell recorded. A real Linux test revealed two approximate p-values differing from the Mac reference by 5.55e-17; the earlier byte-only checker correctly raised a difference but was too strict for portable numerical reproduction. The failed attempt and reviewed differences are retained. Predictions, accuracies, sample counts, exact sign-flip results and other fields remain strict; no frozen result or statistical calculation was changed.

`m5_import --raw` verifies all 809 M5 return-manifest files, independently audits 45 prediction files/525,000 rows and all 15 source-only decisions, and recomputes the 18 M5 contrasts. It rejects incomplete or changed raw directories and writes outside that immutable input. `final_analysis` uses this newly generated audit, verifies the original 244-file B manifest and complete M5 manifest, and combines 60 B models plus 15 M5 models for CKA. The 27 V100 and 21 Ada cost probes remain separate. Figure files are exported as PDF, SVG and PNG. Accuracy/ablation error bars are five-seed SDs; cost-process repetitions do not represent additional training seeds.

An earlier six-method no-Git/no-weight check executed its four analysis/plot commands: seven historical statistics tables and 21 synthesis CSVs reproduced byte-for-byte; all ten color/grayscale PNGs matched exactly. The 30 PDF/SVG/PNG exports were generated successfully. Evidence is recorded in `results/standalone_reproduction.json`. PDF/SVG timestamps or identifiers can vary between runs.

**Accepted M5 return:** `member-c/formal-results/M5-return-v1`. The original 8,084,794-byte archive has SHA256 `7812944d024e5ddd3d321d3884ea3c6c24e743c203d12083426661b7a59f2efa`; its raw files are included in the code ZIP, so the original tar archive is not needed for offline reproduction. Local audit tables reproduced the 18 returned statistical rows exactly. Server-side CKA adds 1,170 rows, 45 probability replays and 45 expected undefined layer-0 CLS values. No weights are needed for this numerical re-audit.

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

The builder requires all seven methods and the accepted M5-inclusive synthesis. It outputs `../output/C-final/C_analysis.pdf` and `member-c/report/C_analysis.pdf`. The revised narrative includes real accuracy, uncertainty, CKA and hardware-matched costs. PDF export alone is not acceptance: render all pages, verify tables/labels/references and confirm the whole group's main content stays within ten pages. The course also requires the group's own identification, integration and AI-use review; these cannot be inferred from code execution.

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

Compare the external `.zip.sha256` value with the delivered archive before extraction. `PACKAGE_MANIFEST.json` describes the archived version and points to the current-status and reproduction instructions. An export is a versioned snapshot, not by itself a certificate of scientific completeness or team review. The final deliverable must be rebuilt from the completed, audited version; do not submit the explicitly incomplete preflight snapshot.

Actual preflight evidence: `results/zip_preflight/PREFLIGHT.json`. Archive version `05b37e3` contains 4,132 files (79,739,964 bytes); Linux CPU job 12371 fully recomputed 450 prediction files/5,250,000 rows and all seven tables after extraction. It used Python 3.10.12, NumPy 1.26.4, pandas 2.2.3 and SciPy 1.15.3. Both the first failed byte-only attempt and the successful rerun are retained. This check excludes unfinished M5 results and is not the final submission package.

## CPU verification tests

With the PYTHONPATH above, run `python -m unittest discover -s member-c/tests -v`. The updated 40-test suite passed in an allocated CPU-only Linux job; `results/validation-40-tests` records raw logs, source hashes and scope. Tests use small synthetic fixtures for training/resume, CKA, costs, archive integrity, roundoff handling and paired-statistics re-execution. Fixture output must never be presented as a formal GPU experiment.

For an already extracted M5 return, `m5_import --raw` performs the same independent numerical audit after checking the complete file inventory and manifest hashes; see M5_RUNBOOK.md. This avoids shipping a redundant tar archive inside the final code ZIP. The M5-inclusive extracted-package check is recorded separately in the delivery evidence; its status must be assessed from actual execution, not this interface description.
