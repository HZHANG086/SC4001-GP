# Experiment and evidence guide

## Five-minute reading order

1. Read `../CURRENT.md`, then `../report/C_analysis.pdf`: the C insert contains five main pages, references and two appendix pages.
2. Use the question-to-evidence table below to locate the numbers behind a claim. All paths in the table are relative to `member-c/`.
3. Install `../requirements-final.txt` from an isolated environment and, from the repository/extracted-package root, run:

   ```bash
   python member-c/scripts/reproduce_all.py --out reproduced/C-complete
   ```

   This performs the full 40-test CPU suite and offline re-analysis, writes six stage logs and `verification.json`, and refuses an existing output directory. It needs no Git history, network download, model weights or GPU after dependencies are installed. It does not train or re-run inference. Allow several minutes; the package contains 5,775,000 prediction records. The tested server used four CPU threads and a 16 GB RAM allocation; this is a tested allocation, not a measured minimum.

4. Use `REPRODUCTION.md` for individual commands and `M5_RUNBOOK.md` for actual GPU retraining. Use `REPORT_INTEGRATION.md` for the group report's page budget, actual contribution review and AI statement.

## Question-to-evidence map

| Question / report location | Evidence | Computation / figure |
|---|---|---|
| Are comparisons controlled? §1 | `protocol/M5-v1.json`; A/B frozen protocols; `results/C-v1/source_file_hashes.csv`; `results/M5-local-audit/local_audit.json` | `src/sc4001_c/reproduce.py`, `m5_import.py`; source-only choices, fixed test IDs/labels and all prediction metrics checked |
| How does adaptation affect accuracy across budgets? §2 | `results/C-final/{imdb,yelp,amazon,ood}_accuracy_summary.csv`; `selected_models.csv` | `final_analysis.py`; Table 1 and supplementary `report/figures/01_accuracy.*`; means and SDs over five seeds |
| Does M6 reliably improve on full tuning? §§2,5 | `results/C-v1/paired_comparisons.csv`, `bootstrap_sensitivity.csv`, `error_stability.csv` | Frozen `analyze.py` through `reproduce.py`; paired effects, intervals and bootstrap sensitivity, including negative results |
| What is LoRA's added evidence? §§2,4,5 | `formal-results/M5-return-v1`; `results/M5-local-audit/paired_comparisons_recomputed.csv`; `results/Ada-stage4-check.json` | 45 candidates, 15 source-selected winners, 45 prediction files; `m5_import.py` re-audits 18 paired contrasts; no B weight needed to train M5 |
| Does preserving representation predict better transfer? §3 and Appendix B | `results/C-final/cka_layer_summary.csv`, `cka_by_domain.csv`, `cka_ood_per_seed.csv`, `cka_paired_effects.csv`, `cka_accuracy_associations.csv` | `collect_cka.py`, `final_analysis.py`; Figs. 1 and 4 / `02_cka_layers.*`, `03_cka_accuracy.*`; mean/CLS pooling, debiasing and shuffled control retained |
| Is the regularizer choice stable? §3 | `results/C-final/ablation_*.csv`, `lambda_selection_counts.csv`; B's original candidate records | Fig. 2 / `04_lambda_ablation.*`; fixed λ grids shown without selecting λ on target results |
| Where do errors occur? §3 | `results/C-v1/error_strata.csv`, `error_transitions.csv`, `error_cases_with_excerpts.csv`, `docs/case_annotations.json` | `analyze.py`, `texts.py`; labels/length/lexical strata and deterministic cases; AI case commentary is not human-coded ground truth |
| What does adaptation cost? §4 and Appendix B | `results/C-final/ada_cost_summary.csv`; `results/C-final/recomputed/`; `results/C-v1/cost_accounting.csv`, `cost_caveats.json`; M5 recorded selected/search times in `formal-results/M5-return-v1/member-c/results/M5-v1/per_seed.csv` | Figs. 3 and 5 / `06_ada_cost.*`, `05_v100_cost.*`; V100/Ada separate, three cost processes separate from five training seeds, fixed-shape probes separate from full training/deployment |

The plotting implementation is `src/sc4001_c/final_plots.py`. Every figure is available in PDF/SVG/PNG in both `report/figures` and `report/figures-grayscale`. The figure numbers in the PDF follow narrative order; numbered file stems identify the original plotting outputs. The return contains expected undefined CKA values for constant layer-0 CLS states; these are not missing experiments.

## Claims a reviewer should challenge

- M6's higher mean-pooled CKA does not imply better accuracy. Its OOD differences from M3 change sign across budgets, and uncertainty remains large.
- M5 saves measured resources relative to M3 on the same Ada setup, but is more expensive than M2 and unstable at the smallest budget. It is not a universal winner.
- The five seeds are paired replications. Domains, layers, examples and timing processes must not be counted as extra independent trained models. Non-significance does not establish equivalence.
- No target-based configuration change followed the public-coursework review. Its practical changes are navigation and reproduction; the original research idea and results predate that review.

## Contribution and handoff boundary

Existing A materials supply the frozen data protocol and M1/M2 evidence; B supplies M3/M4/M6/L2-SP and the actual GPU supplement. C adds audited statistical/representation/error/cost analysis, M5 implementation and real Ada runs, matched seven-method cost probes, figures, report sections and portable reproduction. These are artifact responsibilities, not a certification of individual human effort or peer-review scores. The three members must confirm their actual contributions and AI use themselves.

The delivery covers C and preserves A/B; the team still assembles a single coherent group report, adds its deferred identifiers, reviews the AI declaration and submits through the course system. Repeating completed GPU jobs or downloading B's approximately 29 GB of weights is unnecessary for reviewing the included C statistics. Repeating B representation extraction requires those checkpoints or retraining, as documented separately.
