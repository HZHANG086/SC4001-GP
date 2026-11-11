# Unified report: questions, experiments and evidence

Paths below are relative to the repository root. This maps the **full group report**, not the older C-only insert.

| Report unit | Question / comparison | Inspect first | Code / protocol |
|---|---|---|---|
| §§3–4, Tables 1–3 | What is fixed, and what changes? | `member-a/remote-records/data-v1/audit.json`; `member-c/results/C-final/trainable_parameters.csv` | `member-a/protocol/A-v1.json`; `member-b/protocol/B-v1.json`; `member-c/protocol/M5-v1.json` |
| §5, Tables 4–5 | Does preservation improve source-only-selected target accuracy? | `member-c/results/C-final/selected_models.csv`; per-domain `*_accuracy_summary.csv`; `member-c/results/C-v1/paired_comparisons.csv`; `member-c/results/M5-local-audit/paired_comparisons_recomputed.csv` | `member-c/scripts/reproduce_all.py`; independent report checks in `group-report/scripts/audit_report.py` |
| §5 sensitivity | Is the descriptive ordering specific to accuracy? | `group-report/evidence/metric_sensitivity.csv` | `group-report/scripts/build_report.py`: domain-to-seed-to-method aggregation; `audit_report.py` recomputes independently |
| §5 uncertainty sensitivity | Which inference assumptions matter? | `member-c/results/C-v1/bootstrap_sensitivity.csv` | `member-c/src/sc4001_c/bootstrap.py` |
| §6, Figure 1 / Appendix B | Which representation is preserved? | `member-c/results/C-final/cka_layer_summary.csv`; `recomputed/cka_all_seeds.csv`; `cka_accuracy_per_seed.csv` | `member-c/src/sc4001_c/cka.py`; `collect_cka.py`; `m5_cka.py` |
| §7, Figure 2 | How do penalty strength and source-only choices affect results? | `member-c/results/C-final/lambda_selection_counts.csv`; `member-c/results/C-v1/` analysis and original B candidate records | `member-b/protocol/locked_src/sc4001_b/objectives.py`; `member-c/src/sc4001_c/final_plots.py` |
| §7.1 / Appendix C | Where do predictions change, and are fixes stable across seeds? | `member-c/results/C-v1/error_strata.csv`; `error_stability.csv`; `error_cases_with_excerpts.csv` | `member-c/scripts/reproduce_all.py`; documented deterministic case selection |
| §8, Figure 3 / Appendix C | Does parameter efficiency translate into measured cost savings? | `member-c/results/C-final/ada_cost_summary.csv`; `recomputed/cost_all_repeats.csv`; `member-c/results/C-v1/cost_accounting.csv` | matched Ada and separate V100 probe records; `member-c/src/sc4001_c/final_plots.py` |
| §11 / Appendix C | Can another person recompute the archived findings? | `member-c/docs/REPRODUCTION.md`; `member-c/docs/FINAL_ACCEPTANCE.md`; `group-report/QA.json` | `member-c/scripts/reproduce_all.py`; `member-c/scripts/export_code.py` |

For quick checking, read §§3–5, compare the two focal methods in Table 5 with Figure 1, then inspect Figure 3 and §9's limitations. The code ZIP includes the raw per-example prediction evidence. Offline analysis requires no model weights; actual retraining and representation extraction require data/models and the documented GPU environment.

The macro-F1 check is a later descriptive sensitivity analysis using existing predictions. It is not a new training trial, a test-guided model-selection rule, or an extra independent replication.
