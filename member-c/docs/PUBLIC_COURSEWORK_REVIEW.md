# Public coursework review and implemented improvements

Reviewed 11 November 2026, after the first complete C version. The ordering is recorded in `results/complete_zip_reproduction/FIRST_COMPLETE.json`: source commit `819a69f`, eight-page PDF, real extracted-ZIP job 12381, 40 CPU tests, 5,775,000 prediction rows, 22 synthesis tables and 36 figure exports all passed before the search began. This is a qualitative comparison, not a grade prediction or a comprehensive survey.

## Search and authority

Web queries covered `github NTU SC4001 neural networks group project report`, SC4001 projects in 2024/2025, GitHub-hosted SC4001 projects, and the older CZ4042 course identifier. Repository metadata and README files were checked through GitHub; three publicly linked PDF reports were downloaded at fixed commits and their relevant sections read. No external code was executed or incorporated. No report text, figures, experimental data or results were copied into our study.

The [NTU course page](https://personal.ntu.edu.sg/asjagath/SC4001.html) identifies SC4001/CE4042/CZ4042 and AY25/26. It corroborates the course identity, but its academic year is not our 2026 handout. A publicly indexed 2026 handout mirror was also found; the user's original `GP/Material/project_handout.pdf` remains authoritative. No reviewed repository supplied a verified grade, and none is described as a full-mark exemplar. Repository dates and report metadata support the years below; they do not certify a student's enrollment semester or the exact historical assignment version.

## Comparable sources

| Source and fixed version | Identity/year evidence | What was actually inspected and how it compares |
|---|---|---|
| [IMDB sentiment project](https://github.com/yukiwukii/imdb-with-transformer/tree/1ffd21a72eab7c06123775f4494a32e95daa1777), [report](https://github.com/yukiwukii/imdb-with-transformer/blob/1ffd21a72eab7c06123775f4494a32e95daa1777/Report.pdf) | Report cover says SC4001; README identifies Task C and university-cluster execution. Repository dates March–April 2025; report references accessed in April 2025. Same-course prior sentiment project; exact handout not verified. | README maps report experiments to folders. Report §§3 and 7 discusses adaptation comparisons and unsuccessful augmentation, including split-dependence concerns. Closest task-level comparator, but its data budgets and evaluation do not establish our cross-domain results. |
| [Small Model Is All You Need](https://github.com/ztjhz/miniLM/tree/b74d927ae5486c627296059c7cf2ab0ce0fef166), [report](https://github.com/ztjhz/miniLM/blob/b74d927ae5486c627296059c7cf2ab0ce0fef166/public/Report.pdf) | README explicitly identifies NTU SC4001; report lists NTU affiliation. October–November 2023 repository activity; PDF created 9 November UTC / 10 November Singapore. Exact handout not verified. | README and report §§3.4–3.6 link environment, logging and experiments. Sentiment/model-size framing is relevant, while its slicing and training settings differ. Logging links are useful navigation; their presence alone does not prove reproducibility. |
| [Flowers102 multimodal project](https://github.com/oscarqjh/SC4001-Group-Project/tree/aa25fef8b601a49bbd1c86881a368ef32c91a0ab), [report](https://github.com/oscarqjh/SC4001-Group-Project/blob/aa25fef8b601a49bbd1c86881a368ef32c91a0ab/report/paper.pdf) | Report p1 explicitly names NTU, SC4001 and 2025; repository October–November 2025, PDF 8 November 2025. Same-course prior project, different application and unverified historical handout. | README separates quick evaluation from full training. Report §4.3 includes an unsuccessful augmentation ablation; §5.3 states limitations. Useful for organizing evidence, not a model/dataset benchmark for our sentiment task. |

The first repository is marked Apache-2.0 and the third MIT by GitHub; the third report separately states CC BY 4.0. No license was identified in the second repository metadata. These are observed labels, not an exhaustive licensing audit. Since no external implementation or text was reused, no third-party code is added to our distribution. The reports remain in a local reference cache and are excluded from the delivery ZIP.

## Decisions and actual changes

| Comparison lesson | Decision and concrete implementation |
|---|---|
| Make each reported experiment easy to locate, as in the sentiment project's experiment-to-folder index | **Implemented:** `EXPERIMENT_GUIDE.md` maps the research questions, report sections, figures, source tables and code. It also distinguishes completed GPU evidence from CPU re-analysis. |
| Separate a quick reproduction path from retraining, as in the multimodal project's README | **Implemented:** `scripts/reproduce_all.py` provides one CPU-only command after dependency installation, verifies the original predictions and new M5 return, reconstructs all tables/figures and emits per-stage logs plus a machine-readable pass/fail record. The original detailed commands remain available. It checks dependencies before creating outputs and refuses to overwrite evidence. |
| Connect findings to visible experiment records, as in miniLM's logging links | **Implemented:** the guide provides an explicit claim-to-evidence index and a minimal reviewer reading order; local immutable evidence is sufficient, without a hosted dashboard account. |
| Report failed ideas and split hazards, as in the 2025 sentiment and vision reports | **Already present, rechecked:** M6 mixed paired effects, M5 low-budget instability, source-only choices, CKA measurement boundaries, original split/overlap audits and failed reproduction attempts remain visible. No test-driven adjustment was introduced after reading these projects. |
| Compare many larger models, add unrelated augmentation or reproduce their GPU setups | **Not adopted:** would change the focused representation-preservation question and add search/compute confounds. The seven-method, three-budget, five-seed design already addresses our planned comparison. Published scores on different tasks/splits are not comparable to ours. |
| Copy a prior report's visual template or page count | **Not adopted:** only our 2026 handout governs Arial 10pt, single-column A4 and the ten-main-page limit. Large appendices or prior templates provide no exemption. |

These changes improve assessability and reproduction rather than manufacturing extra positive results. The scientific tables, fixed protocols, source-only choices, PDF numerical claims and original experiment code remain unchanged by the public-project comparison. Tests of the new wrapper and final archive are recorded with the delivery; the successful first-complete baseline remains separately preserved.

## Source fingerprints

GitHub default-branch heads were checked through the public repository API on 11 November 2026. PDF SHA256 values for the fixed report links above:

- IMDB 2025: `c8390b13765573635734d03f62822bf3ac9129aeaf8c3fb126270c0a6a953d11`.
- miniLM 2023: `2412a25d7329cb6953dc7fa83b469b48dae8063968a2e03ef44e3f152918b09d`.
- Flowers102 2025: `b6f0d9388c976a9c5abe311cb5541ffa95cbcae2e034f3fa1ca346887b5369c9`.

Additional search hits included SC4001-NLarge, another flowers project and a meta-learning sentiment project; they were screened by README/search description, not used for full-report comparisons. SC4002/SC3000 projects and unrelated SC4001 product-number results were excluded. This audit does not claim that every public SC4001 project was inspected or that absence of a feature from a README means the authors did not perform it.
