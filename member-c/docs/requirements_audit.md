# C requirements and scope audit

Updated: 2026-11-11 (Asia/Singapore). Primary authority: `GP/Material/project_handout.pdf`, updated 10 October 2026, PDF pages 1–4. Historical plans and repository handoffs establish team commitments, not course rules. Individual Assignment Part A/B in `Assignment/` and `GP/output/submission/` are unrelated to this Group Project.

| Class | Requirement and source | Evidence / planned deliverable | Current state and remaining gap |
|---|---|---|---|
| Course | Focused neural-network research question, baseline, motivated change; handout p1 | Common RoBERTa, fixed IMDb budgets; M1/M2/M3/M4/M6/L2-SP records | Seven-method evidence integrated in report/C_analysis.pdf and results/C-final; includes negative and unstable outcomes |
| Course | Controlled experiments, test separation, limitations and failures; p1 | A-v1/B-v1; C-v1 source-selection and prediction audit | A/B archive and M5 return independently verified; all 45 M5 candidates precede the fixed 15 source-only choices and target testing |
| Course | Experiments 40%, presentation 30%, novelty 20%, peer review 10%; p1 | Requirement coverage and report review | Contribution must be controlled adaptation/investigation, not model size or claimed invention of feature preservation |
| Course | PDF plus code ZIP; main report ≤10 pages, A4, single column, Arial 10pt; p1 | C insert-ready English sections and portable figures/code; optional C PDF | Eight-page C insert rendered (five main pages, references and two appendix pages); whole-group integration must retain the ten-page main-content limit. The course does not explicitly state report language, use English consistently with course and A/B materials |
| Course | Front-page names/group number; intro, existing techniques, methods, experiments/results, discussion, AI Use Statement; p1 | Section handoff and AI-use draft | User deferred identifying information; do not invent it. C insert is not a completed group cover page |
| Course | Commented code, dependency/data/command reproduction instructions; p1 | C README, verification and build commands | Seven-method no-Git/no-weight reproduction passed at job 12381; improved one-command workflow passed at 12382. Exact final-delivery archive acceptance is recorded separately. |
| Course | Verify AI claims/references/code, report actual team experiments, AI declaration; pp1–2 | Source ledger, tests, actual run records, declaration | Do not describe local fixtures as real runs or AI case annotations as human-coded gold labels |
| Course | Manageable compute, checkpoints, no foundation-model pretraining expected; p2 | Existing RoBERTa-base plus M5 adapters | Train M5 from original frozen base revision, not B fine-tuned weights |
| Course | Sentiment under limited labels/domain change is suggested direction C; p3 | IMDb→Yelp/Amazon study | Aligned; this is not mandatory to use a particular architecture or LoRA |
| Team | C owns M5 LoRA; `member-b/HANDOFF_C.md` opening and §3 | M5 fixed protocol, 45 candidates, 15 source-selected winners and predictions | COMPLETE: 45 candidates, 15 source-only choices, 45 prediction files, 1,170 CKA rows and 21 matched Ada cost probes; independent local prediction/statistical/diagnostic acceptance passed |
| Team | Shared data/splits/5 seeds/3 budgets/linear CLS head/FP32/10 epochs/3 candidates; same handoff §3 | M5 protocol and source-only selection ledger | Preserve these; disclose new hardware and late execution after viewing A/B results |
| Team | C statistics, CKA, errors, figures, summary and reproducibility | C-v1 and returned C-handoff-v2 | 60 B plus 15 M5 models, 9,360 CKA rows; 27 V100 and 21 Ada probes kept separate; actual evidence integrated in results/C-final |
| User | Train on RTX 6000 Ada | Isolated `sc4001-gp-c` environment and Slurm jobs | Authorized 2026-11-11; SSH restored; environment/data/base cache verified; 12366 smoke passed; all five training batches, formal evaluations and diagnostics completed; independent return accepted. Post-run evidence is in results/Ada-postrun-recovery.json and results/M5-local-audit. Existing Assignment env/results must remain unchanged |
| User | Academic prose, black-and-white text, ICML-style colored figures | Muted colorblind-friendly palette plus line/marker redundancy | New preference supersedes earlier all-black-and-white figures. No official ICML palette is asserted; retain course typography/layout |
| User | After complete first version, inspect public previous SC4001 projects and implement useful improvements | Search/source/comparability ledger and revision log | COMPLETE after first complete checkpoint: three prior same-course reports inspected; experiment/evidence guide and one-command reproduction implemented and actually validated. See PUBLIC_COURSEWORK_REVIEW.md. |
| Optional | Additional M1/M2 CKA, new datasets/backbones, new target-guided tuning | Not necessary for existing focused question | Not scheduled. Target-guided tuning disallowed; no silent expansion solely to increase method count |

## LoRA scope decision

The course does not name LoRA. The team's current authoritative handoff explicitly assigns M5 to C, and A's related-work draft also anticipates a parameter-efficient comparator. Therefore M5 is retained as required team scope. The earlier 60-model return completed the v2 supplement only; it did not complete all C responsibilities. No new result is implied by this decision.

User authorized RTX 6000 Ada rather than V100 for M5. Accuracy is evaluated under the same data/model/selection protocol, while measured time and memory must be reported in separate hardware strata. The V100 microbenchmark remains a fixed-shape computational probe, not a hardware-normalized estimator of M5 runtime. If an efficiency ranking including M5 is needed, obtain matched-hardware probes rather than compare raw seconds from different GPUs.

## Lecture connection (1-based PDF pages, not printed slide labels)

- `LEC/5_model_selection.pdf`, pp21–23: train/validation/test partition and why validation cannot provide an unbiased final estimate; pp34–39: overfitting, early checkpoint selection, regularization. Our selection remains source-only. Checkpoint selection over 10 epochs is not early termination, and no train+validation refit is introduced because A/B did not use one.
- `LEC/10_transformers.pdf`, pp4–16: encoder/BERT representations and classification; pp50–60: fine-tuning. Verify page text in the source ledger before citing a particular claim.
- `LEC/9_attention.pdf`, pp35–44: multi-head attention; LoRA modifies attention query/value projections without replacing the architecture.
- `LEC/2_regression_classification.pdf`, pp37 and 41: cross-entropy learning objective.
- `LEC/4_dnn.pdf`, p10: backpropagation; adapter gradients and frozen base parameters should be tested explicitly.

These connections explain choices; they do not imply that LoRA, CKA or the original research contribution are fully covered by the lecture slides.
