# Full A/B/C report integration

User request: consolidate the current GitHub A/B/C material into a single report under the original school requirements. User confirmed on this turn that identifying information is temporarily omitted; the cover will contain write-in fields rather than invented names/group numbers. Existing scientific choices, English academic style, black body text and restrained scientific-figure colors are retained.

## Source snapshot

- GitHub `branch-a`: bf1c015cc7557aa9086293a0718e7b093e8c255b; its member-a subtree is identical to the C integration base.
- GitHub `branch-b`: f442a4462364ce0c53cbae45febbfcfc8238988c; its member-b subtree is identical to the C integration base.
- GitHub `member-c-analysis-v1`: 71da6c0d4bc03c3e209a3a845450b3300f2d52ea; complete actual experimental evidence and final-archive acceptance.
- GitHub `main`: 79325396076920acacb04f3501c4d806c5232341; minimal project description, not the source of formal results.
- Authority: GP/Material/project_handout.pdf, 10 October 2026; A4, one column, Arial 10pt, at most ten main pages, explicit AI Use Statement, named front page. We use eight planned main pages plus cover, references and optional appendices.

This is integration of an already confirmed study and completed experimental evidence, not a new research/motivation intake. Academic-writing role prompts are applied inline. The full PaperSpine new-paper pipeline is not invoked: reopening topic selection or imposing unrelated journal deliverables would not serve the course report requested here.

## Claim and section plan

| Unit | Purpose and evidence | Writing decision / boundary |
|---|---|---|
| Whole report | Does preserving a pretrained encoder improve limited-label cross-domain sentiment transfer, at what cost? | One argument across A/B/C; contribution is a controlled adaptation/investigation, not invention of LoRA or feature preservation. |
| Abstract and introduction | Existing seven-method, three-budget, five-seed evidence | State the question, controlled design, main negative result and cost trade-off; do not promise full marks. |
| Related work | Verified original RoBERTa, IMDb, review-dataset, LP-FT, LoRA, L2-SP, DELTA and CKA sources | Separate prior vision evidence from our sentiment results; no unsupported novelty or transfer guarantee. |
| Methods | A/B locked code, M5-v1, parameter counts | Define shared linear CLS head; distinguish mean-pooled regularization from classifier features; give losses and full adaptation controls. |
| Data and experimental protocol | A audit/splits, A-v1/B-v1/M5-v1 | Explain label-budget accounting, duplicates, target balance, dynamic padding, source-only choices, candidate grids and hardware chronology. |
| Accuracy and uncertainty | C-final per-domain summaries; paired/bootstrap tables | Include A baselines and all seven methods; no universal one-percentage-point significance rule; no rank-to-superiority inference. |
| Representation analysis | 9,360 CKA rows with pooled/CLS, debiasing, shuffle control | Preserve pooling mismatch, between-seed variation and no causal interpretation. |
| Ablation and failure analysis | Fixed lambda sweep, deterministic errors/strata | Explain negative results, nonselected ablation candidates and heterogeneous classes/lengths; no target-based reselection. |
| Cost and practical interpretation | Matched Ada probes, separate V100/full-run records | Distinguish trainable parameters, fixed-shape allocated memory, step time, search and deployment. |
| Discussion, conclusion and AI statement | All above; actual recorded assistance | State supported insight and limits. Report contributions by A/B/C artifact role; names are user-deferred. Do not attest unperformed human review or unknown B tool use. |
| Optional appendices | Full domain means/SDs, extended diagnostics, provenance/reproduction | Main results remain in the main body; appendices add detail rather than carry essential evidence. |

## Conflicts resolved

- A's REPORT_A_DRAFT is historical; use the audited formal A results and DATA_AUDIT instead of its pending-job language.
- B's approximate heuristic that sub-one-point differences are not significant is not a statistical decision rule; use the audited paired analysis and explicit five-seed uncertainty.
- B's older M4 ranking predates M5. Use all seven methods: the medium-budget M5 mean exceeds M4 by only 0.012 points.
- B's suggested V100 requirement for M5 was superseded by the user's actual Ada authorization. Keep full training on separate hardware strata and compare costs through actual matched Ada probes.
- The original C PDF is an insert. Remove integration reminders and rewrite introduction/methods/conclusion for a unified group report; never concatenate PDFs.

No frozen member artifacts or experimental values are edited. New work lives in group-report/ on report-integration-v1. Final QA will check source values, citations, all rendered pages, main-page count and actual GitHub publication.
