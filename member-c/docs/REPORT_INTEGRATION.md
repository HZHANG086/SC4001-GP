# Integrating the C analysis into the group report

All seven methods and planned M5 diagnostics are complete and independently audited. This guide describes integration into the group submission; it does not certify the group cover, member review or final submission. The course handout governs the submission: A4, single column, Arial 10pt, at most ten main-content pages, with references/contents/cover/appendices excluded from that limit. Do not use a two-column ICML template merely because the figures use a restrained conference-style palette.

## One scientific question and consistent terminology

Use the shared question: **Does preserving pretrained representations improve out-of-domain sentiment classification under limited source labels, and at what computational cost?** Treat the project's contribution as a controlled investigation. LoRA, LP–FT, L2-SP and feature regularization are established ideas; neither using a larger model nor implementing them makes them new inventions.

- “Label budget” includes training and validation: 400 = 320 + 80; 1,600 = 1,280 + 320; 6,400 = 5,120 + 1,280.
- “OOD accuracy” is the equal mean of Yelp and Amazon accuracy within a seed, followed by the five-seed mean/SD. IMDb is source-domain test accuracy.
- M1: linear probe; M2: final-two-block tuning; M3: full tuning; M4: LP–FT; M5: LoRA; M6: mean-pooled feature preservation; L2-SP: weight-preservation control.
- Five paired training seeds are the replication unit. Different layers, domains, examples or repeated timing processes do not increase that number.
- A source-validation-selected checkpoint is the best epoch among ten trained epochs. This is not a claim that training stopped early.

## Section ownership and proposed space

| Main-report content | Material and editing responsibility | Indicative space |
|---|---|---:|
| Research question and motivation | A/B/C agree on the question above; introduce source-to-target transfer and the representation hypothesis | 0.75 page |
| Existing techniques and positioning | A's `RELATED_WORK.md`, verified against C's reference ledger; B describes the relation to LP–FT and preservation methods | 0.75 page |
| Data, methods and common training protocol | A's data audit and frozen A protocol; B's frozen method definitions; C's actual M5 protocol | 1.5 pages |
| Evaluation protocol, results and analysis | C accuracy, paired uncertainty, CKA, ablation, errors and cost, including the actual M5 results | 4–5 pages |
| Discussion, limitations and conclusion | Joint scientific interpretation, including negative results and evidence boundaries | 0.75 page |
| AI Use Statement and final formatting margin | Actual assistance and team verification; no invented human-review assertions | 0.25 page plus remaining space |

These are editing targets, not a guarantee of the final page count. The C analysis insert has five main pages and repeats some common-method material for standalone readability. Remove that duplication when incorporating it; do not simply concatenate three PDFs. Keep detailed source hashes, full domain/seed tables and extended diagnostics in the appendix or code package, with clear references from the main text. Essential negative results and qualifications belong in the main text.

## Which version to use

`member-a/REPORT_A_DRAFT.md` explicitly labels itself as historical. Use its verified data/method descriptions, but replace its old “results pending” and job-status paragraphs with the actual audited A results. Do not turn the development pilot into a formal result. B's `HANDOFF_C.md` defines the M5 commitment and input interfaces; its original subset-only CKA suggestion has since been extended to all 60 selected B checkpoints through the accepted return.

For C, start from `CURRENT.md`, `report/C_analysis.pdf`, `report/latex/main.tex` and `results/C-final`. M5 is now included using its accepted real return. The historical C `README.md`, `DELIVERY_MANIFEST.json` and `results/C-v1` stay unchanged as provenance. Seven-method final tables must come from the accepted real M5 return, never estimated values or a manually inserted row.

## Claims that need careful wording

1. Higher mean-pooled CKA for M6 is evidence of representation preservation, not evidence of better generalization or a causal mechanism. Its existing M6–M3 accuracy differences are mixed and seed-sensitive.
2. Failure to reject a difference with five seeds does not establish equivalence. Keep the bootstrap sensitivity and the exact-test resolution limitation visible.
3. M4's extra first stage and inherited searches for M6/L2-SP matter. Equal candidate counts do not make total compute equal. Preserve the resumed-run lower-bound qualification.
4. V100 and Ada time/memory measurements stay in separate hardware groups. Matched Ada microprobes can compare training-step costs; they do not measure complete search cost or trained-model deployment throughput.
5. M5 was designed after A/B results were available. State this chronology; its own configuration and source-only selection are frozen before its target tests. Do not describe the overall analysis as prospectively preregistered.
6. Error-case comments are AI-assisted interpretations. Review the 18 discordant cases in `docs/case_annotations.json` against their source-linked excerpts before adopting particular semantic interpretations. If not reviewed, retain that status or omit the qualitative claims; do not certify review that did not occur.

## Human information still required at final group submission

The user deferred group number, names and student identifiers. Add the real information to the group cover when available. Each member must check their actual contribution, understand the full argument, and verify the final AI Use Statement. C can provide the code, evidence, draft and consistency checks; an automated audit cannot attest that a teammate personally read or understood them. Course-system submission is outside the current authorization.
