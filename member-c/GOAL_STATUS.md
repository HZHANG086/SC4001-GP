# C completion goal status

Updated 2026-11-11, Asia/Singapore. ACTIVE; not complete.

- Previous goal turn: concrete progress (audited return, draft report, LoRA/diagnostic pipeline, commit 017cbe6). This turn: real GPU progress verified; operational accounting issue repaired and tested.
- Requirements/lecture audit: `docs/requirements_audit.md`. Six-method data/statistics/CKA/cost analysis and 7-page academic draft are available. The draft has five main pages, one reference page and one appendix, and excludes unfinished M5.
- Ada smoke job 12366 succeeded on RTX 6000 Ada; two steps, merge discrepancy 4.47e-8. Formal job 12367 is RUNNING. Latest saved snapshot: `results/Ada-live-status.json`, 6/45 complete candidates, 0/15 formal test evaluations. The old November 12 queue estimate is superseded.
- Training remains on its original frozen M5-v1 protocol. No B weights needed. All 45 candidates and 15 source-only decisions precede target testing. Existing A/B and historical C evidence are unchanged.
- Server historical Slurm accounting is disabled. Operational amendment adds live snapshots, start/exit events for future stages, and nonfatal accounting capture in return packaging. Stage 0 was already submitted and keeps its old shell script; externally observed status plus its existing completion log will be retained. Original diagnostic protocol preserved alongside amendment.
- Complete local suite: 31 tests passed; `results/validation_31_tests.log`. This includes actual offline training/resume fixtures, preservation of frozen backbone during probes, corrupted-return rejection and packaging with disabled accounting. Tests are not formal GPU results.
- Current entry point: `CURRENT.md`; runtime/recovery guide: `docs/M5_RUNBOOK.md`. Remote root `/share/zhangh/sc4001-gp-c/project`. Do not restart a live job or touch unrelated held jobs.
- Still required: finish real M5 training/test/CKA/matched cost, retrieve and independently audit return, integrate actual results into prose/figures/statistics, create first complete package, then public previous-coursework search and useful revisions, final requirement/runtime/PDF audit.
- User deferred group identifiers. Academic English/black text; restrained colorblind-friendly plots plus grayscale versions; course A4/single-column/Arial 10pt takes precedence over conference templates.

No course-system submission or teammate message authorized. Do not mark the goal complete while required runs/deliverables remain unfinished.
