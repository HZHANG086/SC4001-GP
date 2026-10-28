# Remaining evaluation: ready to submit, not yet submitted

Current verified state: all 90 training candidates completed; all 30 source-only selections completed; 28/30 final test evaluations completed. Job 70089 reached its 05:50 wall limit at 2026-10-28 02:31 Singapore time.

Both remaining winner models exist: M2/b6400/seed87 and M2/b6400/seed100. The frozen sweep entry skips completed training and completed test metrics, preserves the partial test folder, evaluates the remaining models and generates the aggregate report. No model, data, LR, seed or length changes are required.

After authorization, submit on NTU from /tc1home/UG/tsun003/sc4001-a:

```bash
sbatch --time=01:00:00 --job-name=sc4001-finish-eval scripts/formal.sbatch 87 100 --finish
```

Completion still requires results/A-v1/COMPLETE and validated records for all 30 unique selected cells. No task has been resubmitted by this status check.

Update 2026-10-28: authorized and submitted as job70230; completed successfully in00:10:19. All30 evaluations and aggregation are now complete and locally verified. Do not resubmit.
