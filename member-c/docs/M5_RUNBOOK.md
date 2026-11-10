# M5 Ada execution and recovery

Run from the remote project root `/share/zhangh/sc4001-gp-c/project`. This runbook describes the already submitted chain; do not submit duplicates.

```bash
source member-c/scripts/ada/activate.sh
squeue -u "$USER"
```

The isolated Python 3.10 environment pins PyTorch 2.5.1, Transformers 4.51.3, datasets 3.5.0, NumPy 1.26.4, pandas 2.2.3, accelerate 1.6.0, pyarrow 19.0.1 and safetensors 0.5.3. Model revision and dataset versions are in the frozen protocols. All caches, temporary files and outputs reside under `/share`; do not use the nearly full system filesystem for these artifacts.

## Execution contract

`smoke.sbatch` performs a two-step real-GPU compatibility check. A successful smoke must report the correct GPU, merge discrepancy at most 1e-4 and its explicit smoke completion marker. It produces no formal test results.

`formal.sbatch 0` through `formal.sbatch 4` train seeds 13, 21, 42, 87 and 100. Each stage trains budgets 400/1600/6400 at learning rates 1e-4/3e-4/1e-3, ten epochs each. Query/value rank-8 adapters, alpha 16, no extra adapter dropout, original encoder dropout active, classifier trainable, frozen backbone. Effective training batch 32, FP32, no TF32. All 45 candidates finish before the immutable 15-cell global source-validation lock is created. Candidate test performance never affects selection.

`formal.sbatch 5` evaluates only the 15 selected merged checkpoints, writes full per-example IMDb/Yelp/Amazon predictions and collects per-seed metrics. Each stage submits its successor with an `afterok` dependency. No stage silently retries a failed process.

After evaluation, `diagnostics.sbatch` verifies its own frozen source hashes and runs:

1. `python -m sc4001_c.m5_audit`: source-only selection checks, protocol/weight hashes, full test ID/label/probability/metric verification, and 18 exploratory paired OOD contrasts in a separate Holm family.
2. `python -m sc4001_c.m5_cka`: 15 models, the same 256 balanced examples per domain as C-v2, 1,170 layer/pool rows and 45 probability replays; only the pretrained reference is required.
3. Shared clean probes for M1/M2/M3/M4/M6/L2-SP plus the LoRA-specific probe, three fresh processes each on Ada. The LoRA wrapper explicitly prevents the shared benchmark from unfreezing its backbone. M5 inference is unmerged and is not used as a deployment comparison.

GPU work belongs in allocated Slurm jobs, not an interactive login session. The system currently has four allocated GPUs; do not interrupt other users' work or unrelated held jobs.

## Recovery

Inspect `member-c/logs/M5-<job>.out`, the corresponding `.err`, the candidate-specific log, and live `scontrol show job <job>` first. This server reports that historical Slurm accounting is disabled; `sacct` is not an available historical ledger. Live start/exit observations are stored in `member-c/logs/job-evidence`. A shell exit event is not relabeled as an independent Slurm terminal record, and an aged-out job is not assumed successful. Configuration and code hashes must still match the protocol. Complete candidates are verified and skipped. Incomplete runs resume only from checkpoints containing model, trainer, optimizer, scheduler and RNG state. If no valid recovery point exists, preserve the original folder and log the failed attempt before deciding on a fresh run. Do not count a resumed final invocation as the full search runtime.

Resubmit only the failed stage after diagnosing the cause; it will recreate successors after success. A formal stage consumes up to eight hours; diagnoses have a two-hour allocation. Queue waiting time is separate from allocated and active compute time. If a diagnostics failure occurs after training/evaluation completed, rerun diagnostics alone, not training.

## Return and retention

Retain all `member-c/runs/M5-v1` configuration, history, validation, parameter, invocation and source records, the global and per-cell selection files, and 45 selected prediction files. Keep the 15 selected `best/` weights on the server for replay; ordinary analysis return packages exclude their large tensors and tokenizer duplicates. Recovery checkpoints are pruned only after verified successful merged export; nonwinner weights are pruned only after every source selection is frozen. Do not prune A/B evidence.

Return `member-c/results/M5-v1`, `M5-audit`, `M5-CKA`, `Ada-cost`, small formal/smoke run records, all GP logs, available scheduler snapshots (including failures), and a SHA256 file manifest. The diagnostics job automatically creates `/share/zhangh/sc4001-gp-c/returns/M5-return-v1.tar.gz`. Its own final shell-exit record occurs after packaging and must also be retrieved separately. Successful scripts alone do not establish completion: the local audit and final report integration must pass before C is marked complete.

## Local import after the archive exists

Do not run these steps against a partial return. From the local repository root, in the analysis environment:

```bash
export PYTHONPATH=member-c/src:member-a/protocol/locked_src:member-b/protocol/locked_src
python -m sc4001_c.m5_import \
  --archive ../received/M5-return-v1.tar.gz \
  --destination ../received/M5-return-v1
python -m sc4001_c.final_analysis \
  --m5 ../received/M5-return-v1/member-c/results/M5-v1/per_seed.csv
python -m sc4001_c.final_plots
```

The importer rejects unsafe paths, duplicate/unmanifested files, changed hashes and overwriting an existing import. It independently checks all source selections, all 45 prediction files and their test IDs/labels, metrics, and correspondence of M5/B CKA samples. A fresh import destination is required for a second archive; preserve earlier returns. Local auditing does not claim to hash the weights that remain on Ada.

The current report builder deliberately rejects a results table containing M5 until the six-method draft's prose, figures and completion boundary have been revised. Updating a table alone is insufficient to produce the final report.
