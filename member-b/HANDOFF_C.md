# Handoff from Member B to Member C

The team plan assigns Member C: **M5 (LoRA), statistics, representation analysis (CKA), error analysis, figures, summary tables, reproducibility records and the experimental-analysis section.** This page lists everything from A and B that those tasks depend on: what is in the repository, what is not, and how to get it.

The results themselves are described in [formal-results/README.md](formal-results/README.md): files, columns, analysis rules and findings. Read that first.

## 1. What is in the repository

| Need | Where |
|---|---|
| Per-seed results of all six existing methods (M1–M4, M6, L2-SP) | `formal-results/results/B-v1/per_seed.csv` (A's rows included) |
| λ ablation | `formal-results/results/B-v1/ablation.csv` |
| Per-example test predictions | `formal-results/runs/B-v1/<run>/test/*_predictions.jsonl` (B) and `../member-a/formal-results/runs/A-v1/<run>/test/` (A) |
| Frozen selection of every cell | `formal-results/runs/B-v1/selection-*.json` |
| Per-epoch training curves | `formal-results/runs/B-v1/<run>/history.json` |
| Exact config, package versions and source of every run | `run_config.json`, `environment.txt`, `source/` in each run folder |
| Protocol and code hashes | `protocol/B-v1.json`, `protocol/locked_src/` (B); `../member-a/protocol/A-v1.json` (A) |
| SLURM job ledger | `formal-results/job-ledger.tsv` (18 jobs) and `formal-results/logs/` |
| Data indices (training splits, Yelp/Amazon test indices) | `../member-a/remote-records/data-v1/` |

## 2. What is not in the repository, and how to get it

### Model weights (needed for CKA and other representation analysis)

| Models | Status | Location |
|---|---|---|
| B's 60 selected models (M3, M4, M6 and L2-SP × 3 budgets × 5 seeds) | Kept, 0.5 GB each, 29 GB in total | TC1, Member B's account: `runs/B-v1/<winner>/best/`. The winner names are in `selection-*.json`. **Ask Member B for access.** |
| A's 30 selected models (M1, M2) | Kept according to A-v1's retention rule | Member A's TC1 account. Ask Member A. |
| Non-selected candidates, and M4 stage-1 models | **Deleted** by protocol to stay within storage; configs, metrics and histories are kept | Not available. Retraining is required if ever needed. |
| Pretrained reference model | Public | `FacebookAI/roberta-base`, revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b` |

Load a selected model with Member A's class. The weights are a plain `LinearRoberta` (`model.roberta` is the encoder, `model.classifier` the linear head on the CLS state):

```python
from sc4001_a.model import LinearRoberta
from transformers import AutoTokenizer
model = LinearRoberta.from_pretrained('<run>/best', use_safetensors=True).eval()
tok = AutoTokenizer.from_pretrained('<run>/best')
```

### Data (needed for M5 training, CKA inputs and error analysis)

- The prepared data `artifacts/data-v1/` (816 MB) is not committed. Rebuild it on TC1 with Member A's prepare job (see `../member-a/README.md`, "prepare.sbatch --source-lock remote-records/data-v1/sources.json"). It should match A-v1's manifest; `sc4001_b.sweep` contains a file-by-file check (`check_data_matches_a`) you can reuse.
- Contents:
  - `imdb_test` (25,000 examples), `yelp_test` (5,000) and `amazon_test` (5,000), each with columns `text`, `label` and `example_id`;
  - `imdb_train`;
  - `splits/b{budget}_s{seed}.json`: 15 train/validation index files.
- **Error analysis:** the `example_id` in the prediction files (for example `imdb:test:0`) is the same field as in these test sets, so join on it to get the review text.

### Records that do not exist by design

- **Validation-set predictions:** not saved. Each run keeps only the validation accuracy of its chosen checkpoint and `val_feature_distance`.
- **Test results of non-selected M3 and M4 candidates:** never produced, because only winners are tested. The exception is the M6 and L2-SP λ candidates, which were all tested as the pre-registered ablation.

## 3. M5 (LoRA): what makes it comparable

To sit in the same tables as M1–M6, M5 follows the same protocol as A and B:

| Item | Required value |
|---|---|
| Data, splits, seeds, budgets | data-v1; `splits/b{400,1600,6400}_s{13,21,42,87,100}.json` |
| Model | `FacebookAI/roberta-base` at the revision above, with A's `LinearRoberta` (linear head on the final CLS state) |
| Training | FP32, max length 512, batch 8 × accumulation 4, 10 epochs, AdamW, weight decay 0.01, warmup 0.1 |
| Checkpoint choice | Best IMDb validation accuracy over epochs (earliest on a tie) |
| Candidates | A fixed list of 3 per cell, as for every other method (e.g. 3 learning rates). **Fix the LoRA settings and the list in a protocol file before any test access.** |
| Selection | IMDb validation accuracy only; write the selection before testing |
| Testing | A's locked evaluation: `python -m sc4001_a.evaluate --run <winner> --protocol-locked` |
| Cost | Record `trainable_parameters`, training seconds and peak GPU memory **on TC1 V100s** so the efficiency comparison is fair |
| Drift (optional but useful) | `validation_feature_distance()` in `src/sc4001_b/train.py`: the same metric as `val_feature_distance` for B's runs |

**Four pitfalls with A's locked evaluation:**
1. It loads `best/` as a plain `LinearRoberta` from safetensors. **Merge the LoRA adapters before saving** (`peft`'s `merge_and_unload()`), and save the tokenizer alongside the model.
2. It refuses to run unless `run_config.json` has `data_manifest_sha256` (matching data-v1), `smoke: false` and `stage: "formal"`, and the run folder contains a `COMPLETE` file.
3. It refuses to run if `test/` already exists. Each winner is evaluated exactly once.
4. It needs a GPU, so run it inside a SLURM job.

**The quickest route** is to copy B's structure:
- `src/sc4001_b/train.py`: add an `M5` branch that wraps `model.roberta` with LoRA and keeps the head trainable;
- `sweep.py`: resume, selection, evaluation and pruning;
- `scripts/formal.sbatch`.

All of these already satisfy the rules above. Then produce M5 rows with the same columns as `per_seed.csv` and append them. B's `aggregate.py` is frozen and does not know about M5.

**Expected cost:** roughly M2's, about 10–15 GPU-hours for 45 runs.

**TC1 limits per account:**
- partition `UGGPU-TC1`;
- 1 GPU;
- at most 2 jobs submitted at once;
- 6 h wall time per job.

Split jobs by seed (about 3 h each) and resubmit a timed-out job to resume it. At about 15 GPU-hours on one GPU, M5 takes roughly a day of queue time end to end.

## 4. Statistics and representation analysis: what is available

- **Paired differences per seed:** `summary.json` → `paired` → `per_seed_differences`.
- **Per-example paired bootstrap:** use the prediction files of the selected runs. The same `example_id` values appear for every method and seed.
- **Drift vs accuracy (Spearman):** `val_feature_distance` against `ood_accuracy` in `per_seed.csv`. There is one value per B run; M1/M2 have none (M1's encoder is frozen, so its drift is 0 by construction).
- **CKA inputs:** the selected models (section 2), the pretrained reference, and texts from data-v1. Using test texts for analysis is fine, as long as nothing is selected or tuned on them.

## 5. Git

- B's work is on `branch-b`, in PR [#3](https://github.com/HZHANG086/SC4001-GP/pull/3) into `branch-a`.
- Until it is merged, base C's branch on `branch-b` so that B's code and results are present.
