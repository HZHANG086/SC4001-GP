# RoBERTa Adaptation under Limited Labels

A reproducible comparison of frozen representations and partial fine-tuning for sentiment classification across domains. Models are trained on a limited number of IMDb labels and evaluated on IMDb, Yelp Polarity, and Amazon Polarity.

The completed **A-v1** experiment comprises **90 candidate training runs**, **30 selected method–budget–seed configurations**, and **1,050,000 test predictions**. This directory contains the data protocol, implementation, frozen source snapshots, experiment records, and verified results for the project's Member A component.

## Research objective

How much does updating the final Transformer layers improve in-domain accuracy and cross-domain generalisation when labelled data are limited?

The comparison holds the pretrained backbone, classifier architecture, data splits, input length, and evaluation sets fixed. Each method uses its own predefined learning-rate grid, with model selection based exclusively on source-domain validation accuracy.

## Methods

Both methods start from the original pretrained `FacebookAI/roberta-base` checkpoint and use a linear classifier over the final CLS representation.

| Method | Adaptation strategy | Trainable parameters |
|---|---|---:|
| M1 | Freeze the encoder and train the linear classifier | 1,538 |
| M2 | Update the final two Transformer layers and the same classifier | 14,177,282 |

Dropout is disabled in frozen components. Full fine-tuning, staged adaptation, LoRA, and feature-preservation methods are outside this component's experimental scope.

## Data and experimental protocol

### Datasets

| Role | Dataset | Evaluation size |
|---|---|---:|
| Source training and validation | IMDb official training split | Budget-dependent |
| In-domain evaluation | IMDb official test split | 25,000 |
| Cross-domain evaluation | Yelp Polarity official test split | 5,000 |
| Cross-domain evaluation | Amazon Polarity official test split | 5,000 |

Source preparation removes normalised duplicates, conflicting-label text, and overlap with the IMDb test text. The original IMDb test set is retained, including its documented duplicates. Cleaning produces 24,779 eligible source examples. Normalisation is used for duplicate detection; model inputs retain the original text.

Each target test subset contains 2,500 examples per class. Target sampling removes normalised duplicates and excludes source-text overlap and selected overlap between the two target subsets. Amazon inputs concatenate the title and content with a newline. Target labels and predictions are excluded from training and model selection.

Dataset revisions, fixed indices, split hashes, and audit statistics are recorded in [remote-records/data-v1/](remote-records/data-v1/). See [DATA_AUDIT.md](DATA_AUDIT.md) for the detailed audit.

### Label budgets and repetition

Each label budget includes both training and validation labels.

| Total labels | Training labels | Validation labels |
|---:|---:|---:|
| 400 | 320 | 80 |
| 1,600 | 1,280 | 320 |
| 6,400 | 5,120 | 1,280 |

Experiments use five paired seeds: **13, 21, 42, 87, and 100**. Methods share the same split for each budget and seed. Splits are class-balanced; training subsets and validation subsets are separately nested across budgets, with no train–validation overlap across budgets. Subsets from different seeds may overlap.

### Training and model selection

| Setting | A-v1 configuration |
|---|---|
| Maximum input length | 512 tokens |
| Precision | FP32 |
| Optimiser | AdamW |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Batch size | 8 |
| Gradient accumulation | 4 steps |
| Training limit | 10 epochs per candidate |
| M1 learning rates | `1e-4`, `1e-3`, `1e-2` |
| M2 learning rates | `5e-6`, `2e-5`, `5e-5` |
| Checkpoint selection | Highest IMDb validation accuracy; earliest epoch on a tie |
| Candidate selection | Highest IMDb validation accuracy; predefined learning-rate order on a tie |

Two methods × three budgets × five seeds × three learning rates produce 90 candidate runs and 30 selected configurations. All source-validation selections precede formal test evaluation. Input length was fixed after source-validation development experiments; the pilot does not establish a confirmatory claim about sequence length.

The complete specification, model and dataset revisions, and source hashes are in [protocol/A-v1.json](protocol/A-v1.json). Formal jobs execute [protocol/locked_src/](protocol/locked_src/) to preserve the implementation used for the reported results.

## Results

Accuracy is reported as **mean ± sample standard deviation across five seeds**, in percent. The OOD mean is the arithmetic mean of Yelp and Amazon accuracy within each seed; IMDb is excluded.

| Method | Label budget | IMDb | Yelp | Amazon | OOD mean |
|---|---:|---:|---:|---:|---:|
| M1 | 400 | 81.08 ± 3.19 | 83.93 ± 3.92 | 81.54 ± 3.08 | 82.74 ± 3.48 |
| M1 | 1,600 | 86.67 ± 0.50 | 88.96 ± 0.90 | 85.98 ± 1.08 | 87.47 ± 0.85 |
| M1 | 6,400 | 89.54 ± 0.10 | 91.23 ± 1.42 | 88.58 ± 0.65 | 89.91 ± 1.02 |
| M2 | 400 | 90.87 ± 1.17 | 92.38 ± 1.23 | 90.68 ± 0.84 | 91.53 ± 0.87 |
| M2 | 1,600 | 93.52 ± 0.19 | 93.06 ± 1.04 | 91.85 ± 1.14 | 92.45 ± 1.07 |
| M2 | 6,400 | 94.20 ± 0.09 | 94.47 ± 0.57 | 93.29 ± 0.46 | 93.88 ± 0.49 |

M2 achieves higher mean IMDb and OOD accuracy at every budget. Its paired OOD gains over M1 are **8.79**, **4.98**, and **3.97 percentage points** at 400, 1,600, and 6,400 labels, respectively. The gap narrows as the label budget increases.

These observations support partial adaptation in this setting. They do not establish a causal role for representation drift, predict the performance of untested methods, or imply statistical significance from five repetitions alone.

Result entry points:

- [summary.md](formal-results/results/A-v1/summary.md): English accuracy tables and paired differences.
- [per_seed.csv](formal-results/results/A-v1/per_seed.csv): selected learning rates, per-seed scores, parameter counts, and recorded costs.
- [summary.json](formal-results/results/A-v1/summary.json): machine-readable five-seed statistics.
- [VERIFICATION.json](formal-results/results/A-v1/VERIFICATION.json): completed integrity and metric checks.
- [RESULTS_ZH.md](RESULTS_ZH.md): detailed Chinese-language results and resource analysis.

Verification covered prediction IDs, labels, probabilities, recomputed accuracy, source-only model selection, split and source hashes, and paired statistics. All 90 candidate training runs succeeded. A scheduler timeout interrupted final evaluation; a subsequent job completed the remaining two evaluations without retraining candidates. The three formal allocations total approximately 11.43 single-GPU hours, excluding development and pilot jobs.

## Repository structure

```text
member-a/
├── src/sc4001_a/          Data preparation, modelling, training, selection and evaluation
├── protocol/             Frozen A-v1 specification and source snapshot
├── scripts/              Environment setup and SLURM entry points
├── tests/                Data protocol, model-freezing and checkpoint-resume checks
├── remote-records/       Data indices, audits, environment and development records
├── formal-results/
│   ├── results/A-v1/     Selected configurations and aggregate results
│   ├── runs/A-v1/        Candidate records and selected-run predictions
│   └── logs/             Formal job logs
└── DELIVERY_MANIFEST.json
```

Each candidate record includes `run_config.json`, `parameters.json`, `history.json`, `validation.json`, and its source snapshot. Selected runs additionally contain `test/metrics.json` and three `*_predictions.jsonl` files. Prediction rows contain `example_id`, `label`, `prediction`, and `p_positive`, enabling paired analysis without duplicating review text.

Model weights, downloaded datasets, and Conda environments are excluded from Git. Recomputing summary statistics uses the committed records; rerunning model inference requires retained checkpoints or retraining.

## Reproducibility

Run the following commands from `member-a/` unless specified otherwise.

### 1. Environment

The recorded GPU environment uses Python 3.11, PyTorch 2.5.1 with CUDA 11.8, and MKL 2024.0.0. Package versions are pinned in [pyproject.toml](pyproject.toml); captured environment records are in [remote-records/](remote-records/).

For a standalone installation, create an isolated environment, install a compatible PyTorch build, then install the package:

```bash
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -p test_protocol.py -v
```

The included SLURM scripts target the original NTU environment. Review their partition, QoS, resource limits, and environment activation before using them on another system. [scripts/rebuild-environment.sh](scripts/rebuild-environment.sh) documents the NTU-specific reconstruction path; it depends on the original shared environment.

### 2. Inspect and regenerate summaries

The committed records support summary regeneration without a GPU, model weights, or dataset downloads. From `member-a/`:

```bash
cd formal-results
PYTHONPATH=../protocol/locked_src python -m sc4001_a.aggregate \
  --winners results/A-v1/winners.json \
  --out /tmp/sc4001-a-summary
```

The working directory matters because the selected run paths are relative to the experiment root. This command writes a separate summary directory and preserves the committed results.

### 3. Rebuild data and run experiments

For a fresh NTU reproduction, activate the configured environment and create the log directory. Verify the environment and rebuild data using the recorded Hub revisions:

```bash
mkdir -p logs
sbatch scripts/check.sbatch
sbatch scripts/prepare.sbatch \
  --source-lock remote-records/data-v1/sources.json
```

Wait for both jobs to succeed and confirm `artifacts/data-v1/READY` exists before training. The frozen sweep validates the data manifest and source hashes against A-v1. Data preparation requires access to the upstream datasets and model.

The original sweep can be submitted in two dependent stages:

```bash
first_job=$(sbatch --parsable scripts/formal.sbatch 13 21 42)
sbatch --dependency=afterok:"$first_job" scripts/formal.sbatch 87 100 --finish
```

The final stage selects configurations, evaluates all three test sets, and writes `results/A-v1/`. These commands initiate a new training sweep; they are unnecessary for inspecting the completed records. Account for scheduler time limits when planning a reproduction.

The sweep skips completed candidates and resumes incomplete runs when an epoch checkpoint is available. Preserve failed-run logs and recovery records. Each candidate's learning rate, data split, model revision, and other frozen settings must remain unchanged when resuming.

### 4. Output interpretation

A complete sweep requires `results/A-v1/COMPLETE`, 30 unique method–budget–seed cells, and all three test-domain metrics for every selected run. Scheduler exit status alone does not establish experiment completeness.

Training time records include training, validation, and checkpoint writes within the training invocation. They exclude process startup, model loading, and data mapping. Search cost includes all three candidates per cell; interrupted invocations and development jobs require separate accounting. Peak allocated CUDA memory measures process allocations rather than total device occupancy.

## Limitations and provenance

The study uses one source domain and two fixed target subsets. Dataset differences include topic, length, and writing style, so performance changes cannot be attributed to a single domain-shift factor. Normalised hashing does not detect all semantic duplicates. IMDb's official test duplicates are retained and disclosed in the audit. M1 recomputes encoder features rather than caching them, which affects its measured runtime.

The model and data originate from [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base), [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb), [Yelp Polarity](https://huggingface.co/datasets/fancyzhx/yelp_polarity), and [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity). Related work and bibliographic entries are provided in [RELATED_WORK.md](RELATED_WORK.md) and [references.bib](references.bib). Upstream resources remain subject to their respective terms.

AI assistance was used for environment troubleshooting, implementation, experiment orchestration, verification, and documentation. Reported metrics are derived from the recorded experiment outputs.
