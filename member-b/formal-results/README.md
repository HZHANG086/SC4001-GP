# Member B formal results (B-v1): guide for analysis and reporting

All 60 B-v1 cells are complete: 4 methods (M3, M4, M6, L2-SP) × 3 label budgets (400, 1,600, 6,400) × 5 paired seeds (13, 21, 42, 87, 100).

The setup is identical to Member A's A-v1 protocol: the same data (checked file by file), head, training settings and locked evaluation code. M1–M6 and L2-SP can therefore be compared directly.

Model weights are not stored. Everything else is: configs, validation histories, test metrics and per-example predictions.

## Files to use

| File | Contents |
|---|---|
| [results/B-v1/per_seed.csv](results/B-v1/per_seed.csv) | **Main table.** One row per selected method × budget × seed. It includes Member A's M1 and M2 rows (column `source`), giving 90 rows in total. |
| [results/B-v1/ablation.csv](results/B-v1/ablation.csv) | Pre-registered λ ablation: the test results of every M6 and L2-SP λ candidate, plus the selected M3 run as λ = 0. `selected` marks the candidate chosen on validation. |
| [results/B-v1/summary.md](results/B-v1/summary.md) | Ready-made tables: mean ± sample SD, paired differences, cost, drift and the λ ablation. |
| [results/B-v1/summary.json](results/B-v1/summary.json) | The same numbers in machine-readable form, including the per-seed paired differences. |
| `runs/B-v1/selection-<method>-b<budget>-s<seed>.json` | The frozen choice for each cell: winner, candidates and validation scores. It was written before any test access. |
| `runs/B-v1/<run>/test/{imdb,yelp,amazon}_predictions.jsonl` | One line per test example: `example_id`, `label`, `prediction`, `p_positive`. Available for every selected run and every M6/L2-SP λ candidate. Same format as Member A's. |
| `runs/B-v1/<run>/{run_config,validation,history,parameters}.json` | The configuration, the validation result of the chosen checkpoint, the per-epoch history and the trainable-parameter list for every candidate. |

`M4S1-*` folders hold the head-only first stage of M4. They have no test results by design.

### `per_seed.csv` columns

| Column | Meaning |
|---|---|
| `method`, `budget`, `seed` | Cell identity |
| `lr`, `lam` | The selected learning rate and λ (empty if the method has none) |
| `source_val_accuracy` | IMDb validation accuracy of the selected run (the selection criterion) |
| `imdb_accuracy`, `yelp_accuracy`, `amazon_accuracy` | Test accuracy as a fraction, 0–1 |
| `ood_accuracy` | Mean of the Yelp and Amazon accuracies |
| `trainable_parameters` | Number of parameters updated |
| `selected_train_seconds` | Training time of the selected run. For M4 it includes stage 1. |
| `search_train_seconds_recorded` | Training time of all three candidates in the cell |
| `selected_peak_allocated_gib` | Peak GPU memory of the selected run |
| `val_feature_distance` | **Representation drift**: the mean cosine distance between mean-pooled final-layer states of the trained encoder and of the pretrained encoder, on IMDb validation data. 0 means unchanged; higher means further from pretraining. Not measured for M1/M2. |
| `run`, `source` | The run folder and the protocol it comes from (A-v1 or B-v1) |

## Rules for analysis

1. **Do not select anything on test data.** Each cell's winner is fixed in its selection file. The λ ablation may be reported as an ablation, but it must not be used to re-pick λ.
2. **Compare methods by paired seed.** Subtract within the same seed, then average over the 5 seeds. `summary.json` → `paired` already has these values.
3. **With 5 seeds, differences below about 1 percentage point are not significant.** Report means ± SD and the per-seed differences, and avoid wording like "method X is better" unless all seeds agree.
4. Drift is correlational evidence only.

## Results

Test accuracy (%) is the mean over 5 seeds. OOD is the mean of Yelp and Amazon. Drift is `val_feature_distance`.

| Method | 400: IMDb / OOD | 1,600: IMDb / OOD | 6,400: IMDb / OOD | Drift |
|---|---|---|---|---|
| M1 head only (A) | 81.1 / 82.7 | 86.7 / 87.5 | 89.5 / 89.9 | – |
| M2 partial fine-tuning (A) | 90.9 / 91.5 | 93.5 / 92.5 | 94.2 / 93.9 | – |
| M3 full fine-tuning | 92.3 / 92.7 | 93.3 / 92.2 | 94.4 / 93.9 | 0.44–0.50 |
| M4 LP-FT | 92.4 / 91.8 | 93.9 / 93.1 | 94.6 / 94.4 | 0.02–0.09 |
| M6 feature preservation (ours) | 92.3 / 92.0 | 93.5 / 92.9 | 94.5 / 93.9 | 0.02–0.19 |
| L2-SP weight preservation | 92.0 / 92.4 | 93.4 / 92.7 | 94.5 / 94.2 | 0.35–0.42 |

## Main findings

1. **Encoder adaptation is essential.** Head-only training (M1) trails every fine-tuning method by 10–11 points at 400 labels and about 5 points at 6,400.
2. **Partial fine-tuning is the most efficient option.** M2 trains about 11% of the parameters, with about half the training time and a quarter of the GPU memory of M3. It matches M3 at 1,600 and 6,400 labels. Full fine-tuning helps clearly only at 400 labels: +1.4 IMDb and +1.2 OOD (paired).
3. **The cross-domain penalty is small.** Yelp and Amazon accuracy tracks IMDb accuracy closely, so there is little out-of-domain loss for any method to recover.
4. **No preservation method (M4, M6, L2-SP) significantly beats M3.**
   - All paired differences are within about 1 point and smaller than the seed-to-seed spread.
   - M4 has the highest means at 1,600 and 6,400 labels but is the least stable at 400 (OOD SD ±2.5).
   - The feature-distortion effect reported for vision by LP-FT (Kumar et al., 2022) does not appear in this text setting.
5. **Less drift does not mean better transfer.**
   - M6 cuts drift by 60–95% with no loss of IMDb accuracy, but with no significant OOD gain.
   - At 400 labels, M3 drifts most yet has the best OOD mean.
6. **Preserving weights does not preserve representations.** L2-SP's drift stays close to M3's. Its strongest setting (λ = 10) collapses training to about chance level at 400 and 1,600 labels; validation never selected it.
7. **Hyperparameter selection is a bottleneck with few labels.** In the ablation, λ = 1 gives M6's best OOD mean at 1,600 and 6,400 labels, but the small validation sets rarely select it.

Limits:
- 5 seeds, one model (RoBERTa-base), one task (binary English sentiment).
- The two out-of-domain sets are fairly close to IMDb.
- The vision-style distortion effect may still appear under a larger domain shift.

## Suggested figures and analyses

- **Accuracy vs label budget:** IMDb and OOD panels, all six methods, with error bars from `per_seed.csv`.
- **Paired-difference plots:** M6−M3, M4−M3, L2SP−M3 and M6−M4, showing the per-seed points against zero (`summary.json` → `paired`).
- **Drift vs OOD accuracy:** a scatter plot, one point per run, using `val_feature_distance` against `ood_accuracy`.
- **λ curves:** M6 and L2-SP, using `ablation.csv`.
- **Cost table:** parameters, training time and peak memory.
- **Optional error analysis:** compare predictions across methods on the same `example_id` values, for example OOD examples that M3 gets wrong and M4/M6 get right.

## Reproducing the tables

From `member-b/formal-results/`:

```bash
PYTHONPATH=../src python3 -m sc4001_b.aggregate --protocol ../protocol/B-v1.json \
  --a-per-seed ../../member-a/formal-results/results/A-v1/per_seed.csv \
  --runs runs/B-v1 --out results/B-v1
```

This uses only the Python standard library.

Training cost, as measured on TC1 V100s:
- **Training time:** 44.5 GPU-hours (M3 10.0, M4 10.1 plus 1.2 for stage 1, M6 12.4, L2-SP 10.9).
- **Total job time:** about 56 GPU-hours, including test evaluation.
- **Logs:** in `logs/`.
