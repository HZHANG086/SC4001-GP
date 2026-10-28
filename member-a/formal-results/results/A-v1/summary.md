# Member A: formal results

Five paired seeds; sample standard deviations. All scores are absolute accuracies.

| Method | Labels | IMDb | Yelp | Amazon | OOD mean |
|---|---:|---:|---:|---:|---:|
| M1 | 400 | 81.08 ± 3.19 | 83.93 ± 3.92 | 81.54 ± 3.08 | 82.74 ± 3.48 |
| M1 | 1600 | 86.67 ± 0.50 | 88.96 ± 0.90 | 85.98 ± 1.08 | 87.47 ± 0.85 |
| M1 | 6400 | 89.54 ± 0.10 | 91.23 ± 1.42 | 88.58 ± 0.65 | 89.91 ± 1.02 |
| M2 | 400 | 90.87 ± 1.17 | 92.38 ± 1.23 | 90.68 ± 0.84 | 91.53 ± 0.87 |
| M2 | 1600 | 93.52 ± 0.19 | 93.06 ± 1.04 | 91.85 ± 1.14 | 92.45 ± 1.07 |
| M2 | 6400 | 94.20 ± 0.09 | 94.47 ± 0.57 | 93.29 ± 0.46 | 93.88 ± 0.49 |

## Paired M2 minus M1 (percentage points)

| Labels | Metric | Mean paired difference | SD of paired differences |
|---:|---|---:|---:|
| 400 | imdb_accuracy | +9.80 | 2.91 |
| 400 | yelp_accuracy | +8.45 | 4.27 |
| 400 | amazon_accuracy | +9.14 | 2.66 |
| 400 | ood_accuracy | +8.79 | 3.41 |
| 1600 | imdb_accuracy | +6.85 | 0.55 |
| 1600 | yelp_accuracy | +4.09 | 1.40 |
| 1600 | amazon_accuracy | +5.87 | 1.70 |
| 1600 | ood_accuracy | +4.98 | 1.51 |
| 6400 | imdb_accuracy | +4.67 | 0.16 |
| 6400 | yelp_accuracy | +3.24 | 0.97 |
| 6400 | amazon_accuracy | +4.70 | 0.51 |
| 6400 | ood_accuracy | +3.97 | 0.71 |

No significance claim follows automatically from five runs. See per_seed.csv for every selected run, LR and recorded cost.

Recorded training seconds cover successful final invocations; if a run resumed after interruption, add earlier attempts and SLURM allocation time. Search cost and development/pilot cost must be reported separately. Per-example predictions remain in the selected run directories.
