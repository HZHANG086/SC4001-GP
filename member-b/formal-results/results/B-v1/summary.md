# Members A + B: formal results

Status: INCOMPLETE (interim; cells still missing). Mean ± sample SD over paired seeds, accuracy in %. M1/M2 from A-v1; M3/M4/M6/L2SP from B-v1.

| Method | Labels | n | IMDb | Yelp | Amazon | OOD mean |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 400 | 5 | 81.08 ± 3.19 | 83.93 ± 3.92 | 81.54 ± 3.08 | 82.74 ± 3.48 |
| M1 | 1600 | 5 | 86.67 ± 0.50 | 88.96 ± 0.90 | 85.98 ± 1.08 | 87.47 ± 0.85 |
| M1 | 6400 | 5 | 89.54 ± 0.10 | 91.23 ± 1.42 | 88.58 ± 0.65 | 89.91 ± 1.02 |
| M2 | 400 | 5 | 90.87 ± 1.17 | 92.38 ± 1.23 | 90.68 ± 0.84 | 91.53 ± 0.87 |
| M2 | 1600 | 5 | 93.52 ± 0.19 | 93.06 ± 1.04 | 91.85 ± 1.14 | 92.45 ± 1.07 |
| M2 | 6400 | 5 | 94.20 ± 0.09 | 94.47 ± 0.57 | 93.29 ± 0.46 | 93.88 ± 0.49 |
| M3 | 400 | 5 | 92.30 ± 0.41 | 93.28 ± 0.59 | 92.19 ± 1.14 | 92.73 ± 0.75 |
| M3 | 1600 | 5 | 93.27 ± 0.43 | 92.88 ± 1.23 | 91.53 ± 1.49 | 92.21 ± 1.28 |
| M3 | 6400 | 5 | 94.41 ± 0.28 | 94.73 ± 0.69 | 93.04 ± 0.51 | 93.89 ± 0.54 |

## Paired differences (percentage points)

| Comparison | Labels | IMDb | OOD mean | n |
|---|---:|---:|---:|---:|
| M3 − M2 | 400 | 1.43 ± 1.36 | 1.20 ± 0.36 | 5 |
| M3 − M2 | 1600 | -0.26 ± 0.54 | -0.25 ± 1.91 | 5 |
| M3 − M2 | 6400 | 0.20 ± 0.33 | 0.01 ± 0.71 | 5 |

## Cost and representation drift (means over seeds)

| Method | Labels | Trainable params | Selected train min | Search train min | Peak GiB | Val feature drift |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 400 | 1,538 | 1.3 | 4.1 | 0.75 | n/a |
| M1 | 1600 | 1,538 | 3.2 | 9.4 | 0.75 | n/a |
| M1 | 6400 | 1,538 | 10.2 | 30.8 | 0.75 | n/a |
| M2 | 400 | 14,177,282 | 1.6 | 5.0 | 1.13 | n/a |
| M2 | 1600 | 14,177,282 | 3.9 | 11.7 | 1.14 | n/a |
| M2 | 6400 | 14,177,282 | 13.3 | 39.3 | 1.14 | n/a |
| M3 | 400 | 124,056,578 | 3.6 | 10.4 | 4.87 | 0.4380 |
| M3 | 1600 | 124,056,578 | 8.4 | 25.1 | 4.91 | 0.5014 |
| M3 | 6400 | 124,056,578 | 28.0 | 84.2 | 4.92 | 0.4510 |

A higher OOD mean with an unchanged or lower IMDb score is not evidence of better transfer by itself; read both absolute scores. Five seeds do not by themselves establish significance. M4 times include its stage-1 head training. Drift is the mean pooled-representation cosine distance from the pretrained encoder on IMDb validation data (correlational evidence only).
