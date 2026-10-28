# Pilot results: development only

Job 70078 completed successfully in 5m48s on Tesla V100-PCIE-32GB.

320 train + 80 IMDb validation examples; seed 13; 10 epochs; FP32; batch 8; accumulation 4. No test predictions generated.

| Method | Length | Validation accuracy | Training seconds | Peak allocated GiB | Peak reserved GiB | Trainable parameters |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 256 | 83.75% | 60.95 | 0.612 | 0.744 | 1,538 |
| M1 | 512 | 86.25% | 72.79 | 0.753 | 1.027 | 1,538 |
| M2 | 256 | 85.00% | 69.12 | 0.888 | 0.988 | 14,177,282 |
| M2 | 512 | 85.00% | 87.94 | 1.135 | 1.451 | 14,177,282 |

A-v1 uses length 512 after this source-only development pilot: M1 classified two more of 80 validation examples correctly, M2 was unchanged, source-pool truncation fell from 41.85% to 13.68%, and baseline memory/runtime remained affordable. This was a development choice made after the pilot, not a preregistered pilot decision rule. It does not establish that 512 is generally better.

One seed and 80 validation examples cannot establish stability or generalization superiority. Formal runs use the same length for both methods, three budgets, five paired seeds, and three LR candidates per cell. Pilot/smoke runs are excluded from formal selection and test evaluation.

Training wall time includes Trainer training, epoch validation and checkpoint writes, but excludes prior Python imports, model loading, data mapping and environment setup. Allocation time should be reported separately.

CPU job 70081 and CUDA job 70082 passed checkpoint/optimizer/RNG continuation checks: stop after step 2, restore and finish step 4, compare all weights with uninterrupted training at absolute tolerance 1e-7. These tests do not establish resilience to corrupted files or hardware failure.
