# Pilot results: development only

Jobs on TC1 (Tesla V100-PCIE-32GB), 2026-11-07. Records are in [remote-records/](remote-records/) (weights excluded).

| Job | Purpose | Result |
|---|---|---|
| 72764 check | GPU, `pip check`, 14 unit tests on the node | COMPLETED, 1 min 10 s |
| 72765 prepare | data-v1 rebuilt from A-v1's locked Hub revisions | COMPLETED. Manifest SHA-256 `74d278fb…` is identical to A-v1, so the data are byte-for-byte A's |
| 72766 smoke | 2 optimiser steps per method | FAILED at L2-SP. The weight snapshot stayed on the CPU after the Trainer moved the model to the GPU. Fixed in commit "Fix L2-SP reference staying on CPU when training on GPU" |
| 72768 smoke | rerun after the fix | COMPLETED, all five methods (M3, M6, L2-SP, M4S1, M4) |
| 72769 pilot | source-only timing and regulariser magnitudes | COMPLETED, 22 min 36 s |

## Pilot (IMDb validation only; no test or target data read)

Budget 400 (320 train / 80 validation), seed 13, length 512, FP32, batch 8 × 4, LR 2e-5.

| Run | Val accuracy | Train seconds | Peak allocated GiB | Val feature drift | Final logged regulariser |
|---|---:|---:|---:|---:|---:|
| M3, 10 epochs | 91.25% | 211 | 4.91 | 0.5212 | none |
| M6 λ = 1, 10 epochs | 92.50% | 228 | 4.96 | 0.0168 | 0.0188 (cosine distance) |
| L2-SP λ = 0.1, 10 epochs | 91.25% | 223 | 6.17 | 0.4404 | 0.5745 (½‖w − w₀‖²) |
| M3, budget 6,400, 200 steps | 94.14% | 234 | 4.90 | 0.3692 | none |
| M6 λ = 1, budget 6,400, 200 steps | 94.14% | 286 | 4.97 | 0.0137 | 0.0138 |

"Val feature drift" is the mean cosine distance between pooled final-layer states of the selected checkpoint and the frozen pretrained encoder, on the 80 validation reviews.

## Decisions taken from the pilot (before freezing B-v1)

- **M6 λ grid: {0.1, 1, 10} → {0.01, 0.1, 1}.** At λ = 1 the drift is already about 30× lower than M3's (0.017 vs 0.52), so λ = 10 would test an almost frozen representation and add little. The new grid spans weak to strong anchoring.
- **L2-SP λ grid: {0.01, 0.1, 1} → {0.1, 1, 10}.** At λ = 0.1 the weighted penalty (≈ 0.06) is small next to the CE loss and drift falls only from 0.52 to 0.44. The grid is shifted up so its strongest value constrains clearly.
- **LR grids unchanged.** M3/M4 search {1e-5, 2e-5, 3e-5}.
- Validation accuracy was not used to choose any grid; with one seed and 80 examples it cannot.

## Cost implied for the formal grid

Full 10-epoch runs scale to about 3.5 min (400), 9 min (1,600) and 35 min (6,400) for M3. M6 costs about +22% (teacher forward pass); L2-SP about +6%.

| Phase | Runs | Estimated GPU hours |
|---|---:|---:|
| M3 | 45 | 12 |
| M6 | 45 | 14.5 |
| M4 | 15 stage-1 + 45 | 13 |
| L2-SP | 45 | 12.5 |
| Test evaluation | ≈ 120 (4.5 min each, A-v1 measurement) | 9 |
| Total | | ≈ 61 |

TC1 limits for this account (`MyTCinfo`) are 1 GPU, 2 submitted jobs and 6 h per job, so phases run sequentially in ≤ 5 h 50 min jobs that resume where the previous one stopped.

Peak memory is at most 6.2 GiB, far below the V100's 32 GiB.
