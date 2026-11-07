# Full Fine-Tuning and Representation Preservation under Limited Labels

Member B component: full fine-tuning (M3), staged LP-FT (M4), our feature-preservation method (M6), and a weight-preservation control (L2-SP). Everything is run under Member A's A-v1 protocol, so all six methods are directly comparable on the same splits, seeds and test sets.

**Status (2026-11-07): code complete and tested locally (14 tests). Protocol B-v1 is a draft until the GPU pilot confirms the grids; no formal results yet.**

## Research question

Can constraining a fully fine-tuned model to stay close to its pretrained representation improve the trade-off between in-domain accuracy (IMDb) and cross-domain generalisation (Yelp, Amazon) when labels are scarce? When does it help, and when does it hurt?

The LP-FT paper (Kumar et al., ICLR 2022) found that full fine-tuning can distort pretrained features and lose out-of-distribution accuracy, mostly in vision. Whether this happens for text sentiment under small budgets is part of what we test; it is not assumed.

## Methods

All methods start from `FacebookAI/roberta-base` (revision as in A-v1) with Member A's `LinearRoberta`, a single linear classifier on the final CLS state.

| Method | What is trained | Objective | Hyper-parameter search (3 candidates per cell) |
|---|---|---|---|
| M3 full fine-tuning | everything | CE | LR ∈ {1e-5, 2e-5, 3e-5} |
| M4 LP-FT | stage 1: head only (= A's M1, A's selected M1 LR); stage 2: everything, from the stage-1 model | CE | stage-2 LR ∈ {1e-5, 2e-5, 3e-5} |
| **M6 feature preservation (ours)** | everything | CE + λ · R_feat | λ ∈ {0.01, 0.1, 1}; LR = the cell's selected M3 LR |
| L2-SP control | everything | CE + λ · R_weight | λ ∈ {0.1, 1, 10}; LR = the cell's selected M3 LR |

- **R_feat** = mean over the batch of 1 − cos(pool(h_θ(x)), pool(h_0(x))). Here h_θ is the final-layer token states of the model being trained, taken from the same forward pass as the CE loss. h_0 is the same quantity from the frozen pretrained encoder (eval mode, no gradient), and pool is the mean over non-padding tokens.
- **x is the same labelled mini-batch as the CE loss.** M6 uses no extra unlabelled text, so it sees exactly the data M3 sees, and any difference comes from the constraint itself.
- **What is anchored:** the classifier reads the CLS state, while R_feat anchors the pooled token representation. The decision feature can therefore adapt while the general sentence representation stays close to pretraining.
- **R_weight** = ½‖w − w₀‖² over encoder weights (L2-SP, Li et al. 2018). It answers whether *preserving parameters* is equivalent to *preserving representations*.

Required controls, as agreed in the team plan:

| Control | How it is covered |
|---|---|
| λ = 0 | Equals M3: `M6 --lam 0` reproduces M3 bit for bit (`tests/test_train.py`). It is reported in the λ ablation, not used as an M6 candidate. |
| Different λ | Every λ candidate is evaluated after selection (a pre-registered ablation). |
| Weight preservation | L2-SP. |
| Staged training | M4. |

Related ideas exist (DELTA, Li et al. 2019; LwF; R3F). Our contribution is the **adaptation** of feature preservation to low-label text classification and a **controlled test** of when it helps. We do not claim to have invented representation preservation.

## Protocol (B-v1, shared with A-v1)

| Setting | Value |
|---|---|
| Shared with A-v1 | Data artifacts (checked file by file), budgets 400 / 1,600 / 6,400, seeds 13 / 21 / 42 / 87 / 100, length 512, FP32, AdamW, weight decay 0.01, warmup 0.1, batch 8 × accumulation 4, 10 epochs |
| Checkpoint selection | Highest IMDb validation accuracy; earliest epoch on a tie (HF Trainer) |
| Candidate selection | Highest IMDb validation accuracy; protocol list order on a tie. **The selection file of a cell is written before any test access.** |
| Test evaluation | Member A's locked `sc4001_a.evaluate` (same three test sets, same prediction format). It is applied to every winner and to every M6/L2-SP λ candidate (ablation). |
| Extra diagnostic | `val_feature_distance` for every run: R_feat of the selected checkpoint on IMDb validation data |
| Cost | M4 time includes its stage-1 run. M6's extra teacher forward pass is included in its train time. |

The full specification is in [protocol/B-v1.json](protocol/B-v1.json). `python -m sc4001_b.freeze` locks the code by hash into `protocol/locked_src/`; formal jobs only run the locked copy.

## Expected GPU cost

These are estimates to be replaced by the pilot. They scale A's measured V100 times (M2 at 6,400 labels: 13 minutes per run) to full back-propagation:

| Phase | Training runs | Estimated GPU hours |
|---|---:|---:|
| M3 | 45 | ≈ 13 |
| M4 | 15 stage-1 + 45 | ≈ 14 |
| M6 | 45 (teacher forward ≈ +30%) | ≈ 16 |
| L2-SP | 45 | ≈ 13 |
| Test evaluation | 30 winners + 90 λ candidates | ≈ 9 |

If time runs short, the agreed fallback order keeps M3, M6, and the λ = 0 control at every budget. L2-SP is reduced to budgets 400 and 1,600 first, then M4 reuses the M3 LR.

## How to run (from `member-b/` on the TC1 login node; all compute via SLURM)

```bash
mkdir -p logs
# 0. Environment: the same conda env as Member A (sc4001); see ../member-a/README.md and
#    ../member-a/scripts/rebuild-environment.sh (run from ../member-a).
sbatch scripts/check.sbatch                    # GPU, pip check, unit tests
sbatch scripts/prepare.sbatch                  # data-v1 with A-v1 revisions; must print DATA_MATCHES_A_V1
sbatch scripts/smoke.sbatch                    # 2 steps per method on the real model
sbatch scripts/pilot.sbatch                    # source-only timing + regulariser magnitudes
python -m sc4001_b.freeze                      # after the pilot: lock code + grids (commit the result)
sbatch scripts/formal.sbatch M3 13 21 42 87 100
sbatch scripts/formal.sbatch M6 13 21 42 87 100    # after the M3 cells it needs
sbatch scripts/formal.sbatch M4 13 21 42 87 100
sbatch scripts/formal.sbatch L2SP 13 21 42 87 100
python -m sc4001_b.sweep --finish              # results/B-v1/{summary.md,per_seed.csv,ablation.csv}
```

A job that hits the time limit can simply be resubmitted with the same command. Completed candidates are skipped, and interrupted ones resume from their last epoch checkpoint. Phases can be split by seed or budget (`--budgets 400 1600`) to fit the 5:50 time limit.

## Outputs

`runs/B-v1/<candidate>/` follows Member A's layout: `run_config.json`, `parameters.json`, `validation.json`, `history.json`, `source/`, `best/` (winners only), and `test/{metrics.json,*_predictions.jsonl}`. `selection-<method>-b<budget>-s<seed>.json` records each frozen choice. `results/B-v1/` holds the combined A+B tables.

## Tests

```bash
cd tests && PYTHONPATH=../src:../../member-a/protocol/locked_src python -m unittest discover -p 'test_*.py'
```

The tests run on CPU and offline, using a tiny random RoBERTa and fake data. They cover:
- the objectives;
- end-to-end training of every method;
- M6(λ=0) ≡ M3;
- M4's head-only first stage;
- compatibility of the run folder with A's evaluation;
- argument rules;
- sweep selection, dependency, evaluation-order, idempotency and cleanup logic;
- the A-v1 data check;
- aggregation.
