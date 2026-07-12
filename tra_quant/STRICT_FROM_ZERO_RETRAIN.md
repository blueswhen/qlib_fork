# Strict From-Zero Retrain Guide

This document is the operational guide for reproducing the current best
baseline from zero on either an 8-card H200 machine or a dual-3090 machine.

## Current Best Baseline

Formal run:

```text
strict_zero_runs/strict_zero_h200_20260711_1245
```

Final test:

| metric | value |
| --- | ---: |
| `test_with_cost_ann_return` | `0.2804402793535583` |
| `test_with_cost_ir` | `1.6660541147890104` |
| `test_with_cost_mdd` | `-0.10195254346954019` |
| `test_IC` | `0.038227402896315404` |
| `test_Rank_IC` | `0.059762638598347115` |

Locked validation full segment:

| metric | value |
| --- | ---: |
| `valid_with_cost_ann_return` | `0.25664687372949496` |
| `valid_with_cost_ir` | `1.2771148971848802` |
| `valid_with_cost_mdd` | `-0.23370029372504092` |
| `valid_IC` | `0.06389892630367724` |
| `valid_Rank_IC` | `0.07467096363047035` |

Validation year segments:

| segment | ann | IR | MDD |
| --- | ---: | ---: | ---: |
| `Y2020` | `0.4309386668820641` | `2.343423069664322` | `-0.12842425907618166` |
| `Y2021` | `0.38044939750308976` | `1.5778897257065463` | `-0.14052874048819816` |
| `Y2022` | `0.39060580082635143` | `2.2051574154260085` | `-0.09854233431628101` |

Selected candidate:

```text
combo          = 5678,2050,2044
stage2 signal  = cash_quality_z_tw001
strategy_trial = prac_m000_hold7_r095
risk_degree    = 0.95
n_drop         = 1
account        = 150000
topk           = 5
```

The previous current best was `0.2643467300531994`; the new current best is
`0.2804402793535583`.

## Data And Environment

Run from the repository checkout:

```bash
cd /path/to/qlib_fork/tra_quant
```

Required data:

```text
../training_data/cn_data_latest
../training_data/processed/training/csi300_daily_fundamental_features_rankpct_qlib.parquet
```

The script creates or reuses this symlink:

```text
strict_zero_runs/training_data -> ../../training_data
```

Use the Python environment that already runs Qlib training. If needed, set it
explicitly:

```bash
PYTHON_BIN=/path/to/python ./train_strict_fusion_from_zero.sh --skip-execute
```

## One-Key Commands

Default auto-detected hardware:

```bash
./train_strict_fusion_from_zero.sh
```

Named H200 run:

```bash
./train_strict_fusion_from_zero.sh \
  --run-id strict_zero_h200_YYYYMMDD_1
```

Named dual-3090 run:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./train_strict_fusion_from_zero.sh \
  --run-id strict_zero_3090_YYYYMMDD_1 \
  --gpu-slots 0,1 \
  --jobs-per-gpu 1 \
  --stage2-workers 2
```

Dry-run the plan without training:

```bash
./train_strict_fusion_from_zero.sh \
  --run-id strict_zero_plan_check \
  --skip-execute
```

The script refuses to reuse an existing run directory. If a formal run is
interrupted, start a new `--run-id`; do not resume the interrupted directory
for a cross-machine reproduction claim.

## Hardware Adaptation

The formal model definition is fixed:

```text
seeds      = 5678,2050,2044
batch_size = 12000
topk       = 5
account    = 150000
```

Hardware adaptation only changes parallelism:

- H200: all visible GPUs are used; multiple rolling tasks per GPU are allowed
  when VRAM and host RAM allow it.
- Dual 3090: normally one rolling task per GPU because each card has about
  24 GiB VRAM.
- Stage2 validation trials are distributed across CPU workers selected from
  available CPU and RAM.

Resource overrides:

```bash
./train_strict_fusion_from_zero.sh \
  --gpu-slots 0,1 \
  --jobs-per-gpu 1 \
  --stage2-workers 2
```

## Strict Training Flow

The run executes these steps in order:

1. `01_stage1_validation_from_zero`
   Trains validation-side Stage1 seed caches from zero.
2. `02_stage2_validation_00..11`
   Runs the pre-registered Stage2 candidate set on validation only.
3. `03_stage2_validation_stress`
   Applies year, half, and quarter validation stress gates.
4. `04_validation_lock`
   Writes the locked candidate and an immutable pre-test manifest.
5. `05_stage1_test_from_zero_after_lock`
   Trains test-side Stage1 seed caches from zero after the lock.
6. `06_final_test_once`
   Runs the locked Stage2 fused candidate on the test segment once.

The test segment is not used for candidate selection. The selected candidate
comes from validation-only stress ranking.

## Expected Output Files

For a run id `<RUN_ID>`, inspect:

```text
strict_zero_runs/<RUN_ID>/from_zero_manifest.json
strict_zero_runs/<RUN_ID>/pretest_checkpoint.json
strict_zero_runs/<RUN_ID>/completion_audit.json
strict_zero_runs/<RUN_ID>/stage2/tmp/locked_candidate_pretest_immutable.json
strict_zero_runs/<RUN_ID>/stage2/tmp/final_test_once_summary.json
strict_zero_runs/<RUN_ID>/stage2/tmp/final_test_once_final_audit.json
```

Quick result check:

```bash
cat strict_zero_runs/<RUN_ID>/completion_audit.json
```

The current best retained run has these hashes:

```text
locked_candidate_pretest_immutable.json sha256 =
9060ff790b3f63de37539b479a3c3bd5b51404823f62622bc4695911a5a1408c

final_test_once_summary.json sha256 =
b5bf219057993ff4847f75ace63705df060e7251cc1d36efb178a598d726621c
```

## Success Criteria

For a current-best reproduction run, compare `completion_audit.json`:

```text
target_test_ann          = 0.2804402793535583
target_met_or_exceeded   = true
```

Due to floating-point and hardware differences, treat values within `1e-12` of
`0.2804402793535583` as an exact reproduction. A value above that threshold is
a new candidate for current best, but only if the same strict validation-lock
protocol was followed.

## What Not To Do

- Do not copy caches from another run into a new formal run.
- Do not reuse an interrupted run directory for a formal reproduction claim.
- Do not change `batch_size`, seeds, `account`, or `topk` and still call the
  result the same baseline.
- Do not select candidates by final test result.
- Do not edit the immutable pre-test lock after final test.

## Retained Current-Best Artifact

Only this formal artifact should remain as the current best product:

```text
strict_zero_runs/strict_zero_h200_20260711_1245
```

Old exploratory outputs under `stage1/tra_cache`, `stage1/mlruns`,
`stage2/tmp`, `stage2/fusion_cache`, `stage2/mlruns`, and `runs/` are not part
of the current best package.
