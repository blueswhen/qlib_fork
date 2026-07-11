# tra_quant

This directory contains the strict from-zero TRA Stage1 + fused Stage2
training package.

## Current Best

The current best baseline is the strict from-zero run:

```text
strict_zero_runs/strict_zero_h200_20260711_1245
```

Final test result:

```text
test_with_cost_ann_return = 0.2804402793535583
test_with_cost_ir         = 1.6660541147890104
test_with_cost_mdd        = -0.10195254346954019
test_IC                   = 0.038227402896315404
test_Rank_IC              = 0.059762638598347115
```

Locked validation result:

```text
valid_with_cost_ann_return = 0.25664687372949496
valid_with_cost_ir         = 1.2771148971848802
valid_with_cost_mdd        = -0.23370029372504092
valid_IC                   = 0.06389892630367724
valid_Rank_IC              = 0.07467096363047035
```

Hard constraints:

```text
account = 150000
topk    = 5
```

## One-Key Training

Use the strict entry point for both H200 and dual-3090 machines:

```bash
cd /path/to/qlib_fork/tra_quant
./train_strict_fusion_from_zero.sh
```

The script creates a new directory under `strict_zero_runs/`, detects available
NVIDIA GPUs, and adjusts parallelism without changing the formal model
definition. Batch size is fixed at `12000` on every machine.

For a named formal run:

```bash
./train_strict_fusion_from_zero.sh --run-id strict_zero_3090_YYYYMMDD_1
```

For a dual-3090 host:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./train_strict_fusion_from_zero.sh \
  --run-id strict_zero_3090_YYYYMMDD_1 \
  --gpu-slots 0,1 \
  --jobs-per-gpu 1 \
  --stage2-workers 2
```

For an 8-card H200 host, auto mode is normally enough:

```bash
./train_strict_fusion_from_zero.sh --run-id strict_zero_h200_YYYYMMDD_1
```

Detailed protocol, expected outputs, validation metrics, final-test metrics,
and reproduction rules are documented in
[`STRICT_FROM_ZERO_RETRAIN.md`](STRICT_FROM_ZERO_RETRAIN.md).

## Strict Protocol

The script enforces this order:

1. Create a fresh isolated run directory containing source code only.
2. Train Stage1 validation-side seed caches from zero.
3. Search and stress-check fused Stage2 candidates only on validation.
4. Freeze the selected candidate into a read-only pre-test manifest.
5. Train Stage1 test-side seed caches from zero after the lock.
6. Run the locked fused final test once.

No old model, prediction cache, CV grid, locked manifest, or result file is
accepted as an input to a formal run.

## Directory Layout

- `train_strict_fusion_from_zero.sh`: one-key entry point.
- `train_strict_fusion_from_zero.py`: strict orchestration script.
- `STRICT_FROM_ZERO_RETRAIN.md`: detailed H200/3090 training guide.
- `stage1/`: Stage1 TRA source files and docs.
- `stage2/`: Stage2 fusion source files and current best docs.
- `strict_zero_runs/strict_zero_h200_20260711_1245/`: retained current-best
  formal training artifact.

Historical failed or intermediate training artifacts should not be kept in
`stage1/`, `stage2/`, or `runs/`. A formal run should live under
`strict_zero_runs/<run-id>/`.
