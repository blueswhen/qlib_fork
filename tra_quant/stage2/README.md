# tra_quant/stage2

This directory contains Stage2 fusion code and the current best baseline
record. The current best formal artifact is retained outside this source
directory:

```text
../strict_zero_runs/strict_zero_h200_20260711_1245
```

## Frozen Best

- stage1 seeds: `5678,2050,2044`
- stage2 signal: `cash_quality_z_tw001`
- stage2 strategy: `prac_m000_hold7_r095`
- account: `150000`
- topk: `5`
- benchmark: `SH000300`
- instruments: `csi300`
- final test ann: `0.2804402793535583`
- final test ir: `1.6660541147890104`
- final test mdd: `-0.10195254346954019`

Locked validation:

- valid ann: `0.25664687372949496`
- valid IR: `1.2771148971848802`
- valid MDD: `-0.23370029372504092`

## Reproduce From Zero

Run the one-key entry from `tra_quant`, not from this subdirectory:

```bash
cd /path/to/qlib_fork/tra_quant
./train_strict_fusion_from_zero.sh --run-id strict_zero_repro_YYYYMMDD_1
```

Dual-3090:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./train_strict_fusion_from_zero.sh \
  --run-id strict_zero_3090_YYYYMMDD_1 \
  --gpu-slots 0,1 \
  --jobs-per-gpu 1 \
  --stage2-workers 2
```

## Data Layout

Qlib market data:

```text
../../training_data/cn_data_latest
```

Fundamental parquet data:

```text
../../training_data/processed/training/csi300_daily_fundamental_features_rankpct_qlib.parquet
```

See `current_best_baseline.md` and `../STRICT_FROM_ZERO_RETRAIN.md` for the
full protocol, retained artifacts, and validation/test metrics.
