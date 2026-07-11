# tra_quant/stage2

This directory archives the current best stage2 baseline on top of the TRA-only
stage1 rank ensemble.

## Frozen Best

- stage1: `rank_ensemble_3seed_tra72_old6`
- stage2 signal: `cash_quality_z_tw001`
- stage2 strategy: `prac_m000_hold7_r085`
- account: `150000`
- topk: `5`
- benchmark: `SH000300`
- instruments: `csi300`
- final test ann: `0.2643467300531994`
- final test ir: `1.8243593079319251`
- final test mdd: `-0.09423283807818256`

## Reproduce Final Test

From this directory:

```bash
cd /home/blueswhen/DL/qlib_fork/tra_quant/stage2
PYTHONPATH=/home/blueswhen/DL/qlib_fork:$PYTHONPATH python evaluate_stage2_cv_selected_final.py
```

Expected result:

```text
selected_trial=cash_quality_z_tw001__prac_m000_hold7_r085
test_ann=0.2643467300531994
```

## Reproduce CV Search

```bash
cd /home/blueswhen/DL/qlib_fork/tra_quant/stage2
PYTHONPATH=/home/blueswhen/DL/qlib_fork:$PYTHONPATH python run_stage2_signal_cv_search.py \
  --output-prefix tmp/stage2_cv_robust_small_stable_core_20260613_1 \
  --result-suffix stage2_cv_robust_small_stable_core_20260613_1 \
  --signal-profile robust-small \
  --strategy-profile stable-core \
  --num-shards 8 \
  --launch-shards \
  --force-recompute
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

The six required TRA stage1 seed caches are also copied into `tra_cache/` so
stage2 can rebuild the rank-ensemble validation and test signals without using
the original research workspace.

See `current_best_baseline.md` for the full reliability notes and frozen
selection rule.
