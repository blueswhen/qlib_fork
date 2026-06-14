# Current Best Baseline

This document freezes the current best baseline in `examples/my_strategy` for reproducible use.

## Baseline Summary

Current best baseline:

- Alpha360 -> TRA
- TechAlpha158A3DFundamental(rankpct) -> XGBoost(base)
- score fusion: zscore
- fusion weight: TRA 0.85 + Fund-XGB 0.15

This baseline is selected by validation-only model selection, then one final test run.

## Data Split

Fixed split:

- train: 2012-01-01 to 2019-12-31
- valid: 2020-01-01 to 2022-12-31
- test: 2023-01-01 to 2026-04-01

Sources:

- [tra_strict_validation_best.txt](tra_strict_validation_best.txt)
- [alpha360_tra_fundamental_xgb_weight_refine_validation_best.txt](alpha360_tra_fundamental_xgb_weight_refine_validation_best.txt)

## Training Workflow

Stage 1 (shared backbone):

- Script: [tune_tra_alpha360_strict.py](tune_tra_alpha360_strict.py)
- Output summaries:
  - [tra_strict_validation_best.txt](tra_strict_validation_best.txt)
  - [tra_strict_final_test.txt](tra_strict_final_test.txt)

Stage 2 (current best fusion):

- Script: [tune_alpha360_tra_fundamental_xgb_weight_refine.py](tune_alpha360_tra_fundamental_xgb_weight_refine.py)
- Fundamental rankpct data:
  - `/home/blueswhen/DL/qlib/tushare/processed/training/csi300_daily_fundamental_features_rankpct_qlib.parquet`
- Frozen tabular fit params for reproducibility:
  - `num_boost_round = 1000`
  - `early_stopping_rounds = 50`
  - `verbose_eval = 20`
- Output summaries:
  - [alpha360_tra_fundamental_xgb_weight_refine_validation_best.txt](alpha360_tra_fundamental_xgb_weight_refine_validation_best.txt)
  - [alpha360_tra_fundamental_xgb_weight_refine_final_test.txt](alpha360_tra_fundamental_xgb_weight_refine_final_test.txt)

## Frozen Results

Stage 2 validation (current best):

- ann_return = 0.2648268645012359
- ir = 1.3503115407368376
- mdd = -0.280320930487992
- score = 326.73602292831794

Stage 2 final test (current best):

- ann_return = 0.22115735774992631
- ir = 1.4639662839427006
- mdd = -0.13156573083315426

Yearly test breakdown (same qlib risk-analysis metric convention):

- [alpha360_tra_fundamental_xgb_weight_refine_test_yearly_breakdown.txt](alpha360_tra_fundamental_xgb_weight_refine_test_yearly_breakdown.txt)

## One-Command Reproduction

Use controller script:

```bash
cd /home/blueswhen/DL/qlib/examples/my_strategy
python reproduce_current_best_baseline.py
```

Behavior:

- Reuses existing stage summaries by default, then refreshes machine manifest.
- Reruns missing stage summaries automatically.
- Writes consolidated manifest to:
  - [current_best_baseline_manifest.json](current_best_baseline_manifest.json)

Force full rerun:

```bash
cd /home/blueswhen/DL/qlib/examples/my_strategy
python reproduce_current_best_baseline.py --force-stage1 --force-stage2
```

## Reproducibility Notes

- Selection protocol is validation-only, then single final test.
- The archived Fund-XGB branch is frozen against qlib `XGBModel` wrapper drift by explicitly pinning the effective fit params above.
- This document is the authoritative pointer for current best baseline in this directory.
- Historical successful but superseded lines are tracked in [历史成功方案.md](历史成功方案.md).
- Historical failures, superseded paths, and non-promotable attempts are tracked only in [失败.md](失败.md).
- Any future baseline replacement must update this file and [current_best_baseline_manifest.json](current_best_baseline_manifest.json) together.
