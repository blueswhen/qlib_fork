# ALSTM H5-Only Retrain 2026-05-04

This document freezes the successful ALSTM h5-only retrain that was run after removing h3 and wide from the active search direction.

This run is intentionally not promoted to current best yet.

Promotion gate:

- keep [CURRENT_BEST_BASELINE.md](CURRENT_BEST_BASELINE.md) unchanged for now
- only promote this ALSTM line after the corresponding stage2 retrain also exceeds the current best baseline on final test

## Goal

Recover and then exceed the historical ALSTM strict baseline under a tighter, cheaper search process.

Constraints used for this run:

- seed = 42 only
- h5 only
- validation-only selection, then one final test run
- broaden strategy search while avoiding multi-seed and multi-horizon expansion
- exact-parameter-only cache reuse
- row-level resume for validation strategy evaluation

## Final Frozen Search Scope

Training script:

- [tune_alstm_alpha360_global.py](tune_alstm_alpha360_global.py)

Shared rolling engine:

- [rolling_alstm_alpha360_latest.py](rolling_alstm_alpha360_latest.py)

Active model search scope after narrowing:

- model_trial = strict_gru_base_s240_h5
- model_key = gru_base
- step = 240
- label = Ref($close, -6) / Ref($close, -1) - 1

Active strategy search scope:

- 21 strategy trials total
- exact strict baseline strategy retained in search space
- widened around margin / hold / risk_degree / n_drop

Explicit removals before freezing this run:

- h3 removed from active search
- wide removed from active search
- h7 not searched
- no multi-seed training

## Reproducible Run

Run suffix:

- seed42_h5only_restart_20260504_094502

Primary artifacts:

- [alstm_global_validation_best_seed42_h5only_restart_20260504_094502.txt](alstm_global_validation_best_seed42_h5only_restart_20260504_094502.txt)
- [alstm_global_final_test_seed42_h5only_restart_20260504_094502.txt](alstm_global_final_test_seed42_h5only_restart_20260504_094502.txt)
- [alstm_global_validation_results_seed42_h5only_restart_20260504_094502.csv](alstm_global_validation_results_seed42_h5only_restart_20260504_094502.csv)
- [alstm_global_seed42_h5only_restart_20260504_094502.log](alstm_global_seed42_h5only_restart_20260504_094502.log)

Execution environment:

- provider_uri = ~/.qlib/qlib_data/cn_data_latest
- region = REG_CN
- benchmark = SH000300
- instruments = csi300
- gpu_slots = 0,1
- deterministic_runtime = True

Fixed split:

- train: 2012-01-01 to 2019-12-31
- valid: 2020-01-01 to 2022-12-31
- test: 2023-01-01 to 2026-04-01

## Validation Winner

Winner:

- model_trial = strict_gru_base_s240_h5
- strategy_trial = prac_m050_h5_r085
- strategy_class = PracticalTopkDropoutStrategy
- n_drop = 1
- risk_degree = 0.85
- score_margin = 0.005
- hold_thresh = 5
- slot_budget_ratio = 1.0

Validation metrics:

- IC = 0.03360759015064858
- Rank IC = 0.03521773604835402
- ann_return = 0.25745863874353503
- ir = 1.2874037112230095
- mdd = -0.15434181292938026
- score = 318.7419880460979

Source:

- [alstm_global_validation_best_seed42_h5only_restart_20260504_094502.txt](alstm_global_validation_best_seed42_h5only_restart_20260504_094502.txt)

## Final Test Result

Final test metrics:

- IC = 0.035839189635377175
- Rank IC = 0.058523636073760314
- ann_return = 0.12985578707077253
- ir = 0.8381543569468066
- mdd = -0.1421629560946252

Source:

- [alstm_global_final_test_seed42_h5only_restart_20260504_094502.txt](alstm_global_final_test_seed42_h5only_restart_20260504_094502.txt)

## What Actually Improved

The improvement did not come from switching to a wider ALSTM.

Observed result from validation ranking:

- best base score = 318.7419880460979
- best wide score before removal = 184.847127
- top 10 validation rows were dominated by base, with only one wide row present before the search was narrowed

Practical conclusion:

- the useful change was keeping the strong strict_gru_base_s240_h5 signal and retuning the strategy layer
- the winning move was from old practical h1 / r07 to h5 / margin 0.005 / risk 0.85
- wide should remain excluded from the active search path unless new evidence appears

## Comparison Against Older ALSTM Baseline

Historical ALSTM strict baseline:

- [alstm_strict_validation_best.txt](alstm_strict_validation_best.txt)
- [alstm_strict_final_test.txt](alstm_strict_final_test.txt)

Delta versus historical ALSTM final test:

- old ann_return = 0.12214943476100354
- new ann_return = 0.12985578707077253
- delta_ann = 0.00770635230976899
- old ir = 0.8524232688708969
- new ir = 0.8381543569468066
- delta_ir = -0.014268911924090292
- old mdd = -0.2629032381077046
- new mdd = -0.1421629560946252
- delta_mdd = 0.1207402820130794

Interpretation:

- this retrain exceeded the old ALSTM strict line on test ann_return
- this retrain materially improved test max drawdown
- test IR was slightly lower than the old ALSTM strict line

## Comparison Against TRA-Only

TRA-only strict baseline:

- [tra_strict_final_test.txt](tra_strict_final_test.txt)

Delta versus TRA-only final test:

- TRA-only ann_return = 0.12640759423657094
- ALSTM h5-only ann_return = 0.12985578707077253
- delta_ann = 0.003448192834201591
- TRA-only ir = 0.9340812767191283
- ALSTM h5-only ir = 0.8381543569468066
- delta_ir = -0.0959269197723217
- TRA-only mdd = -0.16187643555433998
- ALSTM h5-only mdd = -0.1421629560946252
- delta_mdd = 0.01971347945971478

Interpretation:

- ALSTM h5-only exceeded TRA-only on test ann_return
- ALSTM h5-only also improved test max drawdown
- TRA-only still had the stronger test IR

## Promotion Status

Status:

- frozen as a successful historical ALSTM retrain
- not promoted to current best baseline

Do not update these yet:

- [CURRENT_BEST_BASELINE.md](CURRENT_BEST_BASELINE.md)
- [current_best_baseline_manifest.json](current_best_baseline_manifest.json)

Required promotion condition:

- rerun the corresponding stage2 path on top of this ALSTM h5-only winner
- only update current best if that stage2 retrain also exceeds the existing current best baseline on final test

## Next Intended Path

Continue from this frozen stage1 result only.

Recommended next step:

- retrain the stage2 branch against this ALSTM h5-only winner instead of against the old ALSTM strict baseline

Do not reopen these search directions unless new evidence appears:

- h3
- wide
- h7
- multi-seed ALSTM for this recovery path