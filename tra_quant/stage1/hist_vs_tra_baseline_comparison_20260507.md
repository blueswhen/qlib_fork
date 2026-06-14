# HIST Vs TRA Baseline Comparison

Date: 2026-05-07

## Scope

This note compares the newly finished official-style HIST global run against:

1. The direct TRA-only global counterpart that was previously treated as the Alpha360 TRA baseline.
2. The current promoted baseline in this workspace, which is TRA plus fundamental rankpct XGBoost fusion.

All three use the same fixed split:

- train: 2012-01-01 to 2019-12-31
- valid: 2020-01-01 to 2022-12-31
- test: 2023-01-01 to 2026-04-01

## Compared Runs

### A. HIST official-style global

- Source: hist_global_validation_best.txt / hist_global_final_test.txt
- Model: official_hist_lstm_s240_h2
- Strategy selected on validation: prac_m000_hold1_r095
- Training alignment: qlib_official_hist_alpha360_benchmark

### B. TRA-only global baseline

- Source: tra_global_validation_best_tra_h5base_global_base_refcfg_20260504_1.txt / tra_global_final_test_tra_h5base_global_base_refcfg_20260504_1.txt
- Model: strict_tra_lstm_s240_h5
- Strategy selected on validation: prac_m050_hold3_r085
- This is the closest apples-to-apples counterpart to the HIST global script.

### C. Current promoted workspace baseline

- Source: CURRENT_BEST_BASELINE.md / current_best_baseline_manifest.json / alpha360_tra_fundamental_xgb_weight_refine_final_test.txt
- Backbone: TRA strict stage-1
- Final system: TRA + TechAlpha158A3DFundamental(rankpct) XGBoost(base)
- Fusion: zscore, TRA 0.85 + Fund-XGB 0.15

## Side-by-Side Metrics

| Run | Validation ann | Validation IR | Validation MDD | Test ann | Test IR | Test MDD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HIST global | 0.253964 | 1.368438 | -0.219921 | -0.049146 | -0.351941 | -0.310177 |
| TRA-only global | 0.148560 | 0.727521 | -0.218539 | 0.181796 | 1.148358 | -0.246247 |
| Current promoted baseline | 0.264827 | 1.350312 | -0.280321 | 0.221157 | 1.463966 | -0.131566 |

## Key Deltas

### HIST vs TRA-only global

- Validation ann: +0.105404
- Validation IR: +0.640917
- Validation MDD: -0.001383 worse
- Test ann: -0.230942
- Test IR: -1.500299
- Test MDD: -0.063930 worse

Interpretation:

- HIST looked much stronger on validation than TRA-only global.
- But after the validation-selected strategy was moved to test, HIST flipped to a negative annualized return.
- Relative to the TRA-only global baseline, the failure is entirely in out-of-sample generalization rather than in-sample fitting strength.

### HIST vs current promoted baseline

- Validation ann: -0.010863
- Validation IR: +0.018126
- Validation MDD: +0.060400 better
- Test ann: -0.270303
- Test IR: -1.815907
- Test MDD: -0.178611 worse

Interpretation:

- Even before test time, HIST did not beat the promoted workspace baseline on validation annualized return.
- On final test it is far behind the promoted baseline on all three practical dimensions: return, IR, and drawdown.

## Conclusion

The current HIST global run is not promotable.

The most important point is not that HIST trained badly. It trained and selected well enough on validation. The problem is that the validation-selected HIST strategy does not survive the test regime:

- validation ann = 25.40%
- test ann = -4.91%

By contrast, the direct TRA-only global baseline kept positive out-of-sample performance:

- validation ann = 14.86%
- test ann = 18.18%

And the current promoted workspace baseline remains clearly stronger:

- validation ann = 26.48%
- test ann = 22.12%

So for now the practical ranking is:

1. Current promoted baseline: TRA + Fund-XGB fusion
2. TRA-only global baseline
3. This HIST global run

## Referenced Files

- hist_global_validation_best.txt
- hist_global_final_test.txt
- hist_global_validation_results.csv
- tra_global_validation_best_tra_h5base_global_base_refcfg_20260504_1.txt
- tra_global_final_test_tra_h5base_global_base_refcfg_20260504_1.txt
- CURRENT_BEST_BASELINE.md
- current_best_baseline_manifest.json
- alpha360_tra_fundamental_xgb_weight_refine_final_test.txt