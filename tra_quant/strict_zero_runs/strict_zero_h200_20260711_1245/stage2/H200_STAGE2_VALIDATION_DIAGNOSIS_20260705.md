# H200 Stage2 Validation Diagnosis 2026-07-05

## Protocol

- Fixed split is unchanged:
  - train: `2012-01-01` to `2019-12-31`
  - valid: `2020-01-01` to `2022-12-31`
  - test: `2023-01-01` to `2026-04-01`
- All work in this note is validation-only.
- No new final test is used for the diagnostics below.
- Candidate selection must be based only on validation metrics.

## Seed42 Reproduction

The H200 seed42 refcheck rerun completed with:

- cache: `tra_quant/stage1/tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_seed42_refcheck_valid_only_20260705_2_valid.pkl`
- result: `tra_quant/stage1/rolling_result_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_seed42_refcheck_valid_only_20260705_2_valid.txt`
- IC: `0.039104710275226724`
- Rank IC: `0.05605642952699057`
- with_cost_ann_return: `0.04170557813198423`

Cache-level comparison against the old seed42 validation cache:

- pred shape: `(287568, 5)` vs `(287568, 5)`
- label shape: `(288000, 1)` vs `(288000, 1)`
- pred allclose with `rtol=0, atol=0, equal_nan=True`: `True`
- label allclose with `rtol=0, atol=0, equal_nan=True`: `True`
- pred max_abs_diff: `0.0`
- label max_abs_diff: `0.0`
- daily pred rankcorr mean: `1.0`

Conclusion: the H200 seed42 reproduction issue was caused by using a different training entry/config previously. The refcheck path reproduces the available old seed42 validation cache exactly.

## Current Best Stage2 Validation Mismatch

After rebuilding a validation-only stage1 summary from:

- H200 refcheck seed42 valid cache
- old6/H200-local seed2026 valid cache
- old6/H200-local seed3407 valid cache

the current-best stage2 candidate was re-evaluated on validation only:

- trial: `cash_quality_z_tw001__prac_m000_hold7_r085`
- output: `tmp/stage2_cv_h200_refcheck_old6_currentbest_candidate_validation_20260705_1.json`
- full_ann: `0.16920284380272618`
- year_ann_min: `-0.2197263510226689`
- year_ann_std: `0.3570822153239879`

This does not match the frozen current-best validation row:

- frozen full_ann: `0.2897001638735071`
- frozen year_ann_min: `0.2498259036668698`
- frozen year_ann_std: `0.009296576207662`

The old top-5 validation candidates from the frozen summary were also re-evaluated under the current available validation environment:

| trial_name | full_ann | year_ann_min | year_ann_std |
| --- | ---: | ---: | ---: |
| `cash_quality_z_tw001__prac_m000_hold7_r08` | `0.1743687658487048` | `-0.16075073895742809` | `0.30801321266709636` |
| `cash_quality_hibonus_hq08_b002__prac_m000_hold7_r08` | `0.11335337170652458` | `-0.2935082951137503` | `0.3460073426220535` |
| `large_liquid_hibonus_hq08_b002__prac_m000_hold7_r085` | `0.04990333683638444` | `-0.4437952393937236` | `0.3774297044075193` |
| `low_valuation_lowpen_lq02_p015__prac_m000_hold7_r08` | `0.10253141619118288` | `-0.45558045747569875` | `0.363048882296076` |
| `low_valuation_lowpen_lq02_p015__prac_m000_hold7_r085` | `0.1020583575083036` | `-0.4792148832558618` | `0.3871941294182769` |

Conclusion: the frozen 2026-06-13 stage2 validation rows are not reproducible in the current available local environment even after seed42 is reproduced exactly. The remaining mismatch is likely outside H200 seed42 training itself: stage1 cache provenance for old6, qlib/backtest data environment, or copied frozen-CV artifact provenance.

## Active Validation-Only Work

Running now:

- full `robust-small` + `stable-core` validation-only CV on the H200-refcheck old6 validation summary
- validation-only seed screening:
  - seeds `2028,2029,2030,2031`
  - seeds `2032,2033,2034,2035`

Current partial full-grid result had no candidate passing the frozen validation gate:

- gate: `full_ann >= 0.24`, `year_ann_min >= 0.22`, `year_ann_std <= 0.04`
- partial rows checked at note time: `80`
- strict pass count: `0`

Next legal step is to finish or narrow validation-only searches under the current environment, then lock a candidate only from validation evidence.
