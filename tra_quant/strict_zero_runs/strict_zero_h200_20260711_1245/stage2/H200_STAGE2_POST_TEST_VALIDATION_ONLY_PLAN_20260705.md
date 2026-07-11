# H200 Stage2 Post-Test Validation-Only Plan - 2026-07-05

## Protocol Status

- Fixed split remains unchanged:
  - train: `2012-01-01` to `2019-12-31`
  - valid: `2020-01-01` to `2022-12-31`
  - test: `2023-01-01` to `2026-04-01`
- 2026-07-06 strict rerun policy:
  - The user reset the process after the earlier post-test audit.
  - Historical fixed-test/final outputs are invalid for selection and are audit-only.
  - The current strict rerun has not run `test` or `final`.
  - All tuning, screening, thresholding, and candidate ranking must use validation artifacts only.
  - The fixed test split may be run exactly once only after one candidate is locked by a validation-only rule.
  - No replacement or new test set is used.
- Historically, before the 2026-07-06 strict rerun reset, the fixed test had already been consumed by a strict final-once audit.
- That historical consumed test result is audit-only. It must not be used for any current model, seed, signal, strategy, or threshold selection.
- All further work before any future holdout must be validation-only.

## Final-Once Audit Result

Locked candidate before final:

- combo: `99,11,1234`
- trial: `baseline_seq__prac_m000_hold5_r08`
- validation full ann: `0.3115388874764329`
- validation worst-year ann: `0.2776215064067659`
- validation year ann std: `0.076789807223851`
- locked manifest: `tmp/h200_selector_99_11_1234_target026_locked_20260705_1_locked_manifest.json`

The one allowed final test did not exceed the current best baseline:

- final summary: `tmp/stage2_h200_99_11_1234_target026_20260705_1_final_once_summary.json`
- final audit: `tmp/stage2_h200_99_11_1234_target026_20260705_1_final_once_final_audit.json`
- current best baseline target: `0.2643467300531994`

This failure cannot be followed by another fixed-test run for tuning.

## What Went Wrong

The old annual validation gate was too coarse. The locked candidate had strong full-period and yearly validation metrics, but its within-year path was not stable enough:

- strict stress audit: `tmp/h200_validation_stress_99_11_1234_locked_strictgate_20260705_1_summary.json`
- segment detail: `tmp/h200_validation_stress_99_11_1234_locked_strictgate_20260705_1_segment_detail.csv`
- candidate stress summary: `tmp/h200_validation_stress_99_11_1234_locked_strictgate_20260705_1_candidate_summary.csv`

Stress metrics for `99,11,1234 + baseline_seq__prac_m000_hold5_r08`:

- full ann: `0.31153888747643294`
- year ann min: `0.2776215064067661`
- half ann min: `0.097001`
- quarter ann min: `-0.070167`
- quarter positive ratio: `0.75`
- quarter rank IC min: `0.020708`
- quarter rank IC positive ratio: `1.0`
- stress gate pass: `false`

Negative validation quarters:

- `Q1_2020`: ann `-0.031030`
- `Q3_2020`: ann `-0.070167`
- `Q3_2021`: ann `-0.029586`

Conclusion: the candidate passed annual validation because strong quarters and strong years masked shorter-horizon fragility. A stricter validation-only stress gate would have rejected it before final test.

## H200 vs 3090 Diagnosis So Far

Evidence that H200 training itself is not the main unexplained failure:

- seed42 H200 refcheck reproduced the available seed42 validation cache exactly at prediction and label level.
- The remaining mismatch is in stage2 provenance and validation artifact reproducibility, not a proven H200 numerical nondeterminism issue.

Evidence that the frozen 3090/current-best stage2 artifact is not locally reproducible:

- the current-best frozen row `cash_quality_z_tw001__prac_m000_hold7_r085` re-evaluated locally on validation as:
  - full ann: `0.16920284380272618`
  - year ann min: `-0.2197263510226689`
  - year ann std: `0.3570822153239879`
- frozen documentation reported:
  - full ann: `0.2897001638735071`
  - year ann min: `0.2498259036668698`
  - year ann std: `0.009296576207662`

Likely causes to keep investigating with validation-only evidence:

- missing canonical old stage1 cache provenance for the frozen 3090/current-best run
- qlib/backtest data or environment drift versus the frozen 2026-06-13 artifact
- too-small validation candidate pool and annual-only CV overfitting
- stage2 strategy path sensitivity that is not captured by full/year metrics alone

## New Validation-Only Freeze Rule

Before any future holdout evaluation is considered, a candidate must pass both the old annual gate and the new stress gate.

Annual pre-gate:

- `full_ann >= 0.26`
- `year_ann_min >= 0.26`

Stress gate `v2_no_negative_valid_quarters_20260705`:

- `half_ann_min >= 0.05`
- `quarter_ann_min >= 0.0`
- `quarter_ann_positive_ratio >= 0.80`
- `quarter_rankic_min >= 0.02`
- `quarter_rankic_positive_ratio >= 1.0`

Implementation:

- script: `h200_validation_stress_audit.py`
- selection scope: `validation_only_no_test_loaded`
- test policy: `consumed_test_not_loaded_not_used_for_selection`

## Existing Candidate Stress Sweep

After adding the explicit grid prefilter to `h200_validation_stress_audit.py`, the current validation-only near-candidate pool was swept with:

- prefilter full ann: `>= 0.24`
- prefilter year ann min: `>= 0.20`
- candidate count: `6`
- output summary: `tmp/h200_validation_stress_existing_annual_prefilter6_strictgate_20260705_1_summary.json`
- output candidate table: `tmp/h200_validation_stress_existing_annual_prefilter6_strictgate_20260705_1_candidate_summary.csv`
- output segment detail: `tmp/h200_validation_stress_existing_annual_prefilter6_strictgate_20260705_1_segment_detail.csv`

Result:

- stress gate pass count: `0`

Candidate stress summary:

| combo | trial | full_ann | year_min | half_min | quarter_min | quarter_pos_ratio | stress_pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `99,11,1234` | `baseline_seq__prac_m000_hold5_r08` | `0.311539` | `0.277622` | `0.097001` | `-0.070167` | `0.750000` | `false` |
| `99,11,1234` | `baseline_seq__prac_m000_hold5_r085` | `0.298680` | `0.259653` | `0.045825` | `-0.094367` | `0.833333` | `false` |
| `99,11,1234` | `baseline_seq__prac_m000_hold5_r09` | `0.288311` | `0.233797` | `0.108194` | `-0.059747` | `0.750000` | `false` |
| `7,11,1234` | `baseline_seq__prac_m000_hold7_r09` | `0.258268` | `0.243332` | `0.112247` | `-0.222158` | `0.833333` | `false` |
| `7,11,1234` | `baseline_seq__prac_m000_hold7_r08` | `0.281154` | `0.250368` | `0.018180` | `-0.168696` | `0.833333` | `false` |
| `7,11,1234` | `baseline_seq__prac_m000_hold7_r085` | `0.272747` | `0.216365` | `0.015380` | `-0.201128` | `0.833333` | `false` |

Observed failure pattern:

- `99,11,1234` candidates have negative quarters around `Q1_2020`, `Q3_2020`, and sometimes `Q3_2021` or `Q1_2022`.
- `7,11,1234` candidates have the same negative-quarter pair: `Q4_2020,Q1_2021`.
- None of the existing annual near-candidates is eligible for another holdout evaluation under the new validation-only stress rule.

## Quarter-IC Top Combo Stage2 Check

To test whether more stable seed-level RankIC fixes stage2 robustness, a validation-only quarter-IC screen was run on 28 H200 validation seeds.

Artifacts:

- seed RankIC table: `tmp/h200_quarter_ic_screen_top12_20260705_1_seed_rankic.csv`
- partial combo RankIC table: `tmp/h200_quarter_ic_screen_top12_20260705_1_combo_rankic.csv`
- partial combo summary: `tmp/h200_quarter_ic_screen_top12_20260705_1_partial_summary.json`

The best individual seed-level quarter-IC candidates were:

| seed | valid RankIC | year_min RankIC | quarter_min RankIC | gate_pass |
| ---: | ---: | ---: | ---: | --- |
| `2032` | `0.062001` | `0.054165` | `0.030715` | `true` |
| `5678` | `0.061612` | `0.046628` | `0.031826` | `true` |
| `2036` | `0.060584` | `0.046363` | `0.026369` | `true` |

The top RankIC combos were then checked through stage2 validation-only CV:

- selector grid: `tmp/h200_selector_quarter_ic_top3_20260705_1_grid.csv`
- selector summary: `tmp/h200_selector_quarter_ic_top3_20260705_1_summary.json`
- checked combos:
  - `2032,5678,2036,2038`
  - `2032,5678,2038`
  - `2032,5678,2038,2039`

Result:

- row count: `36`
- annual gate pass count (`full_ann >= 0.26` and `year_ann_min >= 0.26`): `0`
- old frozen gate pass count (`full_ann >= 0.24`, `year_ann_min >= 0.22`, `year_ann_std <= 0.04`): `0`
- max full ann: `0.300660`
- max year_min ann: `0.144216`

Top rows by validation worst-year ann:

| combo | trial | full_ann | year_min | year_std |
| --- | --- | ---: | ---: | ---: |
| `2032,5678,2036,2038` | `baseline_seq__prac_m000_hold7_r09` | `0.207050` | `0.144216` | `0.113062` |
| `2032,5678,2036,2038` | `baseline_seq__prac_m000_hold7_r085` | `0.226867` | `0.143892` | `0.098706` |
| `2032,5678,2036,2038` | `cash_quality_z_tw001__prac_m000_hold7_r09` | `0.114609` | `0.128112` | `0.057111` |
| `2032,5678,2038` | `baseline_seq__prac_m000_hold5_r085` | `0.271400` | `0.114201` | `0.094423` |
| `2032,5678,2038` | `baseline_seq__prac_m000_hold5_r09` | `0.300660` | `0.056796` | `0.102647` |

Conclusion: stable seed-level RankIC is not sufficient. The best RankIC combos still fail stage2 PnL stability, mostly because one validation year is too weak. The selector's automatically written locked manifest uses the script's loose default rule and is not eligible for final or holdout evaluation under the current annual plus stress gate.

## 2026-07-05 Live Validation-Only Follow-up

All items in this section are validation-only partial runs. They are not frozen results and did not use the fixed test split for selection.

Running H200 seed training:

- `stage2_strict_valid_only_new8d_20260705_2_b12000_valid`
  - seeds: `2044,2045,2046,2047,2048,2049,2050,2051`
  - batch size: `12000`
  - segment: `valid`
  - summary: `tra_quant/stage1/tra_global_dual_seed_pattern_summary_stage2_strict_valid_only_new8d_20260705_2_b12000_valid.txt`
  - note: cache and IC/RankIC are available; base PnL summary fields are `None`, so these seeds must be judged through stage2 validation CV rather than base PnL ranking
- `stage2_strict_valid_only_new8e_20260705_1_b32768_valid`
  - seeds: `2052,2053,2054,2055,2056,2057,2058,2059`
  - batch size: `32768`
  - segment: `valid`
  - summary pending: `tra_quant/stage1/tra_global_dual_seed_pattern_summary_stage2_strict_valid_only_new8e_20260705_1_b32768_valid.txt`
- `stage2_strict_valid_only_new8f_20260705_1_b32768_valid`
  - seeds: `2060,2061,2062,2063,2064,2065,2066,2067`
  - batch size: `32768`
  - segment: `valid`
  - summary pending: `tra_quant/stage1/tra_global_dual_seed_pattern_summary_stage2_strict_valid_only_new8f_20260705_1_b32768_valid.txt`
- Resource usage after starting both jobs: about `36.7GB` GPU memory per H200, with two valid-only training waves running concurrently.
- Resource update after the hold6 probe finished: a third valid-only H200 wave (`new8f`) was launched because each GPU still had about `116GB` free memory.
- Resource update after `new8f` workers loaded: `new8e` and `new8f` together use about `53GB` per H200, with all eight GPUs active.

New8d RankIC-mix selector:

- running grid: `tmp/h200_selector_new8d_rankic_mix_20260705_1_grid.csv`
- input new seeds: `2044,2046,2049,2050`
- rationale: `2044` and `2046` have top-tier valid RankIC in the expanded H200 pool, while `2050/2049` are additional positive RankIC candidates; they are mixed with existing high-return seeds `99,11,2036,2038`
- status: running validation-only stage2 CV; no test input is read

Weighted seed ensemble selector:

- grid: `tmp/h200_weighted_selector_targeted10_20260705_1_cv_grid.csv`
- stopped row count: `40`
- annual gate pass count (`full_ann >= 0.26` and `year_ann_min >= 0.26`): `0`
- old frozen gate pass count: `0`
- max full ann: `0.256953`
- max year_min ann: `0.172207`
- best year-min row at snapshot: `99:0.25,11:0.25,2036:0.25,2038:0.25` + `baseline_seq__prac_m000_hold7_r08`, with `full_ann=0.060710`, `year_ann_min=0.172207`

Interpretation: seed weighting can improve worst-year consistency for some rows, but the rows with better worst-year ann have too little full-period return. Rows with full ann near `0.25` still have weak-year ann near `0.00-0.12`. This branch has no candidate eligible for stress audit yet.

Stage2 low-risk and hold6 probes:

- aborted broad stable-focused grid: `tmp/h200_selector_wide_stablefocused_20260705_1_grid.csv`
  - stopped after `3` rows because full all-profile sweep was too slow
  - max year_min ann: `0.098574`
  - gate passes: `0`
- targeted low-risk grid: `tmp/h200_selector_targeted_lowrisk_20260705_1_grid.csv`
  - stopped row count: `11`
  - max full ann: `0.337582`
  - max year_min ann: `0.103598`
  - gate passes: `0`
- hold6/mid-risk probe grid: `tmp/h200_selector_hold6_probe_20260705_1_grid.csv`
  - final row count: `14`
  - max full ann: `0.252904`
  - max year_min ann: `0.208244`
  - gate passes: `0`
  - auto-locked row: `99,11,2036,2038` + `cash_quality_z_tw001__prac_m000_hold7_r075`
  - locked-row metrics: `full_ann=0.197788`, `year_ann_min=0.208244`, `year_ann_std=0.087938`, `full_mdd=-0.184500`

Current failure pattern from these partial grids:

- `2020` is usually not the binding year.
- `2021` and `2022` alternate as the binding weak year.
- Lower risk and longer holding can preserve full ann around `0.25`, but they have not lifted the worst validation year above roughly `0.10-0.17`.
- The hold6/mid-risk probe found a more stable cash-quality row (`cash_quality_z_tw001__prac_m000_hold7_r075`, `year_ann_min=0.208244`) but its full-period ann was only `0.197788`, so it is not a viable candidate.
- This supports the current diagnosis: the H200 gap is not fixed by simple stage2 risk/hold/score-margin retuning on the existing seed pool. More seed diversity or a materially different training branch is needed before another candidate can be locked.

## Next Legal Work

Allowed:

- train more H200 seeds on train/valid only
- rank and ensemble seeds using valid IC/RankIC only
- run stage2 yearly CV on valid only
- run stress audit on valid only
- compare candidates using the annual pre-gate plus stress gate
- prepare a locked manifest for a future holdout only if a new, genuinely unseen holdout period exists

Not allowed:

- rerun the fixed `2023-01-01` to `2026-04-01` test for model selection
- use any existing final/test artifact to rank candidates
- relax validation gates after looking at consumed test behavior
- select a candidate from test-side performance

Current state:

- no candidate is frozen for another final evaluation
- the prior locked candidate is rejected by the new validation-only stress gate
- the current annual near-candidate pool also has `0` stress-gate passes
- the quarter-IC top-combo stage2 check has `0` annual-gate passes
- any future candidate must pass the new gate before a holdout evaluation can be discussed

## 2026-07-06 Strict Rerun Update

All items below are validation-only. No `test` or `final` evaluation has been run in the current strict rerun.

Current-best provenance check:

- The frozen `2026-06-13` current-best CV grid records `cash_quality_z_tw001__prac_m000_hold7_r085` as:
  - full ann: `0.289700`
  - worst-year ann: `0.249826`
  - year ann std: `0.009297`
- Re-evaluating the locally archived old validation caches from `tmp/stage1_low_deterministic_cache_20260620_192408/` does not reproduce that frozen validation row:
  - full ann: `0.169203`
  - worst-year ann: `-0.219726`
  - half ann min: `-0.548563`
  - quarter ann min: `-0.884989`
  - quarter positive ratio: `0.666667`
- Therefore the local old-cache re-evaluation cannot be used to calibrate the H200 validation gate. This reinforces the provenance diagnosis: the canonical 3090/current-best stage1 validation cache or environment is not fully reproducible from the currently available local artifacts.

Updated stress audit on the high annual H200 candidates:

- running artifact: `tmp/h200_stress_validation_scan_rerun_updated_20260706_1_candidate_summary.csv`
- observed pass count so far: `0`
- strongest annual candidates still fail because validation quarters remain negative.
- representative rows:
  - `99,11,1234 + baseline_seq__prac_m000_hold5_r08`: full ann `0.311539`, worst-year ann `0.277622`, quarter ann min `-0.070167`, quarter positive ratio `0.750000`
  - `99,11,1234 + value_quality_hibonus_hq085_b002__prac_m000_hold5_r085`: full ann `0.355078`, worst-year ann `0.317169`, quarter ann min `-0.082714`, quarter positive ratio `0.833333`
  - `99,11,1234 + value_quality_hibonus_hq08_b002__prac_m000_hold5_r085`: full ann `0.309215`, worst-year ann `0.286646`, quarter ann min `-0.102562`, quarter positive ratio `0.833333`

Stopped low-value validation-only branches:

- `tmp/h200_quarter_stability_hq085_first_20260706_1_grid.csv`
  - stopped after hold7 hq085 rows all failed.
  - best observed year ann min was about `0.175`, and quarter ann min stayed below `-0.36`.
- `tmp/h200_weighted_selector_core1234_2036_2050_blend_20260706_1_cv_grid.csv`
  - stopped after the first blended weight showed smooth but low-return behavior.
  - best observed worst-year ann was `0.221483`, but full ann was only `0.182203`, so it did not approach the annual pre-gate.

Running H200 valid-only seed waves:

- `stage2_strict_valid_only_new8l_20260706_1_b32768_valid`: seeds `2108-2115`
- `stage2_strict_valid_only_new8m_20260706_1_b32768_valid`: seeds `2116-2123`
- `stage2_strict_valid_only_new8n_20260706_1_b32768_valid`: seeds `2124-2131`
- These waves keep all eight H200 GPUs active with high memory occupancy. No test caches or test summaries are loaded by these jobs.
