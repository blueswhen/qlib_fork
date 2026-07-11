# H200 Stage2 Validation-Only Follow-up - 2026-07-06

## Protocol

- The fixed `2023-01-01` to `2026-04-01` test split has been consumed by prior final-once audits.
- None of the artifacts in this note loads or ranks by fixed-test results.
- New selection work is restricted to validation caches and validation backtests.
- Any future final/holdout evaluation must be pre-registered before the holdout data is read.

## New Validation-Only Diagnostics

Added script:

- `h200_validation_seed_pool_diagnostics.py`

The script refuses summaries containing non-empty `seed_runs_test` or `test_base_runs`, then computes seed-level validation IC/RankIC stability over:

- full validation: `2020-01-01` to `2022-12-31`
- validation years: `2020`, `2021`, `2022`
- validation quarters: `Q1_2020` through `Q4_2022`

Outputs:

- `tmp/h200_validation_seed_pool_diagnostics_20260706_1_seed_metrics.csv`
- `tmp/h200_validation_seed_pool_diagnostics_20260706_1_segment_detail.csv`
- `tmp/h200_validation_seed_pool_diagnostics_20260706_1_pair_corr.csv`
- `tmp/h200_validation_seed_pool_diagnostics_20260706_1_combo_proposals.csv`
- `tmp/h200_validation_seed_pool_diagnostics_20260706_1_summary.json`

Result:

- seed count: `116`
- seed stability gate pass count: `13`
- top stable seeds by validation-only seed stability score:
  - `2032`
  - `5678`
  - `2127`
  - `2050`
  - `2038`
  - `2036`
  - `2044`
  - `2049`
  - `2046`
  - `2039`
  - `1234`
  - `99`
  - `2040`

Key observation:

- Several new H200 seeds have strong full-validation RankIC, but most fail quarter stability.
- The standout new seed is `2127`, which passed the seed stability gate.
- `2116` has very high full RankIC but missed the quarter threshold narrowly: quarter RankIC min `0.014956` versus the diagnostic gate `0.015`.

## Seed-Stable Combo Scout

The seed diagnostic proposed high-stability combos such as:

- `2032,5678,2038`
- `2032,5678,2036`
- `2032,5678,2050`
- `2032,5678,2127`
- `2032,5678,2049`

A narrowed stage2 annual/year scout was launched with validation-only summaries and then stopped early because the first completed rows were far from the annual gate and the full scout was slow.

Partial artifacts:

- `tmp/h200_selector_seedstable_core_scout_20260706_1_partial_merged.csv`
- `tmp/h200_selector_seedstable_core_scout_20260706_1_partial_summary.json`

Completed partial rows:

| combo | trial | full_ann | year_ann_min | year_ann_std |
| --- | --- | ---: | ---: | ---: |
| `2032,5678,2038` | `baseline_seq__prac_m000_hold5_r085` | `0.271400` | `0.114201` | `0.094423` |
| `2032,5678,2127` | `baseline_seq__prac_m000_hold5_r085` | `0.207891` | `0.031722` | `0.178639` |
| `2032,5678,2050` | `baseline_seq__prac_m000_hold5_r085` | `0.125495` | `0.022799` | `0.123742` |
| `2032,5678,2036` | `baseline_seq__prac_m000_hold5_r085` | `0.024601` | `-0.135370` | `0.080805` |

Result:

- annual gate `full_ann >= 0.26` and `year_ann_min >= 0.26`: `0` passes.
- The best partial full-ann row still has worst-year ann only `0.114201`.
- This reinforces the earlier finding: seed-level RankIC stability is not sufficient to produce stable stage2 PnL.

## Current Diagnosis

The H200 stage2 failure is best explained by a combination of these validation-only facts:

1. The frozen current-best stage2 provenance is not locally reproducible.
   - H200 seed42 can reproduce the available old seed42 validation cache exactly.
   - But the frozen `cash_quality_z_tw001__prac_m000_hold7_r085` validation row does not reproduce under the currently available local old-cache/environment path.
   - This points to missing canonical old6 cache provenance or qlib/backtest/data environment drift, not a proven H200 numerical issue.

2. Annual validation metrics are too coarse.
   - Existing high annual H200 candidates have strong full/year validation metrics but fail quarter stress.
   - The best current-valid row `99,11,1234 + value_quality_hibonus_hq085_b002__prac_m000_hold5_r085` had full ann `0.355078` and year ann min `0.317169`, but quarter ann min `-0.082714`.

3. Seed-level IC stability does not transfer reliably to stage2 PnL.
   - The new seed-pool screen found 13 seed-stability pass seeds.
   - Early stage2 scout rows from those seeds still showed weak worst-year PnL.

4. H200 b32768 later seed waves are mostly not useful for stable PnL.
   - Most `2068-2131` seeds have acceptable full RankIC but negative quarter RankIC minima.
   - The `2127` exception is worth retaining in future combo pools, but the first `2032,5678,2127` stage2 row was not promising.

## Next Validation-Only Method

Do not continue broad annual CV grids until a better prefilter is added. The next selector should require all of the following before running expensive quarter stress:

- seed combo prefilter:
  - at least two seed-stability-gate pass seeds
  - combo mean full RankIC high enough to avoid low-return smooth blends
  - pairwise rank-signal correlation not too high, to avoid duplicated seeds
- annual PnL scout:
  - `full_ann >= 0.24`
  - `year_ann_min >= 0.18`
  - `year_ann_std <= 0.08`
- stress candidate gate:
  - run `h200_validation_stress_audit.py` only on annual-scout survivors
  - keep the strict stress gate unchanged unless there is a validation-only reason to change it before any holdout is viewed

## Future Holdout Plan

Because the fixed test has already been consumed, the next effective holdout must be genuinely unseen at selection time.

Valid future holdout protocol:

1. Freeze the selector code, validation gates, candidate row, seed cache list, and manifest before reading the holdout.
2. Use a new chronological period not included in the consumed test, preferably data after `2026-04-01`.
3. Run the holdout exactly once for the frozen manifest.
4. Do not adjust seeds, signals, strategy, thresholds, or gates after viewing that holdout.

Development-only proxy until new holdout data exists:

- Use nested validation splits inside `2020-2022`.
- Treat `2022` or leave-one-year-out validation as a proxy stress check only.
- Do not call this proxy a final result and do not compare it as if it were a new test.

## TODO

- Build a faster annual-scout driver that can precompute candidate signals once per combo and evaluate a smaller strategy set without repeated slow startup.
- Add combo-level validation IC stability for weighted rank ensembles, not just equal-weight rank ensembles.
- Re-run a targeted stage2 scout only for combos passing the stricter seed-combo prefilter.
- Keep `99,11,1234` as an audit/reference combo only unless it passes the strict stress gate in a future validation-only rerun.
- Locate or reconstruct the canonical old6/current-best stage1 validation cache provenance before trying to explain the 3090 baseline as a hardware reproducibility issue.
