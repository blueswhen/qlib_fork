# Next Paths Without Model Switch

This file defines the remaining active experiment scope after freezing the current best baseline.

## Frozen Baseline

Keep fixed:

- Alpha360 -> TRA
- TechAlpha158A3DFundamental(rankpct) -> XGBoost(base)
- zscore fusion
- seq/tab weight = 0.85 / 0.15

Reference:

- [CURRENT_BEST_BASELINE.md](CURRENT_BEST_BASELINE.md)

Historical failures, removed routes, and non-promotable attempts are tracked only in [失败.md](失败.md).

## Completed Since Replan

### R1. Broad Candidate Pre-Filter Then Baseline Rank

Status:

- completed
- result: validation winner fell back to baseline_no_prefilter
- archived in [失败.md](失败.md)

### R2. Market-Regime Exposure Schedule

Status:

- completed
- result: trend-aware risk schedule reduced MDD but crushed ann_return and IR
- archived in [失败.md](失败.md)

### R3. Robust Strategy Retune With Year-Consistency Objective

Status:

- completed
- result: rolling year-consistency selected tw005, which improved test IR slightly but worsened MDD too much to replace the baseline
- archived in [失败.md](失败.md)

## Remaining High-Potential Paths

No active paths remain under the no-model-switch constraint.

## Execution Order

1. R1 broad pre-filter
2. R3 robust strategy retune
3. R2 regime exposure schedule on the R3 winner

## Stop Rule

Stop rule triggered: all remaining no-model-switch paths have now been executed and none replaced the current best baseline on final test. Optimization on this csi300 line is closed.