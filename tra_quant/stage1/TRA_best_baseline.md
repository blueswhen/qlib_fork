# TRA Best Baseline

This file freezes the current TRA-only stage1 baseline.

## Result

- baseline: old-provenance 3-seed TRA rank ensemble
- search space: `tra-best-72`
- selected strategy: `prac_m000_hold3_r085`
- validation score: `266.0988237852937`
- validation ann: `0.2194366933519968`
- validation ir: `1.1263637944285017`
- validation mdd: `-0.2085999506707754`
- test ann: `0.16268627636270552`
- test ir: `1.0552511615852953`
- test mdd: `-0.23068313292472836`
- valid common rows: `287568`
- test common rows: `232836`

## Fixed Conditions

- train: `2012-01-01` to `2019-12-31`
- valid: `2020-01-01` to `2022-12-31`
- test: `2023-01-01` to `2026-04-01`
- benchmark: `SH000300`
- instruments: `csi300`
- account: `150000`
- topk: `5`

## Strategy Space

The frozen best uses the 72-strategy subset:

- hold_thresh: `3`, `4`, `5`
- n_drop: `1`, `2`, `3`
- risk_degree: `0.85`, `0.95`
- score_margin: `0`, `0.0025`, `0.005`, `0.01`

Selection sorts by:

```text
validation_score
validation_with_cost_ann_return
validation_with_cost_ir
validation_with_cost_mdd
```

all descending.

## Authority

The authoritative artifacts are:

- `tmp/repro_rank_ensemble_single_strategy_v2.json`
- `tmp/rank_ensemble_old6_validation_summary_20260517.txt`
- `tmp/rank_ensemble_old6_final_test_summary_20260517.txt`
- `tmp/rank_ensemble_300_180_72_comparison_20260517.json`

The six referenced TRA seed caches are kept in `tra_cache/`.

## Reproduce

```bash
cd /home/blueswhen/DL/qlib_fork/tra_quant/stage1
PYTHONPATH=/home/blueswhen/DL/qlib_fork:$PYTHONPATH python repro_rank_ensemble_single_strategy.py \
  --strategy-trial prac_m000_hold3_r085 \
  --validation-summary-path tmp/rank_ensemble_old6_validation_summary_20260517.txt \
  --final-test-summary-path tmp/rank_ensemble_old6_final_test_summary_20260517.txt \
  --output-path tmp/repro_rank_ensemble_single_strategy_v2.json
```

This result is the stage1 input used by the current stage2 best baseline.
