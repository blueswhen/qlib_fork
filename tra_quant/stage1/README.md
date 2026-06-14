# tra_quant/stage1

This directory is the minimal TRA-only stage1 package for the current best
TRA baseline.

## Frozen Best

- baseline: old-provenance 3-seed TRA rank ensemble
- search space: `tra-best-72`
- selected strategy: `prac_m000_hold3_r085`
- account: `150000`
- topk: `5`
- benchmark: `SH000300`
- instruments: `csi300`
- validation rows: `287568`
- test rows: `232836`
- test ann: `0.16268627636270552`
- test ir: `1.0552511615852953`
- test mdd: `-0.23068313292472836`

The six required seed caches are kept in `tra_cache/`. Non-best historical
artifacts were intentionally removed from this directory.

## Reproduce TRA-Only Best

```bash
cd /home/blueswhen/DL/qlib_fork/tra_quant/stage1
PYTHONPATH=/home/blueswhen/DL/qlib_fork:$PYTHONPATH python repro_rank_ensemble_single_strategy.py \
  --strategy-trial prac_m000_hold3_r085 \
  --validation-summary-path tmp/rank_ensemble_old6_validation_summary_20260517.txt \
  --final-test-summary-path tmp/rank_ensemble_old6_final_test_summary_20260517.txt \
  --output-path tmp/repro_rank_ensemble_single_strategy_v2.json
```

Expected final test:

```text
test_ann=0.16268627636270552
test_ir=1.0552511615852953
test_mdd=-0.23068313292472836
```

## Optional Search

To rerun the 72-strategy validation search over the archived seed caches:

```bash
cd /home/blueswhen/DL/qlib_fork/tra_quant/stage1
./run_rank_ensemble_search.sh \
  --strategy-profile tra-best-72 \
  --validation-summary-path tmp/rank_ensemble_old6_validation_summary_20260517.txt \
  --final-test-summary-path tmp/rank_ensemble_old6_final_test_summary_20260517.txt \
  --output-prefix tmp/rank_ensemble_old6_tra_best_72
```

## Data

Qlib Alpha360 data is expected at:

```text
../../training_data/cn_data_latest
```

The scripts map old absolute cache paths into this local stage1 directory via
`tra_local_helpers.py`.

## Kept Artifacts

- `TRA_best_baseline.md`
- `docs/TRA_best_baseline.md`
- `tmp/repro_rank_ensemble_single_strategy_v2.json`
- `tmp/rank_ensemble_old6_validation_summary_20260517.txt`
- `tmp/rank_ensemble_old6_final_test_summary_20260517.txt`
- `tmp/rank_ensemble_300_180_72_comparison_20260517.json`
- `tra_cache/*.pkl` for the six selected seed validation/test caches
