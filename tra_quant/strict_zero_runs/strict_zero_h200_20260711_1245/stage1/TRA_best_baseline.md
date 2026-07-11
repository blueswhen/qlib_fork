# TRA Best Baseline

This file freezes the current TRA-only stage1 baseline.

## Result

- baseline: H200 4-seed TRA rank ensemble, batch size 12000
- search space: `tra-best-72`
- selected strategy: `prac_m000_drop3_hold4_r095`
- validation score: `184.427549854415`
- validation ann: `0.2228939772037148`
- validation ir: `1.142295456945299`
- validation mdd: `-0.2169826992264823`
- test ann: `0.17879091198902958`
- test ir: `1.1453728955617313`
- test mdd: `-0.16411547125132675`
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

- `../runs/stage1_h200_20260621_11_4seed_b12000_fast/stage1/tmp/repro_rank_ensemble_h200_4seed_b12000_tra72.json`
- `../runs/stage1_h200_20260621_11_4seed_b12000_fast/stage1/tmp/rank_ensemble_h200_validation_summary_stage1_h200_20260621_11_4seed_b12000_fast.txt`
- `../runs/stage1_h200_20260621_11_4seed_b12000_fast/stage1/tmp/rank_ensemble_h200_final_test_summary_stage1_h200_20260621_11_4seed_b12000_fast.txt`
- `../runs/stage1_h200_20260621_11_4seed_b12000_fast/stage1/tmp/rank_ensemble_h200_search_stage1_h200_20260621_11_4seed_b12000_fast_summary.json`
- `../runs/stage1_h200_20260621_11_4seed_b12000_fast/stage1/tmp/rank_ensemble_h200_search_stage1_h200_20260621_11_4seed_b12000_fast_validation_grid.csv`

The four referenced TRA seed caches are under the run-local `stage1/tra_cache/`.

The old-provenance 3-seed TRA rank ensemble remains the historical stage1 input
behind the frozen stage2 best baseline, but it is no longer the current
TRA-only stage1 best.

## Seed42 Retraining Correction

The authoritative seed42 cache must be retrained through the global/refcheck
configuration, not `tune_tra_alpha360_strict.py`. The strict entry omits the
frozen global model override used by the baseline, most importantly
`eval_freq=5`; using it can reproduce the labels but produce different
predictions.

Correct command:

```bash
cd /home/blueswhen/DL/qlib_fork/tra_quant/stage1
export PYTHONPATH=/home/blueswhen/DL/qlib_fork/tra_quant/stage1:/home/blueswhen/DL/qlib_fork:$PYTHONPATH
export QLIB_PROVIDER_URI=/home/blueswhen/DL/qlib_fork/training_data/cn_data_latest
python retrain_seed42_refcheck.py \
  --suffix seed42_refcheck_retrain_20260620_1 \
  --gpu-slots 0,1 \
  --provider-uri /home/blueswhen/DL/qlib_fork/training_data/cn_data_latest
```

Then verify with:

```bash
python verify_seed42_refcheck.py --suffix seed42_refcheck_retrain_20260620_1
```

Verified seed42 valid reproduction on 2026-06-20:

```text
IC=0.02757476988385409
Rank IC=0.04134057587363596
pred max_abs=0.0
label max_abs=0.0
```

## Reproduce

```bash
cd /home/niushengxiao/qlib_fork
python tra_quant/retrain_stage1_h200.py \
  --run-id stage1_h200_20260621_11_4seed_b12000_fast \
  --seeds 42,2026,3407,7 \
  --gpu-slots 0,1,2,3,4,5,6,7 \
  --valid-gpu-slots 0,1,2,3 \
  --test-gpu-slots 4,5,6,7 \
  --batch-size 12000 \
  --max-concurrent-seeds 2 \
  --run-search \
  --strategy-profile tra-best-72 \
  --search-num-shards 16
```

This result was also evaluated as the stage1 input for the H200 stage2
follow-up run `stage2_cv_h200_4seed_b12000_tra72_20260621_1`.
