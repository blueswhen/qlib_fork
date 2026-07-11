# Current Best Baseline

更新日期：2026-06-13

本文冻结当前 `tra_quant/stage2` 下最新的 best baseline。当前最佳方案是以 TRA best 3-seed rank ensemble 作为 stage1，再在 stage2 中加入一个很轻的 cash-quality 基本面线性微融合，并用 validation-only 年度 CV 规则选择策略。

## 0. 当前冻结结果

最终冻结方案：

- stage1: `rank_ensemble_3seed_tra72_old6`
- stage2 signal: `cash_quality_z_tw001`
- stage2 signal kind: `cash_quality_linear`
- stage2 strategy: `prac_m000_hold7_r085`
- n_drop: `1`
- risk_degree: `0.85`
- kwargs_extra: `{'score_margin': 0.0, 'hold_thresh': 7, 'slot_budget_ratio': 1.0}`

最终 test 结果：

- test_ann = `0.2643467300531994`
- test_ir = `1.8243593079319251`
- test_mdd = `-0.09423283807818256`
- test_IC = `0.04231385517278539`
- test_Rank_IC = `0.06442601943935405`

对比：

- 本轮目标 test_ann = `0.22004029352797247`
- TRA best test_ann = `0.16268627636270552`
- 当前冻结结果超过目标和 TRA best。

重复复核结果：

- rerun test_ann = `0.2643467300531994`
- rerun test_ir = `1.8243593079319258`
- rerun test_mdd = `-0.09423283807818245`

两次运行只有浮点尾数级别差异，结果可复现。

## 1. 固定条件

当前 baseline 沿用以下固定条件：

- window_key: `w1`
- train: `2012-01-01` to `2019-12-31`
- valid: `2020-01-01` to `2022-12-31`
- test: `2023-01-01` to `2026-04-01`
- benchmark: `SH000300`
- instruments: `csi300`
- account: `150000`
- topk: `5`

也就是说，量化资金仍是 15w，股票持有数量仍是不超过 5 只。

## 2. 可靠性 Review

结论：作为当前工程 baseline 可以通过，但需要保留一个边界说明。

通过点：

- stage2 CV 搜索脚本 [run_stage2_signal_cv_search.py](run_stage2_signal_cv_search.py) 只加载 validation cache，不读取 test cache。
- CV 搜索范围为 validation 内部的 `valid_all`、`2020`、`2021`、`2022`。
- final evaluator [evaluate_stage2_cv_selected_final.py](evaluate_stage2_cv_selected_final.py) 先从 CV grid 用固定规则选出唯一候选，再读取 test cache 做一次最终评估。
- 最终 summary 明确记录 `selection_scope=validation_cv_only` 和 `test_usage_policy=single_final_evaluation_after_cv_candidate_fixed`。
- 选中候选不是按 test 指标排序出来的，而是按 validation CV 规则自动选出。

需要保留的边界：

- 本轮研究过程中，曾先对一个 value-quality 过滤候选做过 final test，结果未达标；随后才继续做更稳健的 CV 搜索。因此从严格学术 holdout 角度看，整个研究过程不是“从未看过 test”的完全纯净流程。
- 当前冻结结果没有用 test 在多个成功候选之间挑选，但后续不应继续以 test 反馈反复改规则或换候选。
- 基本面 parquet 文件需要在生产前继续确认 PIT / 公告日延迟处理是否完全正确。当前脚本只使用已有 rankpct 基本面文件，不重新审计 tushare 原始处理链。

基本面文件：

- `../../training_data/processed/training/csi300_daily_fundamental_features_rankpct_qlib.parquet`
- rows: `1037100`
- date_min: `2012-01-04`
- date_max: `2026-04-01`
- columns: `31`

本次入选的 `cash_quality` composite 使用字段：

- plus: `fi_ocf_yoy`, `cf_n_cashflow_act`, `fi_netprofit_margin`, `fi_roe`, `fi_assets_turn`
- minus: `fi_debt_to_assets`

stage2 signal 构造：

- 先对 TRA rank ensemble score 做 daily cross-sectional zscore。
- 再对 cash_quality composite 做 daily cross-sectional zscore。
- 最终 signal = `0.99 * tra_zscore + 0.01 * cash_quality_zscore`。

这个扰动很小，属于轻微基本面质量倾斜，而不是重写 stage1 alpha。

## 3. Validation-only 选择规则

CV 搜索产物：

- [tmp/stage2_cv_robust_small_stable_core_20260613_1_cv_grid.csv](tmp/stage2_cv_robust_small_stable_core_20260613_1_cv_grid.csv)
- [tmp/stage2_cv_robust_small_stable_core_20260613_1_summary.json](tmp/stage2_cv_robust_small_stable_core_20260613_1_summary.json)

搜索结果：

- grid_rows = `510`
- success_rows = `510`
- selection_scope = `validation_only`

候选空间：

- signal_profile: `robust-small`
- strategy_profile: `stable-core`
- signal composite: `value_quality`, `growth_quality`, `cash_quality`, `quality_balance`, `large_liquid`, `low_valuation`
- linear weights: `0.005`, `0.01`, `0.02`
- low-penalty specs: `(0.10, 0.05)`, `(0.15, 0.05)`, `(0.15, 0.10)`, `(0.20, 0.10)`, `(0.20, 0.15)`
- high-bonus specs: `(0.80, 0.02)`, `(0.85, 0.02)`, `(0.85, 0.05)`, `(0.90, 0.05)`
- two-sided specs: `(0.15, 0.05, 0.85, 0.02)`, `(0.20, 0.10, 0.85, 0.02)`
- strategy hold_thresh: `4`, `5`, `7`
- strategy risk_degree: `0.80`, `0.85`
- n_drop: `1`
- score_margin: `0.0`

CV segments:

- valid_all: `2020-01-01` to `2022-12-31`
- 2020: `2020-01-01` to `2020-12-31`
- 2021: `2021-01-01` to `2021-12-31`
- 2022: `2022-01-01` to `2022-12-31`

稳定性得分：

- objective = `ann * 1000 + ir * 50 + mdd * 20`
- stability_penalty = `ann_std * 400 + ann_spread * 300 + max(0, -worst_ann) * 2000`
- stable_score = `segment_mean_score - stability_penalty`

最终 selection rule：

- `full_ann >= 0.24`
- `year_ann_min >= 0.22`
- `year_ann_std <= 0.04`
- sort by `stable_score`, `year_ann_min`, `year_ann_mean`, `full_ann` descending

通过 selection rule 的 3 个候选：

| trial_name | stable_score | year_ann_min | year_ann_mean | year_ann_std | full_ann | full_ir | full_mdd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cash_quality_z_tw001__prac_m000_hold7_r085` | `323.1709421685742` | `0.2498259036668698` | `0.2629536635483977` | `0.009296576207662` | `0.2897001638735071` | `1.5433981279223323` | `-0.1134970181269422` |
| `cash_quality_z_tw001__prac_m000_hold7_r08` | `320.364647...` | `0.247219...` | `0.286864...` | `0.039537...` | `0.255979...` | `1.388954...` | `-0.119797...` |
| `cash_quality_hibonus_hq08_b002__prac_m000_hold7_r08` | `300.543303...` | `0.227625...` | `0.263449...` | `0.029971...` | `0.324560...` | `1.807043...` | `-0.098314...` |

最终按规则选中：

- `cash_quality_z_tw001__prac_m000_hold7_r085`

该候选 validation 年度表现：

- valid_all ann = `0.2897001638735071`
- valid_all ir = `1.5433981279223323`
- valid_all mdd = `-0.1134970181269422`
- 2020 ann = `0.2498259036668698`
- 2021 ann = `0.2701387794725141`
- 2022 ann = `0.2688963075058091`
- year_ann_std = `0.009296576207662`

## 4. 权威产物

Stage1 TRA best 产物：

- [tmp/repro_rank_ensemble_single_strategy_v2.json](tmp/repro_rank_ensemble_single_strategy_v2.json)
- [tmp/rank_ensemble_old6_validation_summary_20260517.txt](tmp/rank_ensemble_old6_validation_summary_20260517.txt)
- [tmp/rank_ensemble_old6_final_test_summary_20260517.txt](tmp/rank_ensemble_old6_final_test_summary_20260517.txt)

Stage2 CV 搜索产物：

- [run_stage2_signal_cv_search.py](run_stage2_signal_cv_search.py)
- [tmp/stage2_cv_robust_small_stable_core_20260613_1_cv_grid.csv](tmp/stage2_cv_robust_small_stable_core_20260613_1_cv_grid.csv)
- [tmp/stage2_cv_robust_small_stable_core_20260613_1_summary.json](tmp/stage2_cv_robust_small_stable_core_20260613_1_summary.json)

Stage2 final test 产物：

- [evaluate_stage2_cv_selected_final.py](evaluate_stage2_cv_selected_final.py)
- [tmp/stage2_cv_cash_quality_z001_hold7r085_final_20260613_1_summary.json](tmp/stage2_cv_cash_quality_z001_hold7r085_final_20260613_1_summary.json)
- [tmp/stage2_cv_cash_quality_z001_hold7r085_final_20260613_1.txt](tmp/stage2_cv_cash_quality_z001_hold7r085_final_20260613_1.txt)

Stage2 final reproducibility check:

- [tmp/stage2_cv_cash_quality_z001_hold7r085_final_repro_20260613_1_summary.json](tmp/stage2_cv_cash_quality_z001_hold7r085_final_repro_20260613_1_summary.json)
- [tmp/stage2_cv_cash_quality_z001_hold7r085_final_repro_20260613_1.txt](tmp/stage2_cv_cash_quality_z001_hold7r085_final_repro_20260613_1.txt)

Final test cache:

- [fusion_cache/cash_quality_z_tw001__prac_m000_hold7_r085_stage2_cv_cash_quality_z001_hold7r085_final_20260613_1_test.pkl](fusion_cache/cash_quality_z_tw001__prac_m000_hold7_r085_stage2_cv_cash_quality_z001_hold7r085_final_20260613_1_test.pkl)

## 5. 复现步骤

### 5.1 从零复现 Stage2 CV 搜索

工作目录：

```bash
cd /home/blueswhen/DL/qlib_fork/tra_quant/stage2
```

运行 validation-only CV 搜索：

```bash
PYTHONPATH=/home/blueswhen/DL/qlib_fork:$PYTHONPATH python run_stage2_signal_cv_search.py \
  --output-prefix tmp/stage2_cv_robust_small_stable_core_20260613_1 \
  --result-suffix stage2_cv_robust_small_stable_core_20260613_1 \
  --signal-profile robust-small \
  --strategy-profile stable-core \
  --num-shards 8 \
  --launch-shards \
  --force-recompute
```

期望产物：

```text
tmp/stage2_cv_robust_small_stable_core_20260613_1_cv_grid.csv
tmp/stage2_cv_robust_small_stable_core_20260613_1_summary.json
```

期望 summary：

```text
run_mode=stage2_signal_cv_validation_merged
selection_scope=validation_only
grid_rows=510
success_rows=510
```

### 5.2 复现 Final Test

运行默认 final evaluator：

```bash
PYTHONPATH=/home/blueswhen/DL/qlib_fork:$PYTHONPATH python evaluate_stage2_cv_selected_final.py
```

期望输出：

```text
selected_trial=cash_quality_z_tw001__prac_m000_hold7_r085
test_ann=0.2643467300531994
test_ir=1.8243593079319251
test_mdd=-0.09423283807818256
```

默认 evaluator 会：

1. 读取 CV grid。
2. 按固定 selection rule 选择唯一候选。
3. 读取 stage1 final test seed cache。
4. 重建 `cash_quality_z_tw001` test signal。
5. 用 `prac_m000_hold7_r085` 做一次 final test。
6. 写出 summary。

### 5.3 复现性复核

可用不同 output suffix 复跑同一固定候选：

```bash
PYTHONPATH=/home/blueswhen/DL/qlib_fork:$PYTHONPATH python evaluate_stage2_cv_selected_final.py \
  --output-prefix tmp/stage2_cv_cash_quality_z001_hold7r085_final_repro_20260613_1 \
  --result-suffix stage2_cv_cash_quality_z001_hold7r085_final_repro_20260613_1
```

期望复核结果：

```text
test_ann=0.2643467300531994
test_ir=1.8243593079319258
test_mdd=-0.09423283807818245
```

## 6. 注意事项

1. 不要再基于 test 结果继续调 stage2 参数。

当前 test 已经被用于最终确认。如果继续调 signal / strategy / selection rule，应建立新的 out-of-sample 区间、走 paper trading，或者明确把当前 test 降级为研究集。

2. 不要把纯 stable_score 第一名直接当 best。

CV grid 中纯 stable_score 第一是 `low_valuation_lowpen_lq02_p015__prac_m000_hold7_r08`，但它的 `full_ann=0.219605`，没有通过当前冻结 selection rule 的 `full_ann >= 0.24`。冻结规则是先过门槛再排序。

3. 不要混用不同 provenance 的 stage1 cache。

当前 stage1 是 old provenance 下的 3-seed TRA rank ensemble。validation 和 test cache 都来自：

- [tmp/rank_ensemble_old6_validation_summary_20260517.txt](tmp/rank_ensemble_old6_validation_summary_20260517.txt)
- [tmp/rank_ensemble_old6_final_test_summary_20260517.txt](tmp/rank_ensemble_old6_final_test_summary_20260517.txt)

4. 生产前需要额外审计基本面 PIT。

当前 stage2 使用的是已有 rankpct parquet。若要实盘化，需要确认每个财务字段在对应交易日是否只使用当时已公告的数据，不能用未来财报回填。

5. 搜索性能。

[run_stage2_signal_cv_search.py](run_stage2_signal_cv_search.py) 支持 `--launch-shards` 和 `--num-shards`。本次用 8 shards 跑完整 510 候选，CPU 使用率正常，不再是低 CPU 搜索。
