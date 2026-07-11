# Current Best Baseline

更新日期：2026-07-11

本文冻结当前 `tra_quant` 的 best baseline。该结果来自严格从零训练：

```text
strict_zero_runs/strict_zero_h200_20260711_1245
```

这次 run 不依赖历史模型、历史 prediction cache、历史 CV grid、历史 lock
manifest 或历史 final result。流程是 validation-only 调参与锁定，锁定后才训练
test 侧 Stage1 cache，并只做一次 final test。

## 0. 当前冻结结果

最终冻结方案：

- stage1: `rank_ensemble_h200_strict_validation_locked_5678_2050_2044_b12000`
- stage1 seeds: `5678,2050,2044`
- stage2 signal: `cash_quality_z_tw001`
- stage2 signal kind: `fused_stage2`
- stage2 signal profile: `cash-quality-quick`
- stage2 strategy profile: `stable-cash-quick`
- stage2 strategy: `prac_m000_hold7_r095`
- n_drop: `1`
- risk_degree: `0.95`
- kwargs_extra: `{'score_margin': 0.0, 'hold_thresh': 7, 'slot_budget_ratio': 1.0}`

最终 test 结果：

- test_ann = `0.2804402793535583`
- test_ir = `1.6660541147890104`
- test_mdd = `-0.10195254346954019`
- test_IC = `0.038227402896315404`
- test_Rank_IC = `0.059762638598347115`

对比：

- previous current best test_ann = `0.2643467300531994`
- new current best test_ann = `0.2804402793535583`
- improvement = `0.0160935493003589`

## 1. 固定条件

- window_key: `w1`
- train: `2012-01-01` to `2019-12-31`
- valid: `2020-01-01` to `2022-12-31`
- test: `2023-01-01` to `2026-04-01`
- benchmark: `SH000300`
- instruments: `csi300`
- account: `150000`
- topk: `5`
- batch_size: `12000`

资金 `150000` 和 `topk=5` 是硬约束。

## 2. Validation-only 锁定结果

锁定候选：

```text
cash_quality_z_tw001__prac_m000_hold7_r095
```

验证集 full 段：

- valid_ann = `0.25664687372949496`
- valid_ir = `1.2771148971848802`
- valid_mdd = `-0.23370029372504092`
- valid_IC = `0.06389892630367724`
- valid_Rank_IC = `0.07467096363047035`

验证集年度：

| segment | ann | IR | MDD |
| --- | ---: | ---: | ---: |
| `Y2020` | `0.4309386668820641` | `2.343423069664322` | `-0.12842425907618166` |
| `Y2021` | `0.38044939750308976` | `1.5778897257065463` | `-0.14052874048819816` |
| `Y2022` | `0.39060580082635143` | `2.2051574154260085` | `-0.09854233431628101` |

压力检查摘要：

- full_ann = `0.2566468737294949`
- year_ann_min = `0.3804493975030897`
- half_ann_min = `0.0941108455275603`
- quarter_ann_min = `-0.1960339739329046`
- quarter_positive_ratio = `0.8333333333333334`
- quarter_rankic_min = `0.0278415316381671`
- quarter_rankic_positive_ratio = `1.0`
- stress_score = `317.68064737683267`

该候选是按 validation stress score 选中，不是按 test 结果挑选。

## 3. Stage1 单 seed 验证表现

| seed | valid ann | valid IR | valid MDD |
| ---: | ---: | ---: | ---: |
| `5678` | `0.19574199151767951` | `1.0428396357756848` | `-0.29466473738825694` |
| `2050` | `0.05916138523263426` | `0.33415741395543624` | `-0.2541195845761003` |
| `2044` | `0.11577685943276218` | `0.5933170235379485` | `-0.2714771918682022` |

## 4. 权威产物

当前 best formal run：

- `../strict_zero_runs/strict_zero_h200_20260711_1245/from_zero_manifest.json`
- `../strict_zero_runs/strict_zero_h200_20260711_1245/pretest_checkpoint.json`
- `../strict_zero_runs/strict_zero_h200_20260711_1245/completion_audit.json`
- `../strict_zero_runs/strict_zero_h200_20260711_1245/stage2/tmp/locked_candidate_pretest_immutable.json`
- `../strict_zero_runs/strict_zero_h200_20260711_1245/stage2/tmp/final_test_once_summary.json`
- `../strict_zero_runs/strict_zero_h200_20260711_1245/stage2/tmp/final_test_once_final_audit.json`

关键哈希：

```text
locked_candidate_pretest_immutable.json sha256 =
9060ff790b3f63de37539b479a3c3bd5b51404823f62622bc4695911a5a1408c

final_test_once_summary.json sha256 =
b5bf219057993ff4847f75ace63705df060e7251cc1d36efb178a598d726621c
```

## 5. 一键从零复现

从 `tra_quant` 目录执行：

```bash
cd /path/to/qlib_fork/tra_quant
./train_strict_fusion_from_zero.sh --run-id strict_zero_repro_YYYYMMDD_1
```

双卡 3090 推荐命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 ./train_strict_fusion_from_zero.sh \
  --run-id strict_zero_3090_YYYYMMDD_1 \
  --gpu-slots 0,1 \
  --jobs-per-gpu 1 \
  --stage2-workers 2
```

H200 推荐命令：

```bash
./train_strict_fusion_from_zero.sh --run-id strict_zero_h200_YYYYMMDD_1
```

完整训练说明见：

```text
../STRICT_FROM_ZERO_RETRAIN.md
```

## 6. 可靠性说明

通过点：

- 本次 formal run 使用全新 run 目录。
- Stage1 validation cache、Stage2 validation grid、locked manifest、Stage1
  test cache 和 final test cache 都写在同一个 run 目录内。
- final test 在 `locked_candidate_pretest_immutable.json` 写入并生成 SHA256
  后才执行。
- final test summary 明确记录：
  - `selection_scope=strict_validation_only_locked_before_test`
  - `test_usage_policy=single_test_after_locked_manifest`
- 选择过程只看 validation stress gate，不看 test。

边界说明：

- 当前 test 区间仍是 `2023-01-01` 到 `2026-04-01`。之后若继续围绕 test
  反馈调参，不能再把同一 test 当作纯 holdout。
- 当前 run 证明该 0.28 结果可以在 H200 上从零复现。若要声称跨机器高置信度，
  需要在双卡 3090 上用同一脚本再跑一次，并比较 `completion_audit.json`。
