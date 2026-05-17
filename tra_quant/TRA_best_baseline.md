# TRA Best Baseline

本文件只记录 examples/my_strategy 目录下和 TRA baseline 直接相关的内容，目标是把以下 4 类信息一次性冻结清楚：

- 运行脚本是谁，分别负责什么
- 训练后会产出哪些 summary、cache、mlruns artifact
- seed 是怎么使用的，哪些结果是单 seed，哪些是双 seed / 三 seed
- 哪些 baseline 看起来名字接近，但底层 cache provenance 实际不同

如果后续再做 TRA 复现、rank ensemble、multiseed ensemble、或者做 cache forensic，先看这份文档，再决定跑哪个脚本。

## 1. 固定数据切分与共用常量

当前 TRA 主线默认使用：

- window_key: w1
- train: 2012-01-01 to 2019-12-31
- valid: 2020-01-01 to 2022-12-31
- test: 2023-01-01 to 2026-04-01
- benchmark: SH000300
- instruments: csi300
- account: 150000
- topk: 5

在 [tune_tra_alpha360_global.py](tune_tra_alpha360_global.py) 中冻结的核心 TRA 配置：

- model_trial: strict_tra_lstm_s240_h5
- model_key: tra_lstm_base
- step: 240
- label: Ref($close, -6) / Ref($close, -1) - 1
- dataset batch_size: 16384
- validation_score_mode: valid_yearly_stability_v1

Stage-1 策略搜索空间在 [tune_tra_alpha360_global.py](tune_tra_alpha360_global.py) 中定义为：

- hold_thresh: 3 / 4 / 5
- n_drop: 1..5
- risk_degree: 0.70 / 0.85 / 0.95
- score_margin: 0 / 0.0025 / 0.005 / 0.01

但需要单独强调：TRA_best_baseline 真正使用的不是上面这个完整 Stage-1 空间，而是一个过滤后的 72 策略子空间。

TRA_best_baseline 的真实 72 搜索空间是：

- hold_thresh: 3 / 4 / 5
- n_drop: 1 / 2 / 3
- risk_degree: 0.85 / 0.95
- score_margin: 不限，沿用全部 4 个 margin 值 0 / 0.0025 / 0.005 / 0.01

总数计算：

- 3 x 3 x 2 x 4 = 72

也就是说，TRA_best_baseline 的策略筛选规则本质上是：

- hold 只保留 3 / 4 / 5
- drop 只保留 1 / 2 / 3
- risk 只保留 0.85 / 0.95
- 其余不额外限制，score_margin 四档全部保留

而 3-seed rank ensemble 的 300 空间在 [run_rank_ensemble_tra_alpha360_once.py](run_rank_ensemble_tra_alpha360_once.py) 中扩为：

- hold_thresh: 3 / 4 / 5
- n_drop: 1..5
- risk_degree: 0.60 / 0.70 / 0.85 / 0.95 / 1.00
- score_margin: 0 / 0.0025 / 0.005 / 0.01

## 2. 关键脚本总表

### 2.1 TRA 单 seed / 双 seed 总控

1. [tune_tra_alpha360_global.py](tune_tra_alpha360_global.py)

用途：

- TRA 全局总控脚本
- 单 seed Stage-1 训练、validation 策略搜索、final test
- dual-seed study 的正式入口
- 负责写 rolling_result txt、tra_cache pkl、以及 dual-seed grid / summary

关键产物：

- tra_global_validation_results.csv
- tra_global_validation_best.txt
- tra_global_final_test.txt
- tra_global_dual_seed_valid_test_grid.csv
- tra_global_dual_seed_pattern_summary.txt
- tra_cache/rolling_cache_tra_*.pkl

1. [run_dualseed_b12000_then_seed42_refcheck.py](run_dualseed_b12000_then_seed42_refcheck.py)

用途：

- 先调用 [tune_tra_alpha360_global.py](tune_tra_alpha360_global.py) 跑 dual-seed formal
- 再单独跑 seed42 refcheck
- 主要用于把 dual-seed formal 和 seed42 对照放在同一条工作链上

这个脚本里冻结的 dual-seed formal suffix：

- dualseed_formal_seedxroll_b12000_20260514_1

这个脚本里冻结的 seed42 refcheck suffix：

- seed42_refcheck_tra18_b16384_20260514_1

### 2.2 三 seed ensemble / rank ensemble

1. [run_current_best_multiseed_ensemble.py](run_current_best_multiseed_ensemble.py)

用途：

- 以 seed 集合 [42, 2026, 3407] 为默认输入
- seq branch 上做简单平均，不是 rank average
- validation 和 test 各生成一份三 seed 平均序列信号
- 结果写入 current_best_multiseed_ensemble_validation.txt 和 current_best_multiseed_ensemble_final_test.txt

核心逻辑：

- seed42 可以直接复用既有 summary/cache
- seed2026 和 seed3407 如果已有 rolling_result txt 就直接复用，否则补跑
- 先做 seq 平均，再和当前 best 的 tabular branch 做 zscore 融合

1. [run_rank_ensemble_tra_alpha360_once.py](run_rank_ensemble_tra_alpha360_once.py)

用途：

- 对 3 个 TRA cache 做按日横截面 percentile rank
- 对 rank 结果等权平均
- 在 300 策略空间上只搜 validation
- 只对 validation winner 跑一次 test

关键产物：

- rank_ensemble_3seed_300_validation_grid.csv
- rank_ensemble_3seed_300_summary.json

注意：

- 这个脚本现在的 validation cache 是从 [current_best_multiseed_ensemble_validation.txt](current_best_multiseed_ensemble_validation.txt) 的 seed_runs_valid 动态读取
- test cache 也是从 [current_best_multiseed_ensemble_final_test.txt](current_best_multiseed_ensemble_final_test.txt) 的 seed_runs_test 动态读取
- 如果 summary 文件不存在，或者 summary 里引用的 cache_path 不存在，脚本会直接失败
- 因此它的 validation provenance 和 test provenance 都必须通过 summary 文件核对，不能只看脚本名字

1. [repro_rank_ensemble_single_strategy.py](repro_rank_ensemble_single_strategy.py)

用途：

- 只复现 1 个指定 strategy_trial
- 同时跑 validation 和 test
- 支持两种 provenance 输入方式：显式传 3 条 validation cache_path + 3 条 test cache_path；或者传 validation/test summary 路径，让脚本自动解析 seed_runs_valid / seed_runs_test
- 会把本次真正使用的 6 条 cache_path 一起写进输出 json，避免只记住 strategy_trial

默认产物：

- tmp/repro_rank_ensemble_single_strategy.json

注意：

- 这是当前最小、最稳定的单策略复现入口
- 旧的 tmp/repro_old_rank_ensemble_paths.py 只是一次性临时脚本，不应再作为正式入口

### 2.3 基线与上游关系

1. [tune_tra_alpha360_strict.py](tune_tra_alpha360_strict.py)

用途：

- 更早期的纯 TRA stage1 strict baseline 入口
- 对应 [历史成功方案.md](历史成功方案.md) 里的 H1

1. [current_best_baseline_manifest.json](current_best_baseline_manifest.json)

用途：

- 这是当前整条 current best baseline 的机器可读 manifest
- 其中 stage1_incumbent 记录纯 TRA strict baseline
- current_best_baseline 记录后续与 tabular 融合后的当前总基线

这份 manifest 不是 rank ensemble 的来源文件，但它定义了当前主线 baseline 和多 seed ensemble 的上游关系。

## 3. 产物层级

同一条 TRA 运行通常会留下 3 层信息：

### 3.1 summary txt

典型命名：

- rolling_result_tra_w1_...txt

作用：

- 记录配置
- 记录 recorder_id
- 记录 cache_path
- 记录 IC / Rank IC / ann / ir / mdd

这是人读最方便的层。

### 3.2 local cache pkl

典型命名：

- tra_cache/rolling_cache_tra_w1_...pkl

内容结构：

- pred
- label

这是后续做单模型重评估、multiseed 平均、rank ensemble 的直接输入层。

### 3.3 mlruns artifact

典型结构：

- mlruns/<experiment_id>/<run_id>/artifacts/pred.pkl
- mlruns/<experiment_id>/<run_id>/artifacts/label.pkl

用途：

- 当本地 tra_cache pkl 被删掉时，可以从 mlruns artifacts 恢复
- 2026-05-16 这次 old rank-ensemble validation 路径就是这样恢复回来的

## 4. Seed 使用规则

### 4.1 单 seed

seed 42 主要扮演：

- 早期 strict baseline 主 seed
- tra18_residual_step1_20260506_1 这条旧 baseline 的主 seed
- 三 seed ensemble 中的 incumbent seed

### 4.2 dual seed

默认 dual-seed formal 使用：

- seed 2026
- seed 3407

dual-seed 只负责补额外 seed，不替代 seed42 的历史基线地位。

### 4.3 三 seed

当前三 seed 组合默认是：

- 42
- 2026
- 3407

在 [run_current_best_multiseed_ensemble.py](run_current_best_multiseed_ensemble.py) 中，默认 seeds 就是 [42, 2026, 3407]。

## 5. 最重要的 provenance 差异

这部分必须单独看，因为最近的排查核心就是这里。

### 5.1 当前 multiseed validation provenance

来源文件：

- [current_best_multiseed_ensemble_validation.txt](current_best_multiseed_ensemble_validation.txt)

其中 seed_runs_valid 记录的 validation cache 是：

1. seed42
   - tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_strict_tra_lstm_s240_h5.pkl
2. seed2026
   - tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_multiseed_s2026_valid.pkl
3. seed3407
   - tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_multiseed_s3407_valid.pkl

这是“当前 multiseed ensemble”那条线的 validation basis。

### 5.2 旧 rank-ensemble validation provenance

这是 2026-05-16 最后成功复现 prac_m000_hold3_r085 时使用的旧 validation 三条路径：

1. seed42 old valid
   - tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_strict_tra_lstm_s240_h5_tra18_residual_step1_20260506_1.pkl
2. seed2026 old valid
   - tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_dualseed_s2026_valid_dualseed_formal_seedxroll_b12000_20260514_1.pkl
3. seed3407 old valid
   - tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_dualseed_s3407_valid_dualseed_formal_seedxroll_b12000_20260514_1.pkl

这 3 份旧 validation pkl 曾经被删掉过，后来通过以下 mlruns artifacts 恢复：

1. seed42 old valid restore source
   - examples/my_strategy/mlruns/198111276539745747/e3ab04501c21438e8ca480b8ac85a6ab/artifacts/pred.pkl
   - examples/my_strategy/mlruns/198111276539745747/e3ab04501c21438e8ca480b8ac85a6ab/artifacts/label.pkl
2. seed2026 old valid restore source
   - mlruns/943174324210839437/8a5d3ba4bd374b3a9fc297fbe1ff0a7c/artifacts/pred.pkl
   - mlruns/943174324210839437/8a5d3ba4bd374b3a9fc297fbe1ff0a7c/artifacts/label.pkl
3. seed3407 old valid restore source
   - mlruns/872892592345132733/c9f9fc21c69e4991bcd2d5f321ce2029/artifacts/pred.pkl
   - mlruns/872892592345132733/c9f9fc21c69e4991bcd2d5f321ce2029/artifacts/label.pkl

### 5.3 旧 rank-ensemble test provenance

旧 test 三条路径是：

1. seed42 old test
   - tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r085_strict_tra_lstm_s240_h5_final_test_tra18_residual_step1_20260506_1.pkl
2. seed2026 old test
   - tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_dualseed_s2026_test_dualseed_formal_seedxroll_b12000_20260514_1.pkl
3. seed3407 old test
   - tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_dualseed_s3407_test_dualseed_formal_seedxroll_b12000_20260514_1.pkl

这个 old test provenance 和 old validation provenance 必须配套使用，不能把旧 validation 和当前 multiseed validation 混着算。

## 6. 训练 / 复现流程

### 6.1 单 seed TRA baseline

流程：

1. 用 [tune_tra_alpha360_global.py](tune_tra_alpha360_global.py) 或历史 strict 入口训练 seed42 base model
2. 在 validation 上搜策略层参数
3. 选 validation winner
4. 对 winner 跑一次 final test
5. 产出 rolling_result txt + tra_cache pkl + mlruns artifact

如果讨论的是 TRA_best_baseline 这条线，那么这里的“在 validation 上搜策略层参数”需要进一步写死为 72 搜索空间，而不是泛指任意 Stage-1 空间。

TRA_best_baseline 的实际筛选方法是：

1. 先固定模型缓存
2. 只在这 72 个策略里做 validation 搜索
3. 从 validation 结果里选最优策略
4. 只对这个 validation winner 跑 1 次 test
5. 不对其他候选重复跑 test

这也是为什么文档里凡是引用 TRA_best_baseline 时，都应该默认它来自“72 策略 validation-first + test-once”这条工作流。

### 6.2 dual-seed formal

流程：

1. 用 [run_dualseed_b12000_then_seed42_refcheck.py](run_dualseed_b12000_then_seed42_refcheck.py) 调 [tune_tra_alpha360_global.py](tune_tra_alpha360_global.py) 的 --dual-seed-study
2. 固定 seeds = 2026, 3407
3. 对 valid 和 test 分别产出双 seed base cache
4. 写 tra_global_dual_seed_valid_test_grid_*.csv 和 tra_global_dual_seed_pattern_summary_*.txt

这条线不是直接挑最终 baseline，而是给后续三 seed 组合提供额外种子。

### 6.3 当前 multiseed average ensemble

流程：

1. 用 [run_current_best_multiseed_ensemble.py](run_current_best_multiseed_ensemble.py)
2. 读取 seed42 incumbent + seed2026 / seed3407 的 validation 或 test cache
3. 对 3 个 seq signal 直接做数值平均
4. 写 current_best_multiseed_ensemble_validation.txt 或 current_best_multiseed_ensemble_final_test.txt

注意：

- 这是 seq average，不是 rank average
- 它生成的 current_best_multiseed_ensemble_validation.txt 现在被 [run_rank_ensemble_tra_alpha360_once.py](run_rank_ensemble_tra_alpha360_once.py) 用作 validation provenance 输入
- 这个脚本只有在 validation_improves_current_best=True 时才会继续跑 test 并写 current_best_multiseed_ensemble_final_test.txt
- 如果 validation 没有超过 current best，它会在写完 current_best_multiseed_ensemble_validation.txt 后直接 return，不会生成 final test summary
- 所以 current_best_multiseed_ensemble_final_test.txt 缺失，不一定代表脚本没跑过，也可能只是 validation gate 没通过

### 6.4 3-seed rank ensemble 策略搜索

流程：

1. 用 [run_rank_ensemble_tra_alpha360_once.py](run_rank_ensemble_tra_alpha360_once.py)
2. validation 侧读取 3 份 cache
3. 每日横截面做 percentile rank
4. 3 个 seed 的 rank 等权平均
5. 在 300 策略空间上做 validation-only 搜索
6. 只对 validation winner 跑一次 test
7. 写 rank_ensemble_3seed_300_validation_grid.csv 和 rank_ensemble_3seed_300_summary.json

这里的 300 个策略不是人工挑出来的，而是脚本固定枚举出来的完整笛卡尔积：

- hold_thresh: 3 / 4 / 5
- n_drop: 1 / 2 / 3 / 4 / 5
- risk_degree: 0.60 / 0.70 / 0.85 / 0.95 / 1.00
- score_margin: 0.0000 / 0.0025 / 0.0050 / 0.0100

总数计算：

- 3 x 5 x 5 x 4 = 300

对应 strategy_trial 命名规则：

- `prac_{margin_tag}{drop_tag}_hold{hold}_{risk_tag}`

其中：

- margin_tag: m000 / m025 / m050 / m100
- drop_tag: n_drop=1 时为空，n_drop=2..5 时分别是 `_drop2` / `_drop3` / `_drop4` / `_drop5`
- risk_tag: 0.60 -> r06, 0.70 -> r07, 0.85 -> r085, 0.95 -> r095, 1.00 -> r10

validation 阶段的筛选条件也必须固定写清楚：

- 先把 300 个候选全部在 validation 上评估或复用已有 validation row
- 然后按 validation_score 降序排序
- 如果 validation_score 相同，则按 validation_with_cost_ann_return 降序排序
- 如果 ann 仍相同，则按 validation_with_cost_ir 降序排序
- 排名第 1 的 strategy_trial 就是 validation winner

test 阶段的规则只有一条：

- 不做 test 全搜索，只给 validation winner 跑 1 次 test

### 6.5 用当前模型缓存重跑 300 策略 TODO

这部分是接下来要执行的操作清单，目标是基于“当前模型缓存”重跑一次完整的 300 策略搜索，并且保持选择逻辑和结果口径固定。

注意：

- 这一节对应的是后续要复跑的 300 空间
- 它不是 TRA_best_baseline 当年真正使用的搜索空间
- TRA_best_baseline 的真实历史搜索空间仍然是上面单独写明的 72 空间

TODO：

1. 先确认 validation 侧使用的是 [current_best_multiseed_ensemble_validation.txt](current_best_multiseed_ensemble_validation.txt) 里的 seed_runs_valid 三条 cache_path，而不是 old rank-ensemble validation provenance。
2. 明确这次要跑满完整 300 个策略，不能抽样，不能只跑局部子集。完整集合必须严格等于以下笛卡尔积：3 个 hold_thresh x 5 个 n_drop x 5 个 risk_degree x 4 个 score_margin。
3. 具体策略集合按以下条件展开：hold_thresh ∈ {3, 4, 5}，n_drop ∈ {1, 2, 3, 4, 5}，risk_degree ∈ {0.60, 0.70, 0.85, 0.95, 1.00}，score_margin ∈ {0.0000, 0.0025, 0.0050, 0.0100}。
4. strategy_trial 命名必须保持和脚本一致，即 `prac_{margin_tag}{drop_tag}_hold{hold}_{risk_tag}`；这样跑完后才能直接和 [rank_ensemble_3seed_300_validation_grid.csv](rank_ensemble_3seed_300_validation_grid.csv) 逐行对照。
5. 运行方法固定为 validation-first：先对 300 个策略全部跑 validation，写出或复用 [rank_ensemble_3seed_300_validation_grid.csv](rank_ensemble_3seed_300_validation_grid.csv)。
6. validation winner 的筛选顺序固定为 validation_score -> validation_with_cost_ann_return -> validation_with_cost_ir，全部按降序。
7. 只在 validation winner 确定之后，再跑 1 次 test；禁止对多个候选重复跑 test，避免把 test 当成调参集。
8. 最终结果至少要同时记录 4 类信息：validation basis 的 3 条 cache_path、test basis 的 3 条 cache_path、validation winner 的 strategy_trial、以及这 1 次 test 的结果。
9. 如果中途发现已有 validation grid 不是这套当前缓存跑出来的，就不能直接复用，必须先清楚区分 provenance，再决定是否用 --force-recompute 重跑全部 300 个策略。

补充前置条件：

- [run_rank_ensemble_tra_alpha360_once.py](run_rank_ensemble_tra_alpha360_once.py) 在 `--validation-only` 模式下只要求 validation summary；只有继续执行 test-once 时，才要求 validation/test 两侧 summary 都存在
- 当前 workspace 里已经存在 [current_best_multiseed_ensemble_validation.txt](current_best_multiseed_ensemble_validation.txt)
- 但 [current_best_multiseed_ensemble_final_test.txt](current_best_multiseed_ensemble_final_test.txt) 可能不存在；如果它不存在，当前 300 空间脚本仍然可以先完成 validation-only 搜索，但无法直接完成 test-once 这一步
- 如果下游只需要 test 侧的 3 条 provenance，可以先用 [run_current_best_multiseed_ensemble.py](run_current_best_multiseed_ensemble.py) 的 `--seed-runs-test-only` 生成只含 seed_runs_test 的 summary；如果需要完整 final test summary，则可以改用 `--materialize-test-summary`
- 不要把“缺少 current_best_multiseed_ensemble_final_test.txt”误判成代码 bug；先检查是不是因为 multiseed validation 没通过 current best gate

## 7. 已确认的单策略复现结果

2026-05-16 已经用“旧 validation 三条路径 + 旧 test 三条路径”成功复现：

- strategy_trial: prac_m000_hold3_r085

正式复现入口：

- [repro_rank_ensemble_single_strategy.py](repro_rank_ensemble_single_strategy.py)

历史落盘文件：

- examples/my_strategy/tmp/repro_old_rank_ensemble_paths.json

建议新的复现输出文件：

- examples/my_strategy/tmp/repro_rank_ensemble_single_strategy.json

复现结果：

- validation_score = 266.0988237852939
- validation_ann = 0.21943669335199678
- validation_ir = 1.1263637944285012
- validation_mdd = -0.2085999506707754
- validation_IC = 0.04551111936124452
- validation_Rank_IC = 0.061059885733037514

- test_ann = 0.16268627636270552
- test_ir = 1.0552511615852955
- test_mdd = -0.23068313292472825
- test_IC = 0.04224388442035959
- test_Rank_IC = 0.06443799872736537

这说明：

- strategy_trial 没找错
- 真正决定结果的是底层 3 份 cache provenance
- 同名 rank ensemble，如果换了 validation basis，结果会完全变掉

这次 old provenance 复现还有两个可直接核对的 sanity check：

- valid_common_rows = 287568
- test_common_rows = 232836

如果这两个 common_rows 对不上，优先怀疑不是同一套 3 cache 交集，而不是先怀疑 strategy_trial。

## 8. 精确复现命令

下面这些命令都默认在仓库根目录执行，也就是 /home/blueswhen/DL/qlib。

### 8.1 复现 old rank-ensemble 的单策略结果

用途：

- 精确复现 prac_m000_hold3_r085 的 old validation + old test 指标
- 同时把实际使用的 6 条 cache_path 写入输出 json

命令：

```bash
python examples/my_strategy/repro_rank_ensemble_single_strategy.py \
  --strategy-trial prac_m000_hold3_r085 \
  --validation-cache-path /home/blueswhen/DL/qlib/examples/my_strategy/tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_strict_tra_lstm_s240_h5_tra18_residual_step1_20260506_1.pkl \
  --validation-cache-path /home/blueswhen/DL/qlib/examples/my_strategy/tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_dualseed_s2026_valid_dualseed_formal_seedxroll_b12000_20260514_1.pkl \
  --validation-cache-path /home/blueswhen/DL/qlib/examples/my_strategy/tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_dualseed_s3407_valid_dualseed_formal_seedxroll_b12000_20260514_1.pkl \
  --test-cache-path /home/blueswhen/DL/qlib/examples/my_strategy/tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r085_strict_tra_lstm_s240_h5_final_test_tra18_residual_step1_20260506_1.pkl \
  --test-cache-path /home/blueswhen/DL/qlib/examples/my_strategy/tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_dualseed_s2026_test_dualseed_formal_seedxroll_b12000_20260514_1.pkl \
  --test-cache-path /home/blueswhen/DL/qlib/examples/my_strategy/tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_dualseed_s3407_test_dualseed_formal_seedxroll_b12000_20260514_1.pkl
```

预期结果：

- validation.score 约等于 266.0988
- test.ann 约等于 0.1627
- 输出 json 中的 validation_cache_paths / test_cache_paths 必须和文档第 5.2 / 5.3 节逐条一致

### 8.2 生成 current multiseed validation summary

用途：

- 生成 [current_best_multiseed_ensemble_validation.txt](current_best_multiseed_ensemble_validation.txt)
- 为 current-basis 的 rank ensemble 提供 validation provenance

命令：

```bash
python examples/my_strategy/run_current_best_multiseed_ensemble.py --validation-only
```

输出文件：

- [current_best_multiseed_ensemble_validation.txt](current_best_multiseed_ensemble_validation.txt)

### 8.3 尝试生成 current multiseed final test summary

用途：

- 在 validation gate 通过时生成 [current_best_multiseed_ensemble_final_test.txt](current_best_multiseed_ensemble_final_test.txt)

命令：

```bash
python examples/my_strategy/run_current_best_multiseed_ensemble.py
```

注意：

- 这条命令不保证一定产出 [current_best_multiseed_ensemble_final_test.txt](current_best_multiseed_ensemble_final_test.txt)
- 只有 validation_improves_current_best=True 时，脚本才会继续跑 test 并写这个 summary
- 如果只看到 validation summary 而没有 final test summary，先读 validation summary 最后一行的 validation_improves_current_best，而不是先怀疑路径写错

如果下游只需要 test 侧 provenance，而不要求完整 final test 指标 summary，可以运行：

```bash
python examples/my_strategy/run_current_best_multiseed_ensemble.py --seed-runs-test-only
```

如果 validation gate 没过，但仍然需要强制落一份完整 final test summary，可以运行：

```bash
python examples/my_strategy/run_current_best_multiseed_ensemble.py --materialize-test-summary
```

### 8.4 用 current summaries 跑 300 空间 rank ensemble

用途：

- 先用 current multiseed provenance 跑 validation-only search
- 如果已经有 test summary，再对 validation winner 跑 1 次 test

先只跑 validation-only：

```bash
python examples/my_strategy/run_rank_ensemble_tra_alpha360_once.py \
   --validation-only \
   --validation-summary-path /home/blueswhen/DL/qlib/examples/my_strategy/current_best_multiseed_ensemble_validation.txt
```

如果 [current_best_multiseed_ensemble_final_test.txt](current_best_multiseed_ensemble_final_test.txt) 已经存在，再执行 test-once：

```bash
python examples/my_strategy/run_rank_ensemble_tra_alpha360_once.py \
   --validation-summary-path /home/blueswhen/DL/qlib/examples/my_strategy/current_best_multiseed_ensemble_validation.txt \
   --final-test-summary-path /home/blueswhen/DL/qlib/examples/my_strategy/current_best_multiseed_ensemble_final_test.txt
```

如果要强制全量重算 validation-only：

```bash
python examples/my_strategy/run_rank_ensemble_tra_alpha360_once.py \
   --validation-only \
  --force-recompute \
   --validation-summary-path /home/blueswhen/DL/qlib/examples/my_strategy/current_best_multiseed_ensemble_validation.txt
```

如果 validation/test summaries 都齐全，并且要强制全量重算再跑 test-once：

```bash
python examples/my_strategy/run_rank_ensemble_tra_alpha360_once.py \
   --force-recompute \
   --validation-summary-path /home/blueswhen/DL/qlib/examples/my_strategy/current_best_multiseed_ensemble_validation.txt \
   --final-test-summary-path /home/blueswhen/DL/qlib/examples/my_strategy/current_best_multiseed_ensemble_final_test.txt
```

失败判据：

- 如果 final test summary 不存在，这条命令会直接失败
- 如果 summary 里引用的某一条 cache_path 不存在，这条命令也会直接失败

## 9. 最小复现检查清单

每次声称“复现成功”前，至少要核对下面 8 项：

1. 运行入口脚本名
2. strategy_trial
3. validation 使用的 3 条 cache_path
4. test 使用的 3 条 cache_path
5. valid_common_rows
6. test_common_rows
7. validation 指标
8. test 指标

缺其中任何一项，都不算完整复现。

尤其不要只报：

- strategy_trial 对上了
- IC / Rank IC 对上了

因为同一个 strategy_trial 在不同 provenance 下，IC 可能局部接近，但 ann / ir / mdd 已经完全不是同一个结果。

## 10. 常见误区与故障定位

1. 不要把 [rank_ensemble_3seed_300_summary.json](rank_ensemble_3seed_300_summary.json) 当成 old provenance 的证据。
   - 它只说明某次 300 空间搜索的 winner 和 test-once 结果。
   - 它不能替代当次运行真正使用的 validation/test 6 条 cache provenance。

2. 不要从文件名表面推断 risk 已经一致。
   - old test 的 seed42 路径包含 r085。
   - old dual-seed test 路径仍然是 r07 命名。
   - 是否属于同一套复现链，必须看文档里冻结的 3+3 条 provenance，而不是只看 suffix。

3. 不要把当前 multiseed validation summary 和 old rank-ensemble validation pkl 混用。
   - 这是之前结果“离谱偏差”的直接原因之一。

4. 不要只看旧 json 落盘文件就认定结果可复现。
   - 正确做法是重新运行 helper，重新生成 json，并核对输出里的 6 条 cache_path。

5. 如果 current 300 空间脚本报 missing summary。
   - 先确认缺的是 [current_best_multiseed_ensemble_validation.txt](current_best_multiseed_ensemble_validation.txt) 还是 [current_best_multiseed_ensemble_final_test.txt](current_best_multiseed_ensemble_final_test.txt)。
   - 缺 validation summary：先跑 8.2。
   - 缺 final test summary：先检查 8.3 的 validation gate 是否通过。

## 11. 推荐操作规范

后续再做 TRA baseline 相关工作时，建议遵守下面几条：

1. 不要只记录 strategy_trial，必须同时记录 3 份 cache_path。
2. validation provenance 和 test provenance 分开记，不要默认它们来自同一批脚本输出。
3. summary txt 不要删，至少保留 recorder_id、cache_path、result_path。
4. 本地 pkl 即使删了，也要先确认对应 mlruns artifacts 还在。
5. 对 rank ensemble 的任何比较，都必须先确认它到底是 old valid basis 还是 current multiseed valid basis。

## 12. 这份文档与其它文档的关系

- [CURRENT_BEST_BASELINE.md](CURRENT_BEST_BASELINE.md) 记录的是当前全局最优基线，包含 tabular 融合。
- [历史成功方案.md](历史成功方案.md) 记录曾经成功、但已经不是 current best 的历史方案。
- 本文件只关注 TRA baseline 本身，以及围绕 TRA 展开的 single-seed / dual-seed / multiseed / rank-ensemble 这条链。
