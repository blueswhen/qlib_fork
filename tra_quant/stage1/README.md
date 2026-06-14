# tra_quant/stage1

这个目录是从 qlib 主仓库整理出来的 TRA/stage1 独立复现包，目标是在 qlib_fork 下单独复现当前 TRA-only 主线训练与结果。

## 目录说明

- `rolling_tra_alpha360_latest.py`：TRA rolling base 训练入口。
- `tune_tra_alpha360_strict.py`：stage1 严格 TRA 训练与筛选。
- `tune_tra_alpha360_global.py`：TRA 全局策略搜索主脚本。
- `run_current_best_multiseed_ensemble.py`：当前 3-seed TRA 主线复现入口。
- `run_rank_ensemble_tra_alpha360_once.py`：3-seed rank ensemble 搜索入口。
- `repro_rank_ensemble_single_strategy.py`：单策略 rank ensemble 复现实验。
- `tra_cache/`：完整 TRA rolling cache。
- `fusion_cache/`：融合结果 cache。
- `mlruns/`：本地训练与评估 artifact。
- `xgb_cache/`：当前 multiseed 主线直接引用的 tabular cache。
- `docs/`：相关说明文档。

## 数据位置

训练数据已经复制到上一级目录：

- `../../training_data/cn_data_latest`

当前脚本默认会把 `QLIB_PROVIDER_URI` 指向这个相对目录，不再依赖旧机器上的 `~/.qlib/qlib_data/cn_data_latest`。

## 运行前提

建议先在 qlib_fork 根目录安装当前仓库：

```bash
cd /path/to/qlib_fork
pip install -e .
```

如果你使用的是新的虚拟环境，确认 `pyqlib`, `torch`, `pandas`, `pyyaml`, `lightgbm`, `xgboost` 等依赖已经可用。

## 推荐运行方式

所有命令都建议从这个目录执行，或者直接调用下面 3 个包装脚本。

## 当前权威 TRA-only baseline

目前 TRA-only/stage1 的权威 baseline 已更新为：

- baseline：old provenance 下的 3-seed TRA rank ensemble
- 搜索空间：`tra-best-72`
- winner：`prac_m000_hold3_r085`
- test_ann：`0.16268627636270552`
- test_ir：`1.0552511615852953`
- test_mdd：`-0.23068313292472836`

注意：`tra_best_72_rerun_20260516_1` 只是单 seed 72 空间 rerun，不再作为权威 TRA best。
完整说明见 `TRA_best_baseline.md` 和 `docs/TRA_best_baseline.md`。

### 1. 重新训练 stage1 严格 TRA

```bash
cd /path/to/qlib_fork/tra_quant/stage1
./train_stage1_tra.sh
```

这一步会重新生成：

- `tra_strict_validation_best.txt`
- `tra_strict_final_test.txt`
- 新的本地 `mlruns/` artifact

### 2. 重新跑当前 3-seed TRA 主线

```bash
cd /path/to/qlib_fork/tra_quant/stage1
./run_current_multiseed.sh
```

这一步会基于本地数据与本地 cache 生成：

- `current_best_multiseed_ensemble_validation.txt`
- `current_best_multiseed_ensemble_final_test.txt`
- 对应新的 `fusion_cache/` 结果

### 3. 重新做 rank ensemble 搜索

```bash
cd /path/to/qlib_fork/tra_quant/stage1
./run_rank_ensemble_search.sh
```

如果要复现权威 72 空间 old-provenance baseline，使用：

```bash
python run_rank_ensemble_tra_alpha360_once.py \
  --strategy-profile tra-best-72 \
  --validation-summary-path tmp/rank_ensemble_old6_validation_summary_20260517.txt \
  --final-test-summary-path tmp/rank_ensemble_old6_final_test_summary_20260517.txt \
  --output-prefix tmp/rank_ensemble_old6_tra_best_72
```

## 关键说明

- 复制过来的 summary/json 里原本写死了旧机器绝对路径；tra_quant 里的脚本已经做了本地路径映射，会自动改到当前目录结构。
- 当前 multiseed 主线会读取打包进来的 xgb valid/test cache 作为 tabular 分支输入，不需要在另一台机器上重新训练 xgb。
- 如果你只关心纯 TRA 训练与评估，重点运行 `train_stage1_tra.sh` 和 `run_current_multiseed.sh` 即可。
- 如果必须完全复现旧 artifact 级 provenance，请保留 `tra_cache/`, `fusion_cache/`, `mlruns/` 的目录结构不变。

## 已打包文档

- `docs/TRA_best_baseline.md`
- `docs/CURRENT_BEST_BASELINE.md`
- `docs/NEXT_PATHS_NO_MODEL_SWITCH.md`
