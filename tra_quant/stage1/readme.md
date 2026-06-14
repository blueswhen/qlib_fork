# tra_quant/stage1 注意事项

这份说明面向另一台服务器上的远端复现，重点补充实际运行时容易踩坑的地方。

## 先看结论

- 如果只是看纯 stage1，也就是 TRA + Alpha360 从零训练，这个目录里的主脚本和训练数据已经基本齐了。
- 如果要复现带 cache、fusion、rank ensemble 的旧结果，不能只依赖普通 git pull，还要确认被 `.gitignore` 忽略的大文件是否真的提交到了远端，或者用其他方式同步过去。
- 当前这套目录默认假设你在 `qlib_fork/tra_quant/stage1` 下运行。

## 目录用途

- `rolling_tra_alpha360_latest.py`：TRA rolling base 训练入口。
- `tune_tra_alpha360_strict.py`：纯 stage1 主脚本，做 Alpha360 + TRA 的 validation 和 final test。
- `run_current_best_multiseed_ensemble.py`：当前 3-seed TRA 主线复现入口。
- `run_rank_ensemble_tra_alpha360_once.py`：3-seed rank ensemble 搜索入口。
- `tra_cache/`：TRA rolling cache。
- `fusion_cache/`：融合 cache。
- `mlruns/`：训练和评估 artifact。
- `xgb_cache/`：当前 multiseed 主线直接引用的 tabular cache。
- `../../training_data/cn_data_latest`：Qlib Alpha360 数据目录。

## 远端运行前提

远端服务器至少需要满足下面这些前提：

1. 在 `qlib_fork` 根目录执行过 `pip install -e .`。
2. Python 环境里有 `pyqlib`、`torch`、`pandas`、`pyyaml`。
3. 如果要跑融合或旧链路，通常还需要 `xgboost`、`lightgbm`。
4. 如果要真正训练 TRA，远端需要可用 GPU 和匹配的 CUDA 环境。
5. 建议 Python 大版本和当前环境一致，至少不要差得太远。

## 数据路径

这里的脚本已经不再依赖旧机器上的 `~/.qlib/qlib_data/cn_data_latest`。

默认数据目录是：

- `../../training_data/cn_data_latest`

包装脚本会自动设置：

- `QLIB_PROVIDER_URI=$SCRIPT_DIR/../../training_data/cn_data_latest`

如果你要改数据位置，可以在运行前手动导出自己的 `QLIB_PROVIDER_URI`。

## 只看 stage1 时，远端还差什么

如果只看纯 stage1，也就是 `tune_tra_alpha360_strict.py` 这条 Alpha360 + TRA 训练链：

- 不依赖预先存在的 `tra_cache/` 才能从零训练。
- 不依赖预先存在的 `mlruns/` 才能从零训练。
- 核心需要的是环境、GPU、以及上面的 `training_data/cn_data_latest`。

也就是说，纯 stage1 从零训练最关键的不是旧 cache，而是：

1. `qlib_fork` 已安装。
2. `training_data/cn_data_latest` 在位。
3. `torch + CUDA` 可用。
4. GPU 号和脚本配置匹配。

## stage1 的两个高频坑

### 1. GPU 槽位写死

`tune_tra_alpha360_strict.py` 当前默认：

- `GPU_SLOTS = "0,1"`

如果远端不是双卡 `0,1`，需要改这个值，或者改脚本再运行。否则很容易因为 GPU 配置不匹配直接失败。

### 2. 脚本会复用旧结果

`tune_tra_alpha360_strict.py` 不是无脑每次全量重跑。

它会：

1. 读取已有的 validation results，跳过已完成 trial。
2. 如果 final test summary 已存在，默认跳过 final test。

所以如果你要在远端做“真正从零重训”，建议先清掉这些旧文件：

- `tra_strict_validation_results.csv`
- `tra_strict_validation_best.txt`
- `tra_strict_final_test.txt`

或者运行时至少加：

- `--force-final`

## git pull 不一定等于内容齐全

这个仓库原本的 `.gitignore` 会忽略很多复现关键产物，比如：

1. `*.pkl`
2. `*.csv`
3. `mlruns/`

所以如果你只是普通 `git add . && git commit && git push`，远端 pull 后很可能缺少：

1. `tra_cache/` 里的 rolling cache
2. `fusion_cache/` 里的融合 cache
3. `mlruns/` 里的 recorder artifact
4. 一部分结果表

这件事对“纯 stage1 从零训练”影响相对小，因为 stage1 可以重新生成这些内容。

但对“旧结果直接复现”影响很大，因为很多 summary 会引用这些 artifact。

## 当前这套目录对不同目标的适用范围

### 目标一：纯 TRA stage1 从零训练

基本可以，前提是：

1. 环境装好。
2. GPU 改对。
3. 数据目录在位。
4. 必要时清理旧结果文件。

### 目标二：当前 3-seed multiseed 主线复现

可以，但更依赖：

1. `tra_cache/`
2. `fusion_cache/`
3. `xgb_cache/`
4. `mlruns/`

如果这些大文件没有真的同步到远端，pull 之后不一定能直接复现。

### 目标三：包含 XGB / fundamental 的整条历史融合链从零重训

当前不保证完全闭环。

原因是这条链路原始上游还涉及额外的 fundamental 数据来源与生成流程，这次整理的重点不是把整条 XGB 数据生产链全部本地化，而是把当前 TRA 主线和 stage1 训练链整理出来。

## 推荐命令

### 1. 远端先安装仓库

```bash
cd /path/to/qlib_fork
pip install -e .
```

### 2. 纯 stage1 从零跑

如果你要尽量避免旧结果干扰：

```bash
cd /path/to/qlib_fork/tra_quant/stage1
rm -f tra_strict_validation_results.csv tra_strict_validation_best.txt tra_strict_final_test.txt
./train_stage1_tra.sh --force-final
```

### 3. 当前 multiseed 主线

```bash
cd /path/to/qlib_fork/tra_quant/stage1
./run_current_multiseed.sh
```

### 4. rank ensemble 搜索

```bash
cd /path/to/qlib_fork/tra_quant/stage1
./run_rank_ensemble_search.sh
```

## 最后的判断标准

如果你只问一句“远端 pull 后能不能直接跑 stage1”：

- 可以接近直接跑，但前提是环境没问题、GPU 配对、数据目录在位。

如果你问“远端 pull 后能不能直接复现所有旧结果”：

- 不能直接假设可以，先检查那些被 `.gitignore` 忽略的大文件到底有没有真的同步过去。
