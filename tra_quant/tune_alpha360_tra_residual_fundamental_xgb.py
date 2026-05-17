from __future__ import annotations

import argparse
import copy
import os
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import StaticDataLoader
from qlib.model.trainer import task_train
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.task.gen import RollingGen, task_generator

from extreme_event_mask import compose_exp_suffix, suffix_path
from rolling_tra_alpha360_latest import PROVIDER_URI, build_base_task as build_tra_base_task
from rolling_xgboost_alpha158_latest import MODEL_PRESETS, WINDOWS, _prepare_handler_cache
from tune_alpha360_alpha158_dnn_strict_fusion import (
    _evaluate_signal,
    _load_alstm_strategy_cfg,
    _load_signal_from_cache,
    _load_signal_from_recorder,
    _normalize_scores,
    _parse_summary_file,
    _persist,
    _write_fused_cache,
)


BASE_DIR = Path(__file__).resolve().parent
RESIDUAL_DATA_DIR = BASE_DIR / "residual_stage2_cache"

DEFAULT_WINDOW_KEY = "w1"
ACCOUNT = 150000
TOPK = 5
BENCHMARK = "SH000300"
INSTRUMENTS = "csi300"
GPU_SLOTS = "0,1"

DEFAULT_TRA_VALIDATION_BEST_PATH = BASE_DIR / "tra_global_validation_best_tra_h5base_global_base_refcfg_20260504_1.txt"
DEFAULT_TRA_FINAL_TEST_PATH = BASE_DIR / "tra_global_final_test_tra_h5base_global_base_refcfg_20260504_1.txt"

FUNDAMENTAL_FEATURE_PATH = (
    "/home/blueswhen/DL/qlib/tushare/processed/training/"
    "csi300_daily_fundamental_features_rankpct_qlib.parquet"
)

SELECTED_TRIAL = {
    "trial_name": "residual_xgb03_techA3d_fund_rankpct_base_s240",
    "handler_class": "TechAlpha158A3DFundamental",
    "handler_module_path": "qlib.contrib.data.custom_handler",
    "handler_kwargs_extra": {
        "fundamental_feature_path": FUNDAMENTAL_FEATURE_PATH,
    },
    "model_key": "xgb_base",
    "model_kwargs_override": {
        "num_boost_round": 1000,
        "early_stopping_rounds": 50,
        "verbose_eval": 20,
    },
    "step": 240,
}

RESIDUAL_ALPHAS = [0.00, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
TRAIN_OOF_SUFFIX = "strict_tra_lstm_s240_h5_train_oof_residual"

_QLIB_INITIALIZED = False


def _ensure_qlib_initialized():
    global _QLIB_INITIALIZED
    if not _QLIB_INITIALIZED:
        qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)
        _QLIB_INITIALIZED = True


def _paths_for_window(window_key: str, result_suffix: str = "") -> dict[str, Path]:
    if window_key != "w1":
        raise ValueError("TRA residual stage2 is currently wired for w1 only")
    return {
        "validation_results": suffix_path(
            BASE_DIR / "alpha360_tra_residual_fundamental_xgb_validation_results.csv", result_suffix
        ),
        "validation_best": suffix_path(
            BASE_DIR / "alpha360_tra_residual_fundamental_xgb_validation_best.txt", result_suffix
        ),
        "final_test": suffix_path(BASE_DIR / "alpha360_tra_residual_fundamental_xgb_final_test.txt", result_suffix),
    }


def _risk_tag(risk_degree: float | str) -> str:
    return str(float(risk_degree)).replace(".", "")


def _train_oof_suffix(result_suffix: str) -> str:
    return compose_exp_suffix(TRAIN_OOF_SUFFIX, result_suffix)


def _expected_train_oof_result_path(tra_validation: dict, result_suffix: str) -> Path:
    suffix = _train_oof_suffix(result_suffix)
    return BASE_DIR / (
        f"rolling_result_tra_{tra_validation['window_key']}_{tra_validation['model_key']}_Alpha360"
        f"_step{int(tra_validation['step'])}_k{TOPK}_d{int(tra_validation['n_drop'])}_a{ACCOUNT}"
        f"_r{_risk_tag(tra_validation['risk_degree'])}_{suffix}.txt"
    )


def _normalize_gpu_slots(gpu_slots) -> list[int]:
    if gpu_slots is None:
        return []
    if isinstance(gpu_slots, (list, tuple)):
        return [int(slot) for slot in gpu_slots]
    if isinstance(gpu_slots, str):
        return [int(slot.strip()) for slot in gpu_slots.split(",") if slot.strip()]
    return [int(gpu_slots)]


def _train_oof_experiment_name(tra_validation: dict, result_suffix: str) -> str:
    return (
        f"rolling_models_tra_latest_{tra_validation['window_key']}_{tra_validation['model_key']}_Alpha360"
        f"_s{int(tra_validation['step'])}_k{TOPK}_d{int(tra_validation['n_drop'])}_a{ACCOUNT}"
        f"_r{_risk_tag(tra_validation['risk_degree'])}_{_train_oof_suffix(result_suffix)}_skipempty"
    )


def _load_successful_oof_artifacts(experiment_name: str) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    exp = R.get_exp(experiment_name=experiment_name)
    if exp is None:
        return []
    exp_dir = Path.cwd() / "mlruns" / exp.id
    if not exp_dir.exists():
        return []

    artifacts: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for recorder_dir in sorted(exp_dir.iterdir()):
        if not recorder_dir.is_dir() or recorder_dir.name == "meta.yaml":
            continue
        pred_path = recorder_dir / "artifacts" / "pred.pkl"
        label_path = recorder_dir / "artifacts" / "label.pkl"
        if not pred_path.exists() or not label_path.exists():
            continue
        pred = pd.read_pickle(pred_path).sort_index()
        label = pd.read_pickle(label_path).sort_index()
        artifacts.append((pred, label))
    return artifacts


def _combine_successful_oof_artifacts(artifacts: list[tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not artifacts:
        raise RuntimeError("no successful TRA train OOF recorder artifacts found")
    pred = pd.concat([item[0] for item in artifacts], axis=0).sort_index()
    label = pd.concat([item[1] for item in artifacts], axis=0).sort_index()
    if pred.index.has_duplicates:
        pred = pred.groupby(level=[0, 1]).mean().sort_index()
    if label.index.has_duplicates:
        label = label.groupby(level=[0, 1]).first().sort_index()
    common_index = pred.index.intersection(label.index).sort_values()
    if common_index.empty:
        raise RuntimeError("combined TRA train OOF has no aligned prediction/label index")
    return pred.loc[common_index].sort_index(), label.loc[common_index].sort_index()


def _train_tra_train_oof_skip_empty(tra_validation: dict, result_suffix: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    _ensure_qlib_initialized()
    base_task = build_tra_base_task(
        topk=TOPK,
        n_drop=int(tra_validation["n_drop"]),
        window_key=tra_validation["window_key"],
        model_key=tra_validation["model_key"],
        handler_class="Alpha360",
        handler_module_path="qlib.contrib.data.handler",
        account=ACCOUNT,
        risk_degree=float(tra_validation["risk_degree"]),
        benchmark=BENCHMARK,
        instruments=INSTRUMENTS,
        model_kwargs_override=tra_validation.get("model_kwargs_override"),
        handler_kwargs_extra=tra_validation.get("handler_kwargs_extra"),
        dataset_kwargs_extra=tra_validation.get("dataset_kwargs_extra"),
        strategy_class=tra_validation["strategy_class"],
        strategy_module_path=tra_validation["strategy_module_path"],
        strategy_kwargs_extra=tra_validation.get("kwargs_extra"),
        eval_segment="train",
        backtest_segment="train",
    )
    task_list = task_generator(
        base_task,
        RollingGen(step=int(tra_validation["step"]), rtype=RollingGen.ROLL_SD, trunc_days=2),
    )
    for task in task_list:
        task["record"] = [
            {
                "class": "SignalRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"},
            }
        ]

    experiment_name = _train_oof_experiment_name(tra_validation, result_suffix)
    slots = _normalize_gpu_slots(GPU_SLOTS) or [0]
    script_path = (BASE_DIR / "rolling_tra_alpha360_latest.py").resolve()

    with tempfile.TemporaryDirectory(prefix="tra_train_oof_skipempty_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        pending: list[tuple[int, Path]] = []
        for idx, task in enumerate(task_list):
            task_path = temp_dir_path / f"task_{idx:02d}.pkl"
            with task_path.open("wb") as fp:
                pickle.dump(task, fp)
            pending.append((idx, task_path))

        running: dict[subprocess.Popen, tuple[int, int, Path]] = {}
        available_slots = slots.copy()
        retries: dict[int, int] = {}

        while pending or running:
            while pending and available_slots:
                idx, task_path = pending.pop(0)
                gpu = available_slots.pop(0)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["QLIB_TRA_DETERMINISTIC"] = "1" if tra_validation.get("deterministic_runtime", True) else "0"
                env.setdefault("PYTHONHASHSEED", "42")
                env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                env.setdefault("OMP_NUM_THREADS", "2")
                env.setdefault("MKL_NUM_THREADS", "2")
                env.setdefault("OPENBLAS_NUM_THREADS", "2")
                env.setdefault("NUMEXPR_NUM_THREADS", "2")
                cmd = [
                    sys.executable,
                    str(script_path),
                    "train_task",
                    "--task-path",
                    str(task_path),
                    "--experiment-name",
                    experiment_name,
                    "--provider-uri",
                    PROVIDER_URI,
                    "--gpu-slot",
                    str(gpu),
                    "--worker-threads",
                    "2",
                ]
                print(f"launch TRA train_oof task {idx + 1}/{len(task_list)} on physical gpu {gpu}")
                running[subprocess.Popen(cmd, env=env)] = (idx, gpu, task_path)
                time.sleep(1)

            finished: list[subprocess.Popen] = []
            for proc, (idx, gpu, task_path) in list(running.items()):
                retcode = proc.poll()
                if retcode is None:
                    continue
                if retcode != 0:
                    retry_count = retries.get(idx, 0)
                    if retry_count < 1:
                        retries[idx] = retry_count + 1
                        print(
                            f"TRA train_oof task {idx + 1}/{len(task_list)} on gpu {gpu} failed with exit code {retcode}; "
                            f"retry {retries[idx]}/1"
                        )
                        pending.insert(0, (idx, task_path))
                    else:
                        print(f"skip empty/failed TRA train_oof task {idx + 1}/{len(task_list)} on gpu {gpu}")
                    available_slots.append(gpu)
                    finished.append(proc)
                    continue
                print(f"finished TRA train_oof task {idx + 1}/{len(task_list)} on physical gpu {gpu}")
                available_slots.append(gpu)
                finished.append(proc)
            for proc in finished:
                running.pop(proc, None)

            if running:
                time.sleep(2)

    artifacts = _load_successful_oof_artifacts(experiment_name)
    pred, label = _combine_successful_oof_artifacts(artifacts)
    return pred, label, experiment_name


def _train_tra_train_direct(tra_validation: dict, result_suffix: str) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    _ensure_qlib_initialized()
    task = build_tra_base_task(
        topk=TOPK,
        n_drop=int(tra_validation["n_drop"]),
        window_key=tra_validation["window_key"],
        model_key=tra_validation["model_key"],
        handler_class="Alpha360",
        handler_module_path="qlib.contrib.data.handler",
        account=ACCOUNT,
        risk_degree=float(tra_validation["risk_degree"]),
        benchmark=BENCHMARK,
        instruments=INSTRUMENTS,
        model_kwargs_override=tra_validation.get("model_kwargs_override"),
        handler_kwargs_extra=tra_validation.get("handler_kwargs_extra"),
        dataset_kwargs_extra=tra_validation.get("dataset_kwargs_extra"),
        strategy_class=tra_validation["strategy_class"],
        strategy_module_path=tra_validation["strategy_module_path"],
        strategy_kwargs_extra=tra_validation.get("kwargs_extra"),
        eval_segment="train",
        backtest_segment="train",
    )
    task["record"] = [
        {
            "class": "SignalRecord",
            "module_path": "qlib.workflow.record_temp",
            "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"},
        }
    ]
    experiment_name = (
        f"tra_train_direct_scorer_{tra_validation['window_key']}_{tra_validation['model_key']}"
        f"_r{_risk_tag(tra_validation['risk_degree'])}_{_train_oof_suffix(result_suffix)}"
    )
    recorder = task_train(task, experiment_name)
    pred, label = _load_signal_from_recorder(recorder.id)
    return pred, label, experiment_name, recorder.id


def _ensure_tra_train_oof(tra_validation: dict, result_suffix: str) -> dict:
    result_path = _expected_train_oof_result_path(tra_validation, result_suffix)
    if result_path.exists():
        return _parse_summary_file(result_path)

    pred, label, experiment_name, recorder_id = _train_tra_train_direct(tra_validation, result_suffix)
    cache_path = BASE_DIR / "tra_cache" / (
        f"rolling_cache_tra_{tra_validation['window_key']}_{tra_validation['model_key']}_Alpha360"
        f"_step{int(tra_validation['step'])}_k{TOPK}_d{int(tra_validation['n_drop'])}_a{ACCOUNT}"
        f"_r{_risk_tag(tra_validation['risk_degree'])}_{_train_oof_suffix(result_suffix)}_skipempty.pkl"
    )
    cache_path.parent.mkdir(exist_ok=True)
    pd.to_pickle({"pred": pred, "label": label}, cache_path)
    result_path.write_text(
        "\n".join(
            [
                f"rolling_models_exp={experiment_name}",
                "final_exp=manual_train_oof_skipempty",
                f"recorder_id={recorder_id}",
                f"window_key={tra_validation['window_key']}",
                f"model_key={tra_validation['model_key']}",
                "handler_class=Alpha360",
                "handler_module_path=qlib.contrib.data.handler",
                f"account={ACCOUNT}",
                f"risk_degree={tra_validation['risk_degree']}",
                f"provider_uri={PROVIDER_URI}",
                f"benchmark={BENCHMARK}",
                f"instruments={INSTRUMENTS}",
                f"strategy_class={tra_validation['strategy_class']}",
                f"strategy_module_path={tra_validation['strategy_module_path']}",
                f"strategy_kwargs_extra={tra_validation.get('kwargs_extra')}",
                "eval_segment=train",
                "backtest_segment=train",
                f"handler_kwargs_extra={tra_validation.get('handler_kwargs_extra')}",
                f"dataset_kwargs_extra={tra_validation.get('dataset_kwargs_extra')}",
                f"model_kwargs_override={tra_validation.get('model_kwargs_override')}",
                "train_prediction_mode=direct_train_scorer",
                "deterministic_runtime=True",
                f"cache_path={cache_path}",
                f"IC={pd.NA}",
                f"Rank IC={pd.NA}",
                f"with_cost_ann_return={pd.NA}",
                f"with_cost_ir={pd.NA}",
                f"with_cost_mdd={pd.NA}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return _parse_summary_file(result_path)


def _build_source_dataset(window_key: str) -> DatasetH:
    _ensure_qlib_initialized()
    trial = SELECTED_TRIAL
    handler_uri = _prepare_handler_cache(
        window_key=window_key,
        handler_class=trial["handler_class"],
        handler_module_path=trial["handler_module_path"],
        instruments=INSTRUMENTS,
        handler_kwargs_extra=trial["handler_kwargs_extra"],
    )
    window = WINDOWS[window_key]
    return DatasetH(
        handler=handler_uri,
        segments={
            "train": window["train"],
            "valid": window["valid"],
            "test": window["test"],
        },
    )


def _feature_frame(dataset: DatasetH, segment: str, *, data_key: str) -> pd.DataFrame:
    return dataset.prepare(segment, col_set="feature", data_key=data_key).copy().sort_index()


def _score_zscore(score: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(score, pd.Series):
        score = score.to_frame("score")
    elif "score" not in score.columns:
        score = score.iloc[:, [0]].rename(columns={score.columns[0]: "score"})
    return _normalize_scores(score, "zscore")


def _label_zscore(label: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(label, pd.DataFrame):
        label_score = label.iloc[:, 0].to_frame("score")
    else:
        label_score = label.to_frame("score")
    return _normalize_scores(label_score, "zscore")


def _add_stage1_features(features: pd.DataFrame, seq_z: pd.DataFrame) -> pd.DataFrame:
    seq_feature = seq_z.rename(columns={"score": "TRA_SCORE_Z"})
    seq_rank = seq_z.groupby(level="datetime", group_keys=False).rank(pct=True).rename(columns={"score": "TRA_SCORE_RANK_PCT"})
    return pd.concat([features, seq_feature, seq_rank], axis=1).sort_index()


def _build_segment_frame(features: pd.DataFrame, residual_target: pd.Series) -> pd.DataFrame:
    return pd.concat({"feature": features, "label": residual_target.to_frame("LABEL0")}, axis=1).sort_index()


def _build_residual_dataset(segment_frames: dict[str, pd.DataFrame], window_key: str) -> DatasetH:
    window = WINDOWS[window_key]
    combined = pd.concat([segment_frames["train"], segment_frames["valid"], segment_frames["test"]], axis=0)
    combined = combined.sort_index()
    RESIDUAL_DATA_DIR.mkdir(exist_ok=True)
    dataset_path = RESIDUAL_DATA_DIR / "tra_residual_fundamental_xgb_latest.pkl"
    pd.to_pickle(combined, dataset_path)
    handler = DataHandlerLP(
        instruments=None,
        start_time=window["train"][0],
        end_time=window["test"][1],
        data_loader=StaticDataLoader(combined),
        infer_processors=[],
        learn_processors=[],
        process_type="independent",
    )
    return DatasetH(
        handler=handler,
        segments={
            "train": window["train"],
            "valid": window["valid"],
            "test": window["test"],
        },
    )


def _xgb_model_config() -> dict:
    model_conf = copy.deepcopy(MODEL_PRESETS[SELECTED_TRIAL["model_key"]])
    model_conf["kwargs"].update(SELECTED_TRIAL["model_kwargs_override"])
    return model_conf


def _train_residual_xgb(dataset: DatasetH) -> tuple[object, str, pd.DataFrame, pd.DataFrame]:
    model = init_instance_by_config(_xgb_model_config())
    with R.start(
        experiment_name="alpha360_tra_residual_fundamental_xgb_fit",
        recorder_name=f"{SELECTED_TRIAL['trial_name']}_fit",
    ):
        recorder = R.get_recorder()
        model.fit(dataset)
        valid_pred = model.predict(dataset, segment="valid").sort_index().to_frame("score")
        valid_label = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_L).iloc[:, [0]].copy()
        valid_label.columns = ["LABEL0"]
        recorder.save_objects(**{"pred.pkl": valid_pred, "label.pkl": valid_label})
    return model, recorder.id, valid_pred, valid_label


def _predict_residual_xgb(model, dataset: DatasetH, segment: str) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    with R.start(
        experiment_name="alpha360_tra_residual_fundamental_xgb_predict",
        recorder_name=f"{SELECTED_TRIAL['trial_name']}_{segment}",
    ):
        recorder = R.get_recorder()
        pred = model.predict(dataset, segment=segment).sort_index().to_frame("score")
        label = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L).iloc[:, [0]].copy()
        label.columns = ["LABEL0"]
        recorder.save_objects(**{"pred.pkl": pred, "label.pkl": label})
    return recorder.id, pred, label


def _write_validation_best(best_row: dict, *, tra_validation: dict, validation_best_path: Path):
    lines = [
        f"window_key={tra_validation['window_key']}",
        f"train={tra_validation['train']}",
        f"valid={tra_validation['valid']}",
        f"test={tra_validation['test']}",
        "selection_scope=tra_residual_fundamental_xgb",
        f"base_model_trial={best_row['base_model_trial']}",
        f"base_strategy_trial={best_row['base_strategy_trial']}",
        f"tabular_trial={best_row['tabular_trial']}",
        f"tabular_handler_class={best_row['tabular_handler_class']}",
        f"tabular_model_key={best_row['tabular_model_key']}",
        f"fusion_mode={best_row['fusion_mode']}",
        f"residual_alpha={best_row['residual_alpha']}",
        f"validation_IC={best_row['IC']}",
        f"validation_Rank_IC={best_row['Rank IC']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"validation_score={best_row['score']}",
        f"validation_cache_path={best_row['validation_cache_path']}",
        f"validation_recorder_id={best_row['validation_recorder_id']}",
        f"tabular_validation_recorder_id={best_row['tabular_validation_recorder_id']}",
        f"tra_train_oof_cache_path={best_row['tra_train_oof_cache_path']}",
    ]
    validation_best_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_test(best_row: dict, test_metrics: dict, *, tra_validation: dict, tra_final: dict, final_test_path: Path):
    improved = float(test_metrics["with_cost_ann_return"]) > float(tra_final["test_with_cost_ann_return"])
    lines = [
        f"window_key={tra_validation['window_key']}",
        f"train={tra_validation['train']}",
        f"valid={tra_validation['valid']}",
        f"test={tra_validation['test']}",
        "selection_scope=tra_residual_fundamental_xgb",
        f"test_improves_tra_baseline={improved}",
        f"base_model_trial={best_row['base_model_trial']}",
        f"base_strategy_trial={best_row['base_strategy_trial']}",
        f"tabular_trial={best_row['tabular_trial']}",
        f"tabular_handler_class={best_row['tabular_handler_class']}",
        f"tabular_model_key={best_row['tabular_model_key']}",
        f"fusion_mode={best_row['fusion_mode']}",
        f"residual_alpha={best_row['residual_alpha']}",
        f"validation_score={best_row['score']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"test_cache_path={test_metrics['cache_path']}",
        f"test_recorder_id={test_metrics['recorder_id']}",
        f"test_IC={test_metrics['IC']}",
        f"test_Rank_IC={test_metrics['Rank IC']}",
        f"test_with_cost_ann_return={test_metrics['with_cost_ann_return']}",
        f"test_with_cost_ir={test_metrics['with_cost_ir']}",
        f"test_with_cost_mdd={test_metrics['with_cost_mdd']}",
        f"tra_baseline_test_with_cost_ann_return={tra_final['test_with_cost_ann_return']}",
        f"tra_baseline_test_with_cost_ir={tra_final['test_with_cost_ir']}",
        f"tra_baseline_test_with_cost_mdd={tra_final['test_with_cost_mdd']}",
        f"tabular_test_recorder_id={test_metrics['tabular_test_recorder_id']}",
        f"tra_train_oof_cache_path={best_row['tra_train_oof_cache_path']}",
    ]
    final_test_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(
    window_key: str = DEFAULT_WINDOW_KEY,
    result_suffix: str = "tra18_residual_fundxgb_20260505_1",
    tra_validation_best_path: str | None = None,
    tra_final_test_path: str | None = None,
):
    paths = _paths_for_window(window_key, result_suffix)
    tra_validation = _parse_summary_file(Path(tra_validation_best_path or DEFAULT_TRA_VALIDATION_BEST_PATH).resolve())
    tra_final = _parse_summary_file(Path(tra_final_test_path or DEFAULT_TRA_FINAL_TEST_PATH).resolve())
    tra_train = _ensure_tra_train_oof(tra_validation, result_suffix)

    strategy_cfg_valid = _load_alstm_strategy_cfg(tra_validation, eval_segment="valid")
    strategy_cfg_test = _load_alstm_strategy_cfg(tra_validation, eval_segment="test")

    source_dataset = _build_source_dataset(window_key)
    train_features = _feature_frame(source_dataset, "train", data_key=DataHandlerLP.DK_L)
    valid_features = _feature_frame(source_dataset, "valid", data_key=DataHandlerLP.DK_L)
    test_features = _feature_frame(source_dataset, "test", data_key=DataHandlerLP.DK_I)

    seq_train_pred, seq_train_label = _load_signal_from_cache(tra_train["cache_path"])
    seq_valid_pred, seq_valid_label = _load_signal_from_cache(tra_validation["validation_cache_path"])
    seq_test_pred, seq_test_label = _load_signal_from_cache(tra_final["test_cache_path"])

    seq_train_z = _score_zscore(seq_train_pred)
    seq_valid_z = _score_zscore(seq_valid_pred)
    seq_test_z = _score_zscore(seq_test_pred)
    label_train_z = _label_zscore(seq_train_label)
    label_valid_z = _label_zscore(seq_valid_label)
    label_test_z = _label_zscore(seq_test_label)

    train_index = train_features.index.intersection(seq_train_z.index).intersection(label_train_z.index).sort_values()
    valid_index = valid_features.index.intersection(seq_valid_z.index).intersection(label_valid_z.index).sort_values()
    test_index = test_features.index.intersection(seq_test_z.index).intersection(label_test_z.index).sort_values()
    if train_index.empty or valid_index.empty or test_index.empty:
        raise RuntimeError(
            f"empty residual alignment: train={len(train_index)} valid={len(valid_index)} test={len(test_index)}"
        )

    train_x = _add_stage1_features(train_features.loc[train_index], seq_train_z.loc[train_index])
    valid_x = _add_stage1_features(valid_features.loc[valid_index], seq_valid_z.loc[valid_index])
    test_x = _add_stage1_features(test_features.loc[test_index], seq_test_z.loc[test_index])

    residual_dataset = _build_residual_dataset(
        {
            "train": _build_segment_frame(
                train_x,
                label_train_z.loc[train_index, "score"].sub(seq_train_z.loc[train_index, "score"], fill_value=0.0),
            ),
            "valid": _build_segment_frame(
                valid_x,
                label_valid_z.loc[valid_index, "score"].sub(seq_valid_z.loc[valid_index, "score"], fill_value=0.0),
            ),
            "test": _build_segment_frame(
                test_x,
                label_test_z.loc[test_index, "score"].sub(seq_test_z.loc[test_index, "score"], fill_value=0.0),
            ),
        },
        window_key,
    )

    rows: list[dict] = [
        {
            "trial_name": "tra_baseline_only",
            "base_model_trial": tra_validation["model_trial"],
            "base_strategy_trial": tra_validation["strategy_trial"],
            "tabular_trial": "none",
            "tabular_handler_class": "none",
            "tabular_model_key": "none",
            "fusion_mode": "baseline",
            "residual_alpha": 0.0,
            "status": "reference",
            "IC": float(tra_validation["validation_IC"]),
            "Rank IC": float(tra_validation["validation_Rank_IC"]),
            "with_cost_ann_return": float(tra_validation["validation_with_cost_ann_return"]),
            "with_cost_ir": float(tra_validation["validation_with_cost_ir"]),
            "with_cost_mdd": float(tra_validation["validation_with_cost_mdd"]),
            "score": float(tra_validation["validation_score"]),
            "validation_cache_path": str(tra_validation["validation_cache_path"]),
            "validation_recorder_id": "baseline",
            "tabular_validation_recorder_id": "none",
            "tra_train_oof_cache_path": str(tra_train["cache_path"]),
        }
    ]
    _persist(rows, paths["validation_results"])

    model, fit_recorder_id, residual_valid_pred, _ = _train_residual_xgb(residual_dataset)
    residual_valid_z = _score_zscore(residual_valid_pred)
    valid_label_original = seq_valid_label.loc[valid_index].sort_index()

    for residual_alpha in RESIDUAL_ALPHAS:
        if residual_alpha == 0.0:
            continue
        trial_name = (
            f"tra_residual_fundxgb_alpha{str(residual_alpha).replace('.', '')}"
            f"_{result_suffix}" if result_suffix else f"tra_residual_fundxgb_alpha{str(residual_alpha).replace('.', '')}"
        )
        row = {
            "trial_name": trial_name,
            "base_model_trial": tra_validation["model_trial"],
            "base_strategy_trial": tra_validation["strategy_trial"],
            "tabular_trial": SELECTED_TRIAL["trial_name"],
            "tabular_handler_class": SELECTED_TRIAL["handler_class"],
            "tabular_model_key": SELECTED_TRIAL["model_key"],
            "fusion_mode": "seq_zscore_plus_residual_zscore",
            "residual_alpha": residual_alpha,
            "status": "running",
        }
        try:
            common_index = seq_valid_z.index.intersection(residual_valid_z.index).intersection(valid_label_original.index)
            fused_valid = seq_valid_z.loc[common_index].add(
                residual_valid_z.loc[common_index].mul(float(residual_alpha)), fill_value=0.0
            )
            validation_cache_path = _write_fused_cache(trial_name, fused_valid, valid_label_original.loc[common_index])
            metrics = _evaluate_signal(
                recorder_name=trial_name,
                pred=fused_valid,
                label=valid_label_original.loc[common_index],
                strategy_cfg=strategy_cfg_valid,
                experiment_name="alpha360_tra_residual_fundamental_xgb_validation_eval",
                fast_mode=True,
            )
            row.update(
                {
                    "status": "success",
                    "IC": metrics["IC"],
                    "Rank IC": metrics["Rank IC"],
                    "with_cost_ann_return": metrics["with_cost_ann_return"],
                    "with_cost_ir": metrics["with_cost_ir"],
                    "with_cost_mdd": metrics["with_cost_mdd"],
                    "score": metrics["score"],
                    "validation_cache_path": str(validation_cache_path),
                    "validation_recorder_id": metrics["recorder_id"],
                    "tabular_validation_recorder_id": fit_recorder_id,
                    "tra_train_oof_cache_path": str(tra_train["cache_path"]),
                }
            )
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)[:500]
        rows = [existing for existing in rows if existing.get("trial_name") != row["trial_name"]] + [row]
        _persist(rows, paths["validation_results"])

    success_rows = [row for row in rows if row["status"] in {"success", "reference"}]
    best_row = max(success_rows, key=lambda item: float(item["score"]))
    _write_validation_best(best_row, tra_validation=tra_validation, validation_best_path=paths["validation_best"])

    if best_row["status"] == "reference":
        test_metrics = {
            "cache_path": tra_final["test_cache_path"],
            "recorder_id": tra_final["test_recorder_id"],
            "IC": tra_final["test_IC"],
            "Rank IC": tra_final["test_Rank_IC"],
            "with_cost_ann_return": tra_final["test_with_cost_ann_return"],
            "with_cost_ir": tra_final["test_with_cost_ir"],
            "with_cost_mdd": tra_final["test_with_cost_mdd"],
            "tabular_test_recorder_id": "none",
        }
        _write_final_test(
            best_row,
            test_metrics,
            tra_validation=tra_validation,
            tra_final=tra_final,
            final_test_path=paths["final_test"],
        )
        return

    test_recorder_id, residual_test_pred, _ = _predict_residual_xgb(model, residual_dataset, "test")
    residual_test_z = _score_zscore(residual_test_pred)
    test_label_original = seq_test_label.loc[test_index].sort_index()
    common_test_index = seq_test_z.index.intersection(residual_test_z.index).intersection(test_label_original.index)
    fused_test = seq_test_z.loc[common_test_index].add(
        residual_test_z.loc[common_test_index].mul(float(best_row["residual_alpha"])), fill_value=0.0
    )
    test_cache_path = _write_fused_cache(f"{best_row['trial_name']}_final_test", fused_test, test_label_original.loc[common_test_index])
    test_metrics = _evaluate_signal(
        recorder_name=f"{best_row['trial_name']}_final_test",
        pred=fused_test,
        label=test_label_original.loc[common_test_index],
        strategy_cfg=strategy_cfg_test,
        experiment_name="alpha360_tra_residual_fundamental_xgb_final_test_eval",
        fast_mode=False,
    )
    test_metrics.update({"cache_path": str(test_cache_path), "tabular_test_recorder_id": test_recorder_id})
    _write_final_test(
        best_row,
        test_metrics,
        tra_validation=tra_validation,
        tra_final=tra_final,
        final_test_path=paths["final_test"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Residual fundamental XGBoost fusion on top of fixed TRA predictions.")
    parser.add_argument("--window-key", default=DEFAULT_WINDOW_KEY, choices=["w1"])
    parser.add_argument("--result-suffix", default="tra18_residual_fundxgb_20260505_1")
    parser.add_argument("--tra-validation-best-path", default=None)
    parser.add_argument("--tra-final-test-path", default=None)
    args = parser.parse_args()
    main(
        window_key=args.window_key,
        result_suffix=args.result_suffix,
        tra_validation_best_path=args.tra_validation_best_path,
        tra_final_test_path=args.tra_final_test_path,
    )