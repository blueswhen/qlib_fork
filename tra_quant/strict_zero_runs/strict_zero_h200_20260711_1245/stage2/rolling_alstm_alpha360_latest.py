from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import qlib
import yaml
from qlib.constant import REG_CN
from qlib.model.ens.ensemble import RollingEnsemble
from qlib.model.trainer import TrainerR, task_train
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord
from qlib.workflow.task.collect import RecorderCollector
from qlib.workflow.task.gen import RollingGen, task_generator


BASE_DIR = Path(__file__).resolve().parent
PROVIDER_URI = str((BASE_DIR.parent.parent / "training_data" / "cn_data_latest").resolve())
CACHE_DIR = BASE_DIR / "alstm_cache"
DETERMINISTIC_SEED = 42


def _deterministic_runtime_enabled(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit
    return os.environ.get("QLIB_ALSTM_DETERMINISTIC", "1").lower() not in {"0", "false", "no"}


def _configure_deterministic_runtime(seed: int = DETERMINISTIC_SEED):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False

WINDOWS = {
    "w1": {
        "start_time": "2012-01-01",
        "end_time": "2026-04-01",
        "fit_start_time": "2012-01-01",
        "fit_end_time": "2019-12-31",
        "train": ["2012-01-01", "2019-12-31"],
        "valid": ["2020-01-01", "2022-12-31"],
        "test": ["2023-01-01", "2026-04-01"],
    },
    "w2": {
        "start_time": "2015-01-01",
        "end_time": "2026-04-01",
        "fit_start_time": "2015-01-01",
        "fit_end_time": "2019-12-31",
        "train": ["2015-01-01", "2019-12-31"],
        "valid": ["2020-01-01", "2022-12-31"],
        "test": ["2023-01-01", "2026-04-01"],
    },
    "w3": {
        "start_time": "2018-01-01",
        "end_time": "2026-04-01",
        "fit_start_time": "2018-01-01",
        "fit_end_time": "2020-12-31",
        "train": ["2018-01-01", "2020-12-31"],
        "valid": ["2021-01-01", "2022-12-31"],
        "test": ["2023-01-01", "2026-04-01"],
    },
}

MODEL_PRESETS = {
    "smoke": {
        "class": "ALSTM",
        "module_path": "qlib.contrib.model.pytorch_alstm",
        "kwargs": {
            "d_feat": 6,
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "n_epochs": 1,
            "lr": 1e-3,
            "early_stop": 1,
            "batch_size": 8192,
            "metric": "loss",
            "loss": "mse",
            "optimizer": "adam",
            "GPU": 0,
            "seed": 42,
            "rnn_type": "GRU",
        },
    },
    "gru_base": {
        "class": "ALSTM",
        "module_path": "qlib.contrib.model.pytorch_alstm",
        "kwargs": {
            "d_feat": 6,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.0,
            "n_epochs": 120,
            "lr": 1e-3,
            "early_stop": 20,
            "batch_size": 2048,
            "metric": "loss",
            "loss": "mse",
            "optimizer": "adam",
            "GPU": 0,
            "seed": 42,
            "rnn_type": "GRU",
        },
    },
    "gru_wide": {
        "class": "ALSTM",
        "module_path": "qlib.contrib.model.pytorch_alstm",
        "kwargs": {
            "d_feat": 6,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.1,
            "n_epochs": 100,
            "lr": 5e-4,
            "early_stop": 16,
            "batch_size": 4096,
            "metric": "loss",
            "loss": "mse",
            "optimizer": "adam",
            "GPU": 0,
            "seed": 42,
            "rnn_type": "GRU",
        },
    },
    "gru_deep": {
        "class": "ALSTM",
        "module_path": "qlib.contrib.model.pytorch_alstm",
        "kwargs": {
            "d_feat": 6,
            "hidden_size": 128,
            "num_layers": 3,
            "dropout": 0.2,
            "n_epochs": 120,
            "lr": 3e-4,
            "early_stop": 18,
            "batch_size": 4096,
            "metric": "loss",
            "loss": "mse",
            "optimizer": "adam",
            "GPU": 0,
            "seed": 42,
            "rnn_type": "GRU",
        },
    },
    "lstm_wide": {
        "class": "ALSTM",
        "module_path": "qlib.contrib.model.pytorch_alstm",
        "kwargs": {
            "d_feat": 6,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.1,
            "n_epochs": 100,
            "lr": 5e-4,
            "early_stop": 16,
            "batch_size": 4096,
            "metric": "loss",
            "loss": "mse",
            "optimizer": "adam",
            "GPU": 0,
            "seed": 42,
            "rnn_type": "LSTM",
        },
    },
}

DEFAULT_HANDLER_KWARGS_EXTRA = {
    "infer_processors": [
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
    ],
    "learn_processors": [
        {"class": "DropnaLabel"},
        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
    ],
    "label": ["Ref($close, -4) / Ref($close, -1) - 1"],
}


def _read_metric_from_store(recorder_id: str, metric_name: str):
    mlruns_dir = Path.cwd() / "mlruns"
    for metric_file in mlruns_dir.glob(f"*/{recorder_id}/metrics/{metric_name}"):
        lines = metric_file.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue
        parts = lines[-1].split()
        if len(parts) >= 2:
            return float(parts[1])
    return None


def _metric_value(metrics: dict, recorder_id: str, metric_name: str):
    value = metrics.get(metric_name)
    if value is not None:
        return value
    return _read_metric_from_store(recorder_id, metric_name)


def _stable_signature(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _read_existing_run_result(result_path: Path, cache_path: Path) -> dict | None:
    if not result_path.exists() or not cache_path.exists():
        return None

    raw = {}
    for line in result_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        raw[key] = value

    def _as_float(key: str):
        value = raw.get(key)
        if value in (None, "None", ""):
            return None
        return float(value)

    return {
        "result_path": str(result_path),
        "cache_path": str(cache_path),
        "recorder_id": raw.get("recorder_id"),
        "IC": _as_float("IC"),
        "Rank IC": _as_float("Rank IC"),
        "with_cost_ann_return": _as_float("with_cost_ann_return"),
        "with_cost_ir": _as_float("with_cost_ir"),
        "with_cost_mdd": _as_float("with_cost_mdd"),
        "eval_segment": raw.get("eval_segment"),
        "backtest_segment": raw.get("backtest_segment"),
    }


def _deep_merge(base: dict, extra: dict | None) -> dict:
    if extra is None:
        return dict(base)
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_gpu_value(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(part) for part in value]
    if isinstance(value, str):
        stripped = value.strip()
        if "," in stripped:
            return [int(part.strip()) for part in stripped.split(",") if part.strip()]
        if stripped.lstrip("-").isdigit():
            return int(stripped)
        return stripped
    return int(value)


def _normalize_model_kwargs(model_kwargs: dict) -> dict:
    normalized = dict(model_kwargs)
    float_keys = {"lr", "dropout"}
    int_keys = {"d_feat", "hidden_size", "num_layers", "n_epochs", "early_stop", "batch_size", "seed"}

    for key in float_keys:
        if isinstance(normalized.get(key), str):
            normalized[key] = float(normalized[key])
    for key in int_keys:
        if isinstance(normalized.get(key), str):
            normalized[key] = int(normalized[key])
    if "GPU" in normalized:
        normalized["GPU"] = _normalize_gpu_value(normalized["GPU"])
    return normalized


def _normalize_gpu_slots(gpu_slots):
    if gpu_slots is None:
        return []
    if isinstance(gpu_slots, (list, tuple)):
        return [int(slot) for slot in gpu_slots]
    if isinstance(gpu_slots, str):
        return [int(slot.strip()) for slot in gpu_slots.split(",") if slot.strip()]
    return [int(gpu_slots)]


def _format_date(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _shrink_task_handler_window(task: dict, lookback_days: int = 200, lookahead_days: int = 30) -> dict:
    task = copy.deepcopy(task)

    dataset_kwargs = task["dataset"]["kwargs"]
    handler_kwargs = dataset_kwargs["handler"]["kwargs"]
    segments = dataset_kwargs["segments"]

    seg_starts = []
    seg_ends = []
    for seg in segments.values():
        if isinstance(seg, (list, tuple)) and len(seg) == 2:
            seg_starts.append(pd.Timestamp(seg[0]))
            seg_ends.append(pd.Timestamp(seg[1]))

    if not seg_starts or not seg_ends:
        return task

    train_seg = segments.get("train")
    if isinstance(train_seg, (list, tuple)) and len(train_seg) == 2:
        fit_start = pd.Timestamp(train_seg[0])
        fit_end = pd.Timestamp(train_seg[1])
    else:
        fit_start = min(seg_starts)
        fit_end = max(seg_ends)

    handler_start = min(seg_starts) - pd.Timedelta(days=lookback_days)
    handler_end = max(seg_ends) + pd.Timedelta(days=lookahead_days)

    original_start = pd.Timestamp(handler_kwargs["start_time"])
    original_end = pd.Timestamp(handler_kwargs["end_time"])
    handler_start = max(handler_start, original_start)
    handler_end = min(handler_end, original_end)

    handler_kwargs["start_time"] = _format_date(handler_start)
    handler_kwargs["end_time"] = _format_date(handler_end)
    handler_kwargs["fit_start_time"] = _format_date(fit_start)
    handler_kwargs["fit_end_time"] = _format_date(fit_end)
    return task


def _resolve_segment(window: dict, segment_key: str):
    if segment_key not in window:
        raise ValueError(f"unknown segment `{segment_key}`; available keys: {sorted(window.keys())}")
    return list(window[segment_key])


def _train_tasks(task_list: list[dict], experiment_name: str, provider_uri: str, gpu_slots, deterministic_runtime: bool):
    slots = _normalize_gpu_slots(gpu_slots)
    if not slots:
        slots = [0]
    R.get_exp(experiment_name=experiment_name)

    script_path = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="alstm_tasks_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        pending = []
        retries = {}
        for idx, task in enumerate(task_list):
            task = _shrink_task_handler_window(task)
            task_path = temp_dir_path / f"task_{idx:02d}.pkl"
            with task_path.open("wb") as fp:
                pickle.dump(task, fp)
            pending.append((idx, task_path))

        running = {}
        available_slots = slots.copy()

        while pending or running:
            while pending and available_slots:
                idx, task_path = pending.pop(0)
                gpu = available_slots.pop(0)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["QLIB_ALSTM_DETERMINISTIC"] = "1" if deterministic_runtime else "0"
                if deterministic_runtime:
                    env.setdefault("PYTHONHASHSEED", str(DETERMINISTIC_SEED))
                    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                env.setdefault("OMP_NUM_THREADS", "1")
                env.setdefault("MKL_NUM_THREADS", "1")
                env.setdefault("OPENBLAS_NUM_THREADS", "1")
                env.setdefault("NUMEXPR_NUM_THREADS", "1")
                cmd = [
                    sys.executable,
                    str(script_path),
                    "train_task",
                    "--task-path",
                    str(task_path),
                    "--experiment-name",
                    experiment_name,
                    "--provider-uri",
                    provider_uri,
                    "--gpu-slot",
                    str(gpu),
                ]
                print(f"launch rolling task {idx + 1}/{len(task_list)} on physical gpu {gpu}")
                proc = subprocess.Popen(cmd, env=env)
                running[proc] = (idx, gpu, task_path)
                # Stagger process startup slightly to reduce transient CUDA init contention.
                time.sleep(2)

            finished = []
            for proc, (idx, gpu, task_path) in running.items():
                retcode = proc.poll()
                if retcode is None:
                    continue
                if retcode != 0:
                    retry_count = retries.get(idx, 0)
                    if retry_count < 2:
                        retries[idx] = retry_count + 1
                        print(
                            f"rolling task {idx + 1}/{len(task_list)} on gpu {gpu} failed with exit code {retcode}; "
                            f"retry {retries[idx]}/2"
                        )
                        pending.insert(0, (idx, task_path))
                        available_slots.append(gpu)
                        finished.append(proc)
                        continue
                    raise RuntimeError(f"rolling task {idx + 1} on gpu {gpu} failed with exit code {retcode}")
                print(f"finished rolling task {idx + 1}/{len(task_list)} on physical gpu {gpu}")
                available_slots.append(gpu)
                finished.append(proc)
            for proc in finished:
                running.pop(proc, None)

            if running:
                time.sleep(2)


def build_base_task(
    topk: int = 5,
    n_drop: int = 1,
    window_key: str = "w1",
    model_key: str = "gru_wide",
    handler_class: str = "Alpha360",
    handler_module_path: str = "qlib.contrib.data.handler",
    account: int = 150000,
    risk_degree: float = 0.70,
    benchmark: str = "SH000300",
    instruments: str = "csi300",
    model_kwargs_override: dict | None = None,
    handler_kwargs_extra: dict | None = None,
    strategy_class: str = "TopkDropoutStrategy",
    strategy_module_path: str = "qlib.contrib.strategy",
    strategy_kwargs_extra: dict | None = None,
    eval_segment: str = "test",
    backtest_segment: str | None = None,
) -> dict:
    window = WINDOWS[window_key]
    model_conf = MODEL_PRESETS[model_key]
    model_kwargs = _normalize_model_kwargs(_deep_merge(model_conf["kwargs"], model_kwargs_override))
    handler_kwargs = _deep_merge(DEFAULT_HANDLER_KWARGS_EXTRA, handler_kwargs_extra)
    eval_range = _resolve_segment(window, eval_segment)
    backtest_range = _resolve_segment(window, backtest_segment or eval_segment)

    strategy_kwargs = {
        "signal": "<PRED>",
        "topk": topk,
        "n_drop": n_drop,
        "risk_degree": risk_degree,
    }
    if strategy_kwargs_extra:
        strategy_kwargs.update(strategy_kwargs_extra)

    data_handler_config = {
        "start_time": window["start_time"],
        "end_time": window["end_time"],
        "fit_start_time": window["fit_start_time"],
        "fit_end_time": window["fit_end_time"],
        "instruments": instruments,
    }
    data_handler_config.update(handler_kwargs)

    port_analysis_config = {
        "strategy": {
            "class": strategy_class,
            "module_path": strategy_module_path,
            "kwargs": strategy_kwargs,
        },
        "backtest": {
            "start_time": backtest_range[0],
            "end_time": backtest_range[1],
            "account": account,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }
    return {
        "model": {
            "class": model_conf["class"],
            "module_path": model_conf["module_path"],
            "kwargs": model_kwargs,
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": handler_class,
                    "module_path": handler_module_path,
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": window["train"],
                    "valid": window["valid"],
                    "test": eval_range,
                },
            },
        },
        "record": [
            {
                "class": "SignalRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"},
            },
            {
                "class": "SigAnaRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {"ana_long_short": False, "ann_scaler": 252},
            },
            {
                "class": "PortAnaRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {"config": port_analysis_config},
            },
        ],
    }


def run(
    step: int = 120,
    topk: int = 5,
    n_drop: int = 1,
    window_key: str = "w1",
    model_key: str = "gru_wide",
    exp_suffix: str = "",
    handler_class: str = "Alpha360",
    handler_module_path: str = "qlib.contrib.data.handler",
    account: int = 150000,
    risk_degree: float = 0.70,
    provider_uri: str = PROVIDER_URI,
    benchmark: str = "SH000300",
    instruments: str = "csi300",
    model_kwargs_override: dict | None = None,
    handler_kwargs_extra: dict | None = None,
    strategy_class: str = "TopkDropoutStrategy",
    strategy_module_path: str = "qlib.contrib.strategy",
    strategy_kwargs_extra: dict | None = None,
    gpu_slots="0,1",
    eval_segment: str = "test",
    backtest_segment: str | None = None,
    deterministic_runtime: bool | None = None,
):
    if _deterministic_runtime_enabled(deterministic_runtime):
        _configure_deterministic_runtime()

    qlib.init(provider_uri=provider_uri, region=REG_CN)

    base_task = build_base_task(
        topk=topk,
        n_drop=n_drop,
        window_key=window_key,
        model_key=model_key,
        handler_class=handler_class,
        handler_module_path=handler_module_path,
        account=account,
        risk_degree=risk_degree,
        benchmark=benchmark,
        instruments=instruments,
        model_kwargs_override=model_kwargs_override,
        handler_kwargs_extra=handler_kwargs_extra,
        strategy_class=strategy_class,
        strategy_module_path=strategy_module_path,
        strategy_kwargs_extra=strategy_kwargs_extra,
        eval_segment=eval_segment,
        backtest_segment=backtest_segment,
    )

    signature = _stable_signature(
        {
            "step": step,
            "topk": topk,
            "n_drop": n_drop,
            "window_key": window_key,
            "model_key": model_key,
            "handler_class": handler_class,
            "handler_module_path": handler_module_path,
            "account": account,
            "risk_degree": risk_degree,
            "provider_uri": provider_uri,
            "benchmark": benchmark,
            "instruments": instruments,
            "strategy_class": strategy_class,
            "strategy_module_path": strategy_module_path,
            "eval_segment": eval_segment,
            "backtest_segment": backtest_segment,
            "deterministic_runtime": _deterministic_runtime_enabled(deterministic_runtime),
            "base_task": base_task,
        }
    )
    suffix_parts = [part for part in [exp_suffix, f"sig{signature}"] if part]
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
    risk_tag = str(risk_degree).replace(".", "")

    result_path = BASE_DIR / (
        f"rolling_result_alstm_{window_key}_{model_key}_{handler_class}_step{step}"
        f"_k{topk}_d{n_drop}_a{account}_r{risk_tag}{suffix}.txt"
    )
    cache_path = CACHE_DIR / (
        f"rolling_cache_alstm_{window_key}_{model_key}_{handler_class}_step{step}"
        f"_k{topk}_d{n_drop}_a{account}_r{risk_tag}{suffix}.pkl"
    )

    existing_result = _read_existing_run_result(result_path, cache_path)
    if existing_result is not None:
        print(f"reuse exact-match rolling result: {result_path}")
        return existing_result

    rolling_models_exp = (
        f"rolling_models_alstm_latest_{window_key}_{model_key}_{handler_class}_s{step}_k{topk}_d{n_drop}_a{account}_r{risk_tag}{suffix}"
    )
    final_exp = (
        f"rolling_eval_alstm_latest_{window_key}_{model_key}_{handler_class}_s{step}_k{topk}_d{n_drop}_a{account}_r{risk_tag}{suffix}"
    )

    task_list = task_generator(
        base_task,
        RollingGen(step=step, rtype=RollingGen.ROLL_SD, trunc_days=2),
    )
    for task in task_list:
        task["record"] = [
            {
                "class": "SignalRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"},
            }
        ]

    print(f"rolling tasks: {len(task_list)}")
    print(f"rolling model experiment: {rolling_models_exp}")
    print(f"final evaluation experiment: {final_exp}")
    print(f"gpu slots: {_normalize_gpu_slots(gpu_slots)}")
    print(f"deterministic runtime: {_deterministic_runtime_enabled(deterministic_runtime)}")

    _train_tasks(
        task_list,
        rolling_models_exp,
        provider_uri,
        gpu_slots,
        _deterministic_runtime_enabled(deterministic_runtime),
    )

    collector = RecorderCollector(
        experiment=rolling_models_exp,
        artifacts_key=["pred", "label"],
        process_list=[RollingEnsemble()],
        artifacts_path={"pred": "pred.pkl", "label": "label.pkl"},
    )
    res = collector()

    with R.start(experiment_name=final_exp, recorder_name=f"rolling_step_{step}"):
        rec = R.get_recorder()
        rec.save_objects(**{"pred.pkl": res["pred"], "label.pkl": res["label"]})

        for record in build_base_task(
            topk=topk,
            n_drop=n_drop,
            window_key=window_key,
            model_key=model_key,
            handler_class=handler_class,
            handler_module_path=handler_module_path,
            account=account,
            risk_degree=risk_degree,
            benchmark=benchmark,
            instruments=instruments,
            model_kwargs_override=model_kwargs_override,
            handler_kwargs_extra=handler_kwargs_extra,
            strategy_class=strategy_class,
            strategy_module_path=strategy_module_path,
            strategy_kwargs_extra=strategy_kwargs_extra,
            eval_segment=eval_segment,
            backtest_segment=backtest_segment,
        )["record"]:
            if record["class"] == "SignalRecord":
                continue
            if record["class"] == "SigAnaRecord":
                SigAnaRecord(recorder=rec, **record["kwargs"]).generate()
            elif record["class"] == "PortAnaRecord":
                PortAnaRecord(recorder=rec, **record["kwargs"]).generate()

        metrics = rec.list_metrics()

    CACHE_DIR.mkdir(exist_ok=True)
    pd.to_pickle({"pred": res["pred"], "label": res["label"]}, cache_path)

    result_path.write_text(
        "\n".join(
            [
                f"rolling_models_exp={rolling_models_exp}",
                f"final_exp={final_exp}",
                f"recorder_id={rec.id}",
                f"window_key={window_key}",
                f"model_key={model_key}",
                f"handler_class={handler_class}",
                f"handler_module_path={handler_module_path}",
                f"account={account}",
                f"risk_degree={risk_degree}",
                f"provider_uri={provider_uri}",
                f"benchmark={benchmark}",
                f"instruments={instruments}",
                f"strategy_class={strategy_class}",
                f"strategy_module_path={strategy_module_path}",
                f"strategy_kwargs_extra={strategy_kwargs_extra}",
                f"eval_segment={eval_segment}",
                f"backtest_segment={backtest_segment or eval_segment}",
                f"handler_kwargs_extra={handler_kwargs_extra}",
                f"model_kwargs_override={model_kwargs_override}",
                f"cache_path={cache_path}",
                f"IC={metrics.get('IC')}",
                f"Rank IC={metrics.get('Rank IC')}",
                f"with_cost_ann_return={_metric_value(metrics, rec.id, '1day.excess_return_with_cost.annualized_return')}",
                f"with_cost_ir={_metric_value(metrics, rec.id, '1day.excess_return_with_cost.information_ratio')}",
                f"with_cost_mdd={_metric_value(metrics, rec.id, '1day.excess_return_with_cost.max_drawdown')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with_cost_ann_return = _metric_value(metrics, rec.id, "1day.excess_return_with_cost.annualized_return")
    with_cost_ir = _metric_value(metrics, rec.id, "1day.excess_return_with_cost.information_ratio")
    with_cost_mdd = _metric_value(metrics, rec.id, "1day.excess_return_with_cost.max_drawdown")

    print("\n=== Rolling ALSTM Result ===")
    print(f"IC: {metrics.get('IC')}")
    print(f"Rank IC: {metrics.get('Rank IC')}")
    print(f"with_cost_ann_return: {with_cost_ann_return}")
    print(f"with_cost_ir: {with_cost_ir}")
    print(f"with_cost_mdd: {with_cost_mdd}")
    print(f"saved summary: {result_path}")
    print(f"saved cache: {cache_path}")
    return {
        "result_path": str(result_path),
        "cache_path": str(cache_path),
        "recorder_id": rec.id,
        "IC": metrics.get("IC"),
        "Rank IC": metrics.get("Rank IC"),
        "with_cost_ann_return": with_cost_ann_return,
        "with_cost_ir": with_cost_ir,
        "with_cost_mdd": with_cost_mdd,
        "eval_segment": eval_segment,
        "backtest_segment": backtest_segment or eval_segment,
    }


def run_from_config(config_path: str):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    qlib_init = config.get("qlib_init", {})
    rolling = config.get("rolling", {})
    data = config.get("data", {})
    strategy = config.get("strategy", {})
    model = config.get("model", {})

    run(
        step=rolling.get("step", 120),
        topk=strategy.get("topk", 5),
        n_drop=strategy.get("n_drop", 1),
        window_key=data.get("window_key", "w1"),
        model_key=model.get("model_key", "gru_wide"),
        exp_suffix=config.get("exp_suffix", ""),
        handler_class=data.get("handler_class", "Alpha360"),
        handler_module_path=data.get("handler_module_path", "qlib.contrib.data.handler"),
        account=strategy.get("account", 150000),
        risk_degree=strategy.get("risk_degree", 0.70),
        provider_uri=qlib_init.get("provider_uri", PROVIDER_URI),
        benchmark=data.get("benchmark", "SH000300"),
        instruments=data.get("instruments", "csi300"),
        model_kwargs_override=model.get("kwargs"),
        handler_kwargs_extra=data.get("handler_kwargs_extra"),
        strategy_class=strategy.get("class", "TopkDropoutStrategy"),
        strategy_module_path=strategy.get("module_path", "qlib.contrib.strategy"),
        strategy_kwargs_extra=strategy.get("kwargs_extra"),
        gpu_slots=rolling.get("gpu_slots", "0,1"),
        eval_segment=rolling.get("eval_segment", "test"),
        backtest_segment=rolling.get("backtest_segment"),
        deterministic_runtime=not config.get("legacy_runtime", False),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rolling Alpha360 + ALSTM experiments.")
    parser.add_argument("command", choices=["run", "run_from_config", "train_task"])
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--step", type=int, default=120)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--n-drop", dest="n_drop", type=int, default=1)
    parser.add_argument("--window-key", dest="window_key", default="w1")
    parser.add_argument("--model-key", dest="model_key", default="gru_wide")
    parser.add_argument("--exp-suffix", dest="exp_suffix", default="")
    parser.add_argument("--account", type=int, default=150000)
    parser.add_argument("--risk-degree", dest="risk_degree", type=float, default=0.70)
    parser.add_argument("--gpu")
    parser.add_argument("--gpu-slots", dest="gpu_slots", default="0,1")
    parser.add_argument("--task-path", dest="task_path")
    parser.add_argument("--experiment-name", dest="experiment_name")
    parser.add_argument("--provider-uri", dest="provider_uri")
    parser.add_argument("--gpu-slot", dest="gpu_slot", type=int)
    parser.add_argument("--eval-segment", dest="eval_segment", default="test")
    parser.add_argument("--backtest-segment", dest="backtest_segment")
    parser.add_argument("--legacy-runtime", action="store_true", help="Disable deterministic ALSTM runtime controls.")
    args = parser.parse_args()

    if args.command == "run_from_config":
        if not args.config_path:
            parser.error("--config is required for run_from_config")
        run_from_config(args.config_path)
    elif args.command == "train_task":
        if not args.task_path or not args.experiment_name or not args.provider_uri:
            parser.error("--task-path, --experiment-name and --provider-uri are required for train_task")
        qlib.init(provider_uri=args.provider_uri, region=REG_CN)
        with Path(args.task_path).open("rb") as fp:
            task = pickle.load(fp)
        task["model"]["kwargs"]["GPU"] = 0
        recorder = task_train(task, args.experiment_name)
        print(f"task recorder={recorder.id} physical_gpu={args.gpu_slot}")
    else:
        model_kwargs_override = {"GPU": args.gpu} if args.gpu is not None else None
        run(
            step=args.step,
            topk=args.topk,
            n_drop=args.n_drop,
            window_key=args.window_key,
            model_key=args.model_key,
            exp_suffix=args.exp_suffix,
            account=args.account,
            risk_degree=args.risk_degree,
            model_kwargs_override=model_kwargs_override,
            gpu_slots=args.gpu_slots,
            eval_segment=args.eval_segment,
            backtest_segment=args.backtest_segment,
            deterministic_runtime=not args.legacy_runtime,
        )
