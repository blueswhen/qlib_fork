from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import pickle
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
import yaml
from qlib.constant import REG_CN
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import StaticDataLoader
from qlib.model.ens.ensemble import RollingEnsemble
from qlib.model.trainer import TrainerR, _exe_task, _log_task_info, task_train
from qlib.utils import hash_args, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord
from qlib.workflow.task.collect import RecorderCollector
from qlib.workflow.task.gen import RollingGen, task_generator


BASE_DIR = Path(__file__).resolve().parent
PROVIDER_URI = os.environ.get("QLIB_PROVIDER_URI", str((BASE_DIR.parent / "training_data" / "cn_data_latest").resolve()))
CACHE_DIR = BASE_DIR / "tra_cache"
HANDLER_CACHE_DIR = CACHE_DIR / "handlers"
DETERMINISTIC_SEED = 42
DEFAULT_PARALLEL_MODE = "fork_readonly"

_FORK_PRELOADED_HANDLERS: dict[str, DataHandlerLP] = {}


def _deterministic_runtime_enabled(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit
    return os.environ.get("QLIB_TRA_DETERMINISTIC", "1").lower() not in {"0", "false", "no"}


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
        "start_time": "2016-01-01",
        "end_time": "2026-04-01",
        "fit_start_time": "2016-01-01",
        "fit_end_time": "2021-12-31",
        "train": ["2016-01-01", "2021-12-31"],
        "valid": ["2022-01-01", "2022-12-31"],
        "test": ["2023-01-01", "2026-04-01"],
    },
}

MODEL_PRESETS = {
    "tra_lstm_base": {
        "class": "TRAModel",
        "module_path": "qlib.contrib.model.pytorch_tra",
        "kwargs": {
            "model_config": {
                "input_size": 6,
                "hidden_size": 64,
                "num_layers": 2,
                "rnn_arch": "LSTM",
                "use_attn": True,
                "dropout": 0.0,
            },
            "tra_config": {
                "num_states": 3,
                "rnn_arch": "LSTM",
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
                "tau": 1.0,
                "src_info": "LR_TPE",
            },
            "model_type": "RNN",
            "lr": 1e-3,
            "n_epochs": 100,
            "early_stop": 20,
            "max_steps_per_epoch": None,
            "lamb": 1.0,
            "rho": 0.99,
            "alpha": 0.5,
            "seed": 42,
            "logdir": None,
            "eval_train": False,
            "eval_test": False,
            "pretrain": True,
            "init_state": None,
            "reset_router": False,
            "freeze_model": False,
            "freeze_predictors": False,
            "transport_method": "router",
            "memory_mode": "sample",
        },
    },
    "tra_gru_base": {
        "class": "TRAModel",
        "module_path": "qlib.contrib.model.pytorch_tra",
        "kwargs": {
            "model_config": {
                "input_size": 6,
                "hidden_size": 64,
                "num_layers": 2,
                "rnn_arch": "GRU",
                "use_attn": True,
                "dropout": 0.0,
            },
            "tra_config": {
                "num_states": 3,
                "rnn_arch": "GRU",
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
                "tau": 1.0,
                "src_info": "LR_TPE",
            },
            "model_type": "RNN",
            "lr": 1e-3,
            "n_epochs": 100,
            "early_stop": 20,
            "max_steps_per_epoch": None,
            "lamb": 1.0,
            "rho": 0.99,
            "alpha": 0.5,
            "seed": 42,
            "logdir": None,
            "eval_train": False,
            "eval_test": False,
            "pretrain": True,
            "init_state": None,
            "reset_router": False,
            "freeze_model": False,
            "freeze_predictors": False,
            "transport_method": "router",
            "memory_mode": "sample",
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
    mlruns_dir = BASE_DIR / "mlruns"
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


def _deep_merge(base: dict, extra: dict | None) -> dict:
    if extra is None:
        return copy.deepcopy(base)
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_gpu_slots(gpu_slots):
    if gpu_slots is None:
        return []
    if isinstance(gpu_slots, (list, tuple)):
        return [int(slot) for slot in gpu_slots]
    if isinstance(gpu_slots, str):
        return [int(slot.strip()) for slot in gpu_slots.split(",") if slot.strip()]
    return [int(gpu_slots)]


def _resolve_segment(window: dict, segment_key: str):
    if segment_key not in window:
        raise ValueError(f"unknown segment `{segment_key}`; available keys: {sorted(window.keys())}")
    return list(window[segment_key])


def _build_handler_config(
    window_key: str,
    handler_class: str,
    handler_module_path: str,
    instruments: str,
    handler_kwargs_extra: dict | None,
) -> dict:
    window = WINDOWS[window_key]
    handler_kwargs = _deep_merge(DEFAULT_HANDLER_KWARGS_EXTRA, handler_kwargs_extra)
    return {
        "class": handler_class,
        "module_path": handler_module_path,
        "kwargs": {
            "start_time": window["start_time"],
            "end_time": window["end_time"],
            "fit_start_time": window["fit_start_time"],
            "fit_end_time": window["fit_end_time"],
            "instruments": instruments,
            **handler_kwargs,
        },
    }


def _prepare_handler_cache(
    window_key: str,
    handler_class: str,
    handler_module_path: str,
    instruments: str,
    handler_kwargs_extra: dict | None,
) -> str:
    HANDLER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    handler_config = _build_handler_config(
        window_key=window_key,
        handler_class=handler_class,
        handler_module_path=handler_module_path,
        instruments=instruments,
        handler_kwargs_extra=handler_kwargs_extra,
    )
    handler_hash = hash_args(handler_config)
    handler_path = HANDLER_CACHE_DIR / f"{handler_class}.{handler_hash[:10]}.safe.pkl"
    handler_uri = f"file://{handler_path}"
    if handler_uri in _FORK_PRELOADED_HANDLERS and handler_path.exists():
        return handler_uri
    if not handler_path.exists():
        handler = init_instance_by_config(handler_config, accept_types=DataHandlerLP)
        cached_handler = DataHandlerLP.cast(handler)
        cached_handler.data_loader = StaticDataLoader(pd.DataFrame())
        cached_handler.data_loader.fields = copy.deepcopy(handler.data_loader.fields)
        cached_handler.to_pickle(handler_path, dump_all=True)
    try:
        init_instance_by_config(handler_uri, accept_types=DataHandlerLP)
    except Exception:
        handler = init_instance_by_config(handler_config, accept_types=DataHandlerLP)
        cached_handler = DataHandlerLP.cast(handler)
        cached_handler.data_loader = StaticDataLoader(pd.DataFrame())
        cached_handler.data_loader.fields = copy.deepcopy(handler.data_loader.fields)
        cached_handler.to_pickle(handler_path, dump_all=True)
    return handler_uri


def _preload_fork_handlers(task_list: list[dict]):
    handler_uris = sorted(
        {
            str(task["dataset"]["kwargs"]["handler"])
            for task in task_list
            if isinstance(task["dataset"]["kwargs"].get("handler"), (str, Path))
        }
    )
    missing_handler_uris = [handler_uri for handler_uri in handler_uris if handler_uri not in _FORK_PRELOADED_HANDLERS]
    if missing_handler_uris:
        _FORK_PRELOADED_HANDLERS.update(
            {
                handler_uri: init_instance_by_config(handler_uri, accept_types=DataHandlerLP)
                for handler_uri in missing_handler_uris
            }
        )
    return handler_uris


def _materialize_exec_task(task: dict) -> dict:
    exec_task = copy.deepcopy(task)
    handler_ref = exec_task.get("dataset", {}).get("kwargs", {}).get("handler")
    if isinstance(handler_ref, Path):
        handler_ref = str(handler_ref)
    if isinstance(handler_ref, str) and handler_ref in _FORK_PRELOADED_HANDLERS:
        exec_task["dataset"]["kwargs"]["handler"] = _FORK_PRELOADED_HANDLERS[handler_ref]
    return exec_task


def _run_task_worker(
    task: dict,
    experiment_name: str,
    provider_uri: str,
    gpu_slot: int,
    deterministic_runtime: bool,
    worker_threads: int,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_slot)
    os.environ["QLIB_TRA_DETERMINISTIC"] = "1" if deterministic_runtime else "0"
    os.environ["OMP_NUM_THREADS"] = str(worker_threads)
    os.environ["MKL_NUM_THREADS"] = str(worker_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(worker_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(worker_threads)
    if deterministic_runtime:
        _configure_deterministic_runtime()

    qlib.init(provider_uri=provider_uri, region=REG_CN)

    import torch

    torch.set_num_threads(worker_threads)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(worker_threads)

    exec_task = _materialize_exec_task(task)
    with R.start(experiment_name=experiment_name):
        _log_task_info(task)
        _exe_task(exec_task)
        recorder = R.get_recorder()
    print(f"task recorder={recorder.id} physical_gpu={gpu_slot} worker_threads={worker_threads}")


def _train_tasks(
    task_list: list[dict],
    experiment_name: str,
    provider_uri: str,
    gpu_slots,
    deterministic_runtime: bool,
    parallel_mode: str = DEFAULT_PARALLEL_MODE,
):
    slots = _normalize_gpu_slots(gpu_slots)
    if not slots:
        slots = [0]
    cpu_count = os.cpu_count() or len(slots) or 1
    configured_worker_threads = os.environ.get("QLIB_TRA_WORKER_THREADS")
    if configured_worker_threads:
        worker_threads = max(1, int(configured_worker_threads))
    else:
        # TRA rolling spends a lot of time in pandas/numpy preprocessing.
        # A small per-process thread cap avoids turning the whole run CPU-bound.
        worker_threads = max(1, min(2, cpu_count // max(1, len(slots))))
    R.get_exp(experiment_name=experiment_name)
    if parallel_mode == "fork_readonly":
        start_methods = mp.get_all_start_methods()
        if "fork" not in start_methods:
            raise RuntimeError("fork_readonly parallel mode requires a Python runtime with fork support")
        preloaded_handlers = _preload_fork_handlers(task_list)
        if preloaded_handlers:
            print(
                f"preloaded {len(preloaded_handlers)} shared TRA handler(s) before fork for read-only memory reuse"
            )
        ctx = mp.get_context("fork")
    elif parallel_mode == "trainer":
        preloaded_handlers = _preload_fork_handlers(task_list)
        if preloaded_handlers:
            print(f"using {len(preloaded_handlers)} preloaded TRA handler(s) in inline trainer mode")
        for idx, task in enumerate(task_list):
            print(f"launch rolling TRA task {idx + 1}/{len(task_list)} inline worker_threads={worker_threads}")
            with R.start(experiment_name=experiment_name):
                _log_task_info(task)
                _exe_task(_materialize_exec_task(task))
                recorder = R.get_recorder()
            print(f"finished rolling TRA task {idx + 1}/{len(task_list)} recorder={recorder.id}")
        return
    else:
        raise ValueError(f"unsupported parallel_mode `{parallel_mode}`")

    pending = [(idx, task) for idx, task in enumerate(task_list)]
    retries = {}
    running = {}
    available_slots = slots.copy()
    while pending or running:
        while pending and available_slots:
            idx, task = pending.pop(0)
            gpu = available_slots.pop(0)
            print(
                f"launch rolling TRA task {idx + 1}/{len(task_list)} on physical gpu {gpu} "
                f"mode={parallel_mode} worker_threads={worker_threads}"
            )
            proc = ctx.Process(
                target=_run_task_worker,
                args=(task, experiment_name, provider_uri, gpu, deterministic_runtime, worker_threads),
            )
            proc.start()
            running[proc] = (idx, gpu, task)
            time.sleep(1)

        finished = []
        for proc, (idx, gpu, task) in list(running.items()):
            retcode = proc.exitcode
            if retcode is None:
                continue
            proc.join()
            if retcode != 0:
                retry_count = retries.get(idx, 0)
                if retry_count < 2:
                    retries[idx] = retry_count + 1
                    print(
                        f"rolling TRA task {idx + 1}/{len(task_list)} on gpu {gpu} failed with exit code {retcode}; "
                        f"retry {retries[idx]}/2"
                    )
                    pending.insert(0, (idx, task))
                    available_slots.append(gpu)
                    finished.append(proc)
                    continue
                raise RuntimeError(f"rolling TRA task {idx + 1} on gpu {gpu} failed with exit code {retcode}")
            print(f"finished rolling TRA task {idx + 1}/{len(task_list)} on physical gpu {gpu}")
            available_slots.append(gpu)
            finished.append(proc)
        for proc in finished:
            running.pop(proc, None)
        if running:
            time.sleep(1)


def build_base_task(
    topk: int = 5,
    n_drop: int = 1,
    window_key: str = "w1",
    model_key: str = "tra_lstm_base",
    handler_class: str = "Alpha360",
    handler_module_path: str = "qlib.contrib.data.handler",
    account: int = 150000,
    risk_degree: float = 0.70,
    benchmark: str = "SH000300",
    instruments: str = "csi300",
    model_kwargs_override: dict | None = None,
    handler_kwargs_extra: dict | None = None,
    dataset_kwargs_extra: dict | None = None,
    strategy_class: str = "TopkDropoutStrategy",
    strategy_module_path: str = "qlib.contrib.strategy",
    strategy_kwargs_extra: dict | None = None,
    eval_segment: str = "test",
    backtest_segment: str | None = None,
    handler_override=None,
) -> dict:
    window = WINDOWS[window_key]
    model_conf = MODEL_PRESETS[model_key]
    model_kwargs = _deep_merge(model_conf["kwargs"], model_kwargs_override)
    handler = handler_override or _build_handler_config(
        window_key=window_key,
        handler_class=handler_class,
        handler_module_path=handler_module_path,
        instruments=instruments,
        handler_kwargs_extra=handler_kwargs_extra,
    )
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

    tra_states = int(model_kwargs["tra_config"]["num_states"])
    model_input = int(model_kwargs["model_config"]["input_size"])
    return {
        "model": {
            "class": model_conf["class"],
            "module_path": model_conf["module_path"],
            "kwargs": model_kwargs,
        },
        "dataset": {
            "class": "MTSDatasetH",
            "module_path": "qlib.contrib.data.dataset",
            "kwargs": _deep_merge({
                "handler": handler,
                "segments": {
                    "train": window["train"],
                    "valid": window["valid"],
                    "test": eval_range,
                },
                "seq_len": 60,
                "input_size": model_input,
                "num_states": tra_states,
                "batch_size": 16384,
                "n_samples": None,
                "memory_mode": model_kwargs.get("memory_mode", "sample"),
                "drop_last": True,
            }, dataset_kwargs_extra),
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
    step: int = 240,
    topk: int = 5,
    n_drop: int = 1,
    window_key: str = "w1",
    model_key: str = "tra_lstm_base",
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
    dataset_kwargs_extra: dict | None = None,
    strategy_class: str = "TopkDropoutStrategy",
    strategy_module_path: str = "qlib.contrib.strategy",
    strategy_kwargs_extra: dict | None = None,
    gpu_slots="0,1",
    eval_segment: str = "test",
    backtest_segment: str | None = None,
    rolling_task_limit: int | None = None,
    deterministic_runtime: bool = True,
    parallel_mode: str = DEFAULT_PARALLEL_MODE,
    handler_override=None,
):
    deterministic_runtime = _deterministic_runtime_enabled(deterministic_runtime)
    os.environ["QLIB_TRA_DETERMINISTIC"] = "1" if deterministic_runtime else "0"
    if deterministic_runtime and parallel_mode != "fork_readonly":
        _configure_deterministic_runtime()
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    cached_handler = handler_override or _prepare_handler_cache(
        window_key=window_key,
        handler_class=handler_class,
        handler_module_path=handler_module_path,
        instruments=instruments,
        handler_kwargs_extra=handler_kwargs_extra,
    )

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
        dataset_kwargs_extra=dataset_kwargs_extra,
        strategy_class=strategy_class,
        strategy_module_path=strategy_module_path,
        strategy_kwargs_extra=strategy_kwargs_extra,
        eval_segment=eval_segment,
        backtest_segment=backtest_segment,
        handler_override=cached_handler,
    )

    suffix = f"_{exp_suffix}" if exp_suffix else ""
    risk_tag = str(risk_degree).replace(".", "")
    rolling_models_exp = (
        f"rolling_models_tra_latest_{window_key}_{model_key}_{handler_class}_s{step}_k{topk}_d{n_drop}_a{account}_r{risk_tag}{suffix}"
    )
    final_exp = (
        f"rolling_eval_tra_latest_{window_key}_{model_key}_{handler_class}_s{step}_k{topk}_d{n_drop}_a{account}_r{risk_tag}{suffix}"
    )

    task_list = task_generator(base_task, RollingGen(step=step, rtype=RollingGen.ROLL_SD, trunc_days=2))
    total_task_count = len(task_list)
    if rolling_task_limit is not None:
        task_list = task_list[: max(1, int(rolling_task_limit))]
    for task in task_list:
        task["record"] = [
            {
                "class": "SignalRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"},
            }
        ]

    print(f"rolling TRA tasks: {len(task_list)}/{total_task_count}")
    print(f"rolling model experiment: {rolling_models_exp}")
    print(f"final evaluation experiment: {final_exp}")
    physical_gpu_slot = os.environ.get("QLIB_TRA_PHYSICAL_GPU_SLOT")
    if physical_gpu_slot is not None:
        print(f"physical gpu slot: {physical_gpu_slot}; logical gpu slots: {_normalize_gpu_slots(gpu_slots)}")
    else:
        print(f"gpu slots: {_normalize_gpu_slots(gpu_slots)}")
    print(f"deterministic runtime: {deterministic_runtime}")
    print(f"parallel mode: {parallel_mode}")

    _train_tasks(task_list, rolling_models_exp, provider_uri, gpu_slots, deterministic_runtime, parallel_mode)

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
            dataset_kwargs_extra=dataset_kwargs_extra,
            strategy_class=strategy_class,
            strategy_module_path=strategy_module_path,
            strategy_kwargs_extra=strategy_kwargs_extra,
            eval_segment=eval_segment,
            backtest_segment=backtest_segment,
            handler_override=cached_handler,
        )["record"]:
            if record["class"] == "SignalRecord":
                continue
            if record["class"] == "SigAnaRecord":
                SigAnaRecord(recorder=rec, **record["kwargs"]).generate()
            elif record["class"] == "PortAnaRecord":
                PortAnaRecord(recorder=rec, **record["kwargs"]).generate()

        metrics = rec.list_metrics()

    result_path = BASE_DIR / (
        f"rolling_result_tra_{window_key}_{model_key}_{handler_class}_step{step}"
        f"_k{topk}_d{n_drop}_a{account}_r{risk_tag}{suffix}.txt"
    )
    cache_path = CACHE_DIR / (
        f"rolling_cache_tra_{window_key}_{model_key}_{handler_class}_step{step}"
        f"_k{topk}_d{n_drop}_a{account}_r{risk_tag}{suffix}.pkl"
    )

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
                f"rolling_task_limit={rolling_task_limit}",
                f"rolling_task_count={len(task_list)}",
                f"rolling_task_total={total_task_count}",
                f"handler_kwargs_extra={handler_kwargs_extra}",
                f"dataset_kwargs_extra={dataset_kwargs_extra}",
                f"model_kwargs_override={model_kwargs_override}",
                f"deterministic_runtime={deterministic_runtime}",
                f"physical_gpu_slot={physical_gpu_slot}",
                f"parallel_mode={parallel_mode}",
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

    print("\n=== Rolling TRA Result ===")
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
        "rolling_task_limit": rolling_task_limit,
        "rolling_task_count": len(task_list),
        "rolling_task_total": total_task_count,
    }


def run_from_config(config_path: str):
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    qlib_init = config.get("qlib_init", {})
    rolling = config.get("rolling", {})
    data = config.get("data", {})
    strategy = config.get("strategy", {})
    model = config.get("model", {})

    run(
        step=rolling.get("step", 240),
        topk=strategy.get("topk", 5),
        n_drop=strategy.get("n_drop", 1),
        window_key=data.get("window_key", "w1"),
        model_key=model.get("model_key", "tra_lstm_base"),
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
        dataset_kwargs_extra=data.get("dataset_kwargs_extra"),
        strategy_class=strategy.get("class", "TopkDropoutStrategy"),
        strategy_module_path=strategy.get("module_path", "qlib.contrib.strategy"),
        strategy_kwargs_extra=strategy.get("kwargs_extra"),
        gpu_slots=rolling.get("gpu_slots", "0,1"),
        eval_segment=rolling.get("eval_segment", "test"),
        backtest_segment=rolling.get("backtest_segment"),
        rolling_task_limit=rolling.get("rolling_task_limit"),
        deterministic_runtime=rolling.get("deterministic_runtime", True),
        parallel_mode=rolling.get("parallel_mode", DEFAULT_PARALLEL_MODE),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rolling Alpha360 + TRA experiments.")
    parser.add_argument("command", choices=["run", "run_from_config", "train_task"])
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--step", type=int, default=240)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--n-drop", dest="n_drop", type=int, default=1)
    parser.add_argument("--window-key", dest="window_key", default="w1")
    parser.add_argument("--model-key", dest="model_key", default="tra_lstm_base")
    parser.add_argument("--exp-suffix", dest="exp_suffix", default="")
    parser.add_argument("--account", type=int, default=150000)
    parser.add_argument("--risk-degree", dest="risk_degree", type=float, default=0.70)
    parser.add_argument("--gpu-slots", dest="gpu_slots", default="0,1")
    parser.add_argument("--task-path", dest="task_path")
    parser.add_argument("--experiment-name", dest="experiment_name")
    parser.add_argument("--provider-uri", dest="provider_uri")
    parser.add_argument("--gpu-slot", dest="gpu_slot", type=int)
    parser.add_argument("--worker-threads", dest="worker_threads", type=int, default=1)
    parser.add_argument("--eval-segment", dest="eval_segment", default="test")
    parser.add_argument("--backtest-segment", dest="backtest_segment")
    parser.add_argument("--parallel-mode", dest="parallel_mode", default=DEFAULT_PARALLEL_MODE, choices=["fork_readonly", "trainer"])
    parser.add_argument("--legacy-runtime", action="store_true", help="disable forced deterministic runtime to mimic older TRA runs")
    args = parser.parse_args()

    if args.command == "run_from_config":
        if not args.config_path:
            parser.error("--config is required for run_from_config")
        run_from_config(args.config_path)
    elif args.command == "train_task":
        if not args.task_path or not args.experiment_name or not args.provider_uri:
            parser.error("--task-path, --experiment-name and --provider-uri are required for train_task")
        if _deterministic_runtime_enabled():
            _configure_deterministic_runtime()
        qlib.init(provider_uri=args.provider_uri, region=REG_CN)
        import torch

        torch.set_num_threads(args.worker_threads)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(args.worker_threads)
        with Path(args.task_path).open("rb") as fp:
            task = pickle.load(fp)
        recorder = task_train(task, args.experiment_name)
        print(f"task recorder={recorder.id} physical_gpu={args.gpu_slot} worker_threads={args.worker_threads}")
    else:
        run(
            step=args.step,
            topk=args.topk,
            n_drop=args.n_drop,
            window_key=args.window_key,
            model_key=args.model_key,
            exp_suffix=args.exp_suffix,
            account=args.account,
            risk_degree=args.risk_degree,
            gpu_slots=args.gpu_slots,
            eval_segment=args.eval_segment,
            backtest_segment=args.backtest_segment,
            deterministic_runtime=not args.legacy_runtime,
            parallel_mode=args.parallel_mode,
        )
