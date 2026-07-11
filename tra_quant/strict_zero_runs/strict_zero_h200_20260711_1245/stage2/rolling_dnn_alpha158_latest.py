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
import yaml
from qlib.constant import REG_CN
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.ens.ensemble import RollingEnsemble
from qlib.model.trainer import TrainerR, task_train
from qlib.utils import hash_args, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord
from qlib.workflow.task.collect import RecorderCollector
from qlib.workflow.task.gen import RollingGen, task_generator


BASE_DIR = Path(__file__).resolve().parent
PROVIDER_URI = str((BASE_DIR.parent.parent / "training_data" / "cn_data_latest").resolve())
CACHE_DIR = BASE_DIR / "dnn_cache"
HANDLER_CACHE_DIR = CACHE_DIR / "handlers"

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
}

MODEL_PRESETS = {
    "dnn_base": {
        "class": "DNNModelPytorch",
        "module_path": "qlib.contrib.model.pytorch_nn",
        "kwargs": {
            "loss": "mse",
            "lr": 0.002,
            "optimizer": "adam",
            "max_steps": 4000,
            "batch_size": 65536,
            "early_stop_rounds": 50,
            "eval_steps": 50,
            "GPU": 0,
            "weight_decay": 0.0002,
            "seed": 42,
            "pt_model_kwargs": {
                "layers": (256,),
            },
        },
    },
    "dnn_wide": {
        "class": "DNNModelPytorch",
        "module_path": "qlib.contrib.model.pytorch_nn",
        "kwargs": {
            "loss": "mse",
            "lr": 0.0015,
            "optimizer": "adam",
            "max_steps": 5000,
            "batch_size": 65536,
            "early_stop_rounds": 60,
            "eval_steps": 50,
            "GPU": 0,
            "weight_decay": 0.0003,
            "seed": 42,
            "pt_model_kwargs": {
                "layers": (512, 256),
            },
        },
    },
}

DEFAULT_HANDLER_KWARGS_EXTRA = {
    "infer_processors": [
        {
            "class": "DropCol",
            "kwargs": {"col_list": ["VWAP0"]},
        },
        {
            "class": "CSZFillna",
            "kwargs": {"fields_group": "feature"},
        },
    ],
    "learn_processors": [
        {
            "class": "DropCol",
            "kwargs": {"col_list": ["VWAP0"]},
        },
        {
            "class": "DropnaProcessor",
            "kwargs": {"fields_group": "feature"},
        },
        "DropnaLabel",
        {
            "class": "CSZScoreNorm",
            "kwargs": {"fields_group": "label"},
        },
    ],
    "process_type": "independent",
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
    normalized = copy.deepcopy(model_kwargs)
    float_keys = {"lr", "weight_decay"}
    int_keys = {"max_steps", "batch_size", "early_stop_rounds", "eval_steps", "seed"}

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
    data_handler_config = {
        "start_time": window["start_time"],
        "end_time": window["end_time"],
        "fit_start_time": window["fit_start_time"],
        "fit_end_time": window["fit_end_time"],
        "instruments": instruments,
    }
    data_handler_config.update(_deep_merge(DEFAULT_HANDLER_KWARGS_EXTRA, handler_kwargs_extra))
    return {
        "class": handler_class,
        "module_path": handler_module_path,
        "kwargs": data_handler_config,
    }


def _prepare_handler_cache(
    window_key: str,
    handler_class: str,
    handler_module_path: str,
    instruments: str,
    handler_kwargs_extra: dict | None,
) -> tuple[str, int]:
    HANDLER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    handler_config = _build_handler_config(
        window_key=window_key,
        handler_class=handler_class,
        handler_module_path=handler_module_path,
        instruments=instruments,
        handler_kwargs_extra=handler_kwargs_extra,
    )
    handler_hash = hash_args(handler_config)
    handler_path = HANDLER_CACHE_DIR / f"{handler_class}.{handler_hash[:10]}.cast.pkl"
    handler_uri = f"file://{handler_path}"
    if not handler_path.exists():
        handler = init_instance_by_config(handler_config, accept_types=DataHandlerLP)
        DataHandlerLP.cast(handler).to_pickle(handler_path, dump_all=True)
    try:
        handler = init_instance_by_config(handler_uri, accept_types=DataHandlerLP)
    except Exception:
        handler = init_instance_by_config(handler_config, accept_types=DataHandlerLP)
        DataHandlerLP.cast(handler).to_pickle(handler_path, dump_all=True)
        handler = init_instance_by_config(handler_uri, accept_types=DataHandlerLP)
    input_dim = len(handler.get_cols(col_set="feature", data_key=DataHandlerLP.DK_I))
    if input_dim <= 0:
        raise ValueError("empty feature columns when preparing cached handler")
    return handler_uri, int(input_dim)


def _build_dataset_config(
    window_key: str,
    handler_class: str,
    handler_module_path: str,
    instruments: str,
    handler_kwargs_extra: dict | None,
    eval_segment: str,
    handler_override=None,
):
    window = WINDOWS[window_key]
    handler = handler_override or _build_handler_config(
        window_key=window_key,
        handler_class=handler_class,
        handler_module_path=handler_module_path,
        instruments=instruments,
        handler_kwargs_extra=handler_kwargs_extra,
    )
    return {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler,
            "segments": {
                "train": window["train"],
                "valid": window["valid"],
                "test": _resolve_segment(window, eval_segment),
            },
        },
    }


def build_base_task(
    topk: int = 30,
    n_drop: int = 3,
    window_key: str = "w1",
    model_key: str = "dnn_base",
    handler_class: str = "Alpha158",
    handler_module_path: str = "qlib.contrib.data.handler",
    account: int = 100000000,
    risk_degree: float = 0.95,
    benchmark: str = "SH000300",
    instruments: str = "csi300",
    model_kwargs_override: dict | None = None,
    strategy_class: str = "TopkDropoutStrategy",
    strategy_module_path: str = "qlib.contrib.strategy",
    strategy_kwargs_extra: dict | None = None,
    handler_kwargs_extra: dict | None = None,
    eval_segment: str = "test",
    backtest_segment: str | None = None,
    handler_override=None,
    input_dim_override: int | None = None,
) -> dict:
    window = WINDOWS[window_key]
    model_conf = MODEL_PRESETS[model_key]
    eval_range = _resolve_segment(window, eval_segment)
    backtest_range = _resolve_segment(window, backtest_segment or eval_segment)
    input_dim = input_dim_override
    if input_dim is None:
        _, input_dim = _prepare_handler_cache(
            window_key=window_key,
            handler_class=handler_class,
            handler_module_path=handler_module_path,
            instruments=instruments,
            handler_kwargs_extra=handler_kwargs_extra,
        )
    model_kwargs = _normalize_model_kwargs(_deep_merge(model_conf["kwargs"], model_kwargs_override))
    model_kwargs["pt_model_kwargs"] = _deep_merge(model_kwargs["pt_model_kwargs"], {"input_dim": input_dim})

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

    return {
        "model": {
            "class": model_conf["class"],
            "module_path": model_conf["module_path"],
            "kwargs": model_kwargs,
        },
        "dataset": _build_dataset_config(
            window_key=window_key,
            handler_class=handler_class,
            handler_module_path=handler_module_path,
            instruments=instruments,
            handler_kwargs_extra=handler_kwargs_extra,
            eval_segment=eval_segment,
            handler_override=handler_override,
        ),
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


def _train_tasks(task_list: list[dict], experiment_name: str, provider_uri: str, gpu_slots):
    slots = _normalize_gpu_slots(gpu_slots)
    R.get_exp(experiment_name=experiment_name)
    if len(slots) <= 1:
        task_list = [copy.deepcopy(task) for task in task_list]
        if slots:
            for task in task_list:
                task["model"]["kwargs"]["GPU"] = slots[0]
        trainer = TrainerR(experiment_name=experiment_name)
        trainer(task_list)
        return

    script_path = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="dnn_tasks_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        pending = []
        retries = {}
        for idx, task in enumerate(task_list):
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


def run(
    step: int = 60,
    topk: int = 30,
    n_drop: int = 3,
    window_key: str = "w1",
    model_key: str = "dnn_base",
    exp_suffix: str = "",
    handler_class: str = "Alpha158",
    handler_module_path: str = "qlib.contrib.data.handler",
    account: int = 100000000,
    risk_degree: float = 0.95,
    provider_uri: str = PROVIDER_URI,
    benchmark: str = "SH000300",
    instruments: str = "csi300",
    model_kwargs_override: dict | None = None,
    strategy_class: str = "TopkDropoutStrategy",
    strategy_module_path: str = "qlib.contrib.strategy",
    strategy_kwargs_extra: dict | None = None,
    handler_kwargs_extra: dict | None = None,
    gpu_slots="0,1",
    eval_segment: str = "test",
    backtest_segment: str | None = None,
):
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    handler_uri, input_dim = _prepare_handler_cache(
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
        strategy_class=strategy_class,
        strategy_module_path=strategy_module_path,
        strategy_kwargs_extra=strategy_kwargs_extra,
        handler_kwargs_extra=handler_kwargs_extra,
        handler_override=handler_uri,
        input_dim_override=input_dim,
        eval_segment=eval_segment,
        backtest_segment=backtest_segment,
    )
    suffix = f"_{exp_suffix}" if exp_suffix else ""
    rolling_models_exp = (
        f"rolling_models_dnn_latest_{window_key}_{model_key}_{handler_class}_s{step}_k{topk}_d{n_drop}_a{account}"
        f"_r{str(risk_degree).replace('.', '')}{suffix}"
    )
    final_exp = (
        f"rolling_eval_dnn_latest_{window_key}_{model_key}_{handler_class}_s{step}_k{topk}_d{n_drop}_a{account}"
        f"_r{str(risk_degree).replace('.', '')}{suffix}"
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
    print(f"cached handler: {handler_uri}")
    print(f"input_dim: {input_dim}")

    _train_tasks(task_list, rolling_models_exp, provider_uri, gpu_slots)

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
            strategy_class=strategy_class,
            strategy_module_path=strategy_module_path,
            strategy_kwargs_extra=strategy_kwargs_extra,
            handler_kwargs_extra=handler_kwargs_extra,
            handler_override=handler_uri,
            input_dim_override=input_dim,
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

    result_path = BASE_DIR / (
        f"rolling_result_dnn_{window_key}_{model_key}_{handler_class}_step{step}_k{topk}_d{n_drop}_a{account}"
        f"_r{str(risk_degree).replace('.', '')}{suffix}.txt"
    )
    cache_path = CACHE_DIR / (
        f"rolling_cache_dnn_{window_key}_{model_key}_{handler_class}_step{step}_k{topk}_d{n_drop}_a{account}"
        f"_r{str(risk_degree).replace('.', '')}{suffix}.pkl"
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
                f"handler_kwargs_extra={_deep_merge(DEFAULT_HANDLER_KWARGS_EXTRA, handler_kwargs_extra)}",
                f"handler_cache_path={handler_uri}",
                f"cache_path={cache_path}",
                f"eval_segment={eval_segment}",
                f"backtest_segment={backtest_segment or eval_segment}",
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

    return {
        "result_path": str(result_path),
        "cache_path": str(cache_path),
        "recorder_id": rec.id,
        "IC": metrics.get("IC"),
        "Rank IC": metrics.get("Rank IC"),
        "with_cost_ann_return": _metric_value(metrics, rec.id, "1day.excess_return_with_cost.annualized_return"),
        "with_cost_ir": _metric_value(metrics, rec.id, "1day.excess_return_with_cost.information_ratio"),
        "with_cost_mdd": _metric_value(metrics, rec.id, "1day.excess_return_with_cost.max_drawdown"),
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
        topk=strategy.get("topk", 30),
        n_drop=strategy.get("n_drop", 3),
        window_key=data.get("window_key", "w1"),
        model_key=model.get("model_key", "dnn_base"),
        exp_suffix=config.get("exp_suffix", ""),
        handler_class=data.get("handler_class", "Alpha158"),
        handler_module_path=data.get("handler_module_path", "qlib.contrib.data.handler"),
        account=strategy.get("account", 100000000),
        risk_degree=strategy.get("risk_degree", 0.95),
        provider_uri=qlib_init.get("provider_uri", PROVIDER_URI),
        benchmark=data.get("benchmark", "SH000300"),
        instruments=data.get("instruments", "csi300"),
        model_kwargs_override=model.get("kwargs"),
        strategy_class=strategy.get("class", "TopkDropoutStrategy"),
        strategy_module_path=strategy.get("module_path", "qlib.contrib.strategy"),
        strategy_kwargs_extra=strategy.get("kwargs_extra"),
        handler_kwargs_extra=data.get("handler_kwargs_extra"),
        gpu_slots=rolling.get("gpu_slots", "0,1"),
        eval_segment=rolling.get("eval_segment", "test"),
        backtest_segment=rolling.get("backtest_segment"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rolling Alpha158 + DNN experiments.")
    parser.add_argument("command", choices=["run", "run_from_config", "train_task"])
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--step", type=int, default=120)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--n-drop", dest="n_drop", type=int, default=3)
    parser.add_argument("--window-key", dest="window_key", default="w1")
    parser.add_argument("--model-key", dest="model_key", default="dnn_base")
    parser.add_argument("--exp-suffix", dest="exp_suffix", default="")
    parser.add_argument("--account", type=int, default=100000000)
    parser.add_argument("--risk-degree", dest="risk_degree", type=float, default=0.95)
    parser.add_argument("--gpu")
    parser.add_argument("--gpu-slots", dest="gpu_slots", default="0,1")
    parser.add_argument("--task-path", dest="task_path")
    parser.add_argument("--experiment-name", dest="experiment_name")
    parser.add_argument("--provider-uri", dest="provider_uri")
    parser.add_argument("--gpu-slot", dest="gpu_slot", type=int)
    parser.add_argument("--eval-segment", dest="eval_segment", default="test")
    parser.add_argument("--backtest-segment", dest="backtest_segment")
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
        )
