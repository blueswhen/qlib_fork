from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import qlib
import yaml
from qlib.constant import REG_CN
from qlib.model.trainer import task_train
from qlib.model.ens.ensemble import RollingEnsemble
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord
from qlib.workflow.task.collect import RecorderCollector
from qlib.workflow.task.gen import RollingGen, task_generator

from joint_tra_alpha360_tech158 import build_joint_task
from rolling_tra_alpha360_latest import (
    PROVIDER_URI,
    _configure_deterministic_runtime,
    _deterministic_runtime_enabled,
    _metric_value,
    _normalize_gpu_slots,
    _train_tasks,
)


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "joint_tra_cache"
MODEL_TAG = "joint_tra_alpha360_tech158_v1"
HANDLER_TAG = "Alpha360TechAlpha158Handler"
DATASET_TAG = "JointMTSDatasetH"


def run(
    step: int = 240,
    topk: int = 5,
    n_drop: int = 1,
    window_key: str = "w1",
    exp_suffix: str = "",
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
):
    deterministic_runtime = _deterministic_runtime_enabled(deterministic_runtime)
    if deterministic_runtime:
        _configure_deterministic_runtime()
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    base_task = build_joint_task(
        topk=topk,
        n_drop=n_drop,
        window_key=window_key,
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
    )

    suffix = f"_{exp_suffix}" if exp_suffix else ""
    risk_tag = str(risk_degree).replace(".", "")
    rolling_models_exp = (
        f"rolling_models_{MODEL_TAG}_{window_key}_s{step}_k{topk}_d{n_drop}_a{account}_r{risk_tag}{suffix}"
    )
    final_exp = f"rolling_eval_{MODEL_TAG}_{window_key}_s{step}_k{topk}_d{n_drop}_a{account}_r{risk_tag}{suffix}"

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

    print(f"rolling JointTRA tasks: {len(task_list)}/{total_task_count}")
    print(f"rolling model experiment: {rolling_models_exp}")
    print(f"final evaluation experiment: {final_exp}")
    print(f"gpu slots: {_normalize_gpu_slots(gpu_slots)}")
    print(f"deterministic runtime: {deterministic_runtime}")

    _train_tasks(task_list, rolling_models_exp, provider_uri, gpu_slots, deterministic_runtime)

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

        for record in build_joint_task(
            topk=topk,
            n_drop=n_drop,
            window_key=window_key,
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
        )["record"]:
            if record["class"] == "SignalRecord":
                continue
            if record["class"] == "SigAnaRecord":
                SigAnaRecord(recorder=rec, **record["kwargs"]).generate()
            elif record["class"] == "PortAnaRecord":
                PortAnaRecord(recorder=rec, **record["kwargs"]).generate()

        metrics = rec.list_metrics()

    result_path = BASE_DIR / (
        f"rolling_result_{MODEL_TAG}_{window_key}_step{step}_k{topk}_d{n_drop}_a{account}_r{risk_tag}{suffix}.txt"
    )
    cache_path = CACHE_DIR / (
        f"rolling_cache_{MODEL_TAG}_{window_key}_step{step}_k{topk}_d{n_drop}_a{account}_r{risk_tag}{suffix}.pkl"
    )

    CACHE_DIR.mkdir(exist_ok=True)
    Path(cache_path).write_bytes(pickle.dumps({"pred": res["pred"], "label": res["label"]}))

    with_cost_ann_return = _metric_value(metrics, rec.id, "1day.excess_return_with_cost.annualized_return")
    with_cost_ir = _metric_value(metrics, rec.id, "1day.excess_return_with_cost.information_ratio")
    with_cost_mdd = _metric_value(metrics, rec.id, "1day.excess_return_with_cost.max_drawdown")

    result_path.write_text(
        "\n".join(
            [
                f"rolling_models_exp={rolling_models_exp}",
                f"final_exp={final_exp}",
                f"recorder_id={rec.id}",
                f"window_key={window_key}",
                f"model_tag={MODEL_TAG}",
                f"handler_class={HANDLER_TAG}",
                f"dataset_class={DATASET_TAG}",
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
                f"cache_path={cache_path}",
                f"IC={metrics.get('IC')}",
                f"Rank IC={metrics.get('Rank IC')}",
                f"with_cost_ann_return={with_cost_ann_return}",
                f"with_cost_ir={with_cost_ir}",
                f"with_cost_mdd={with_cost_mdd}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n=== Rolling JointTRA Result ===")
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
        exp_suffix=config.get("exp_suffix", ""),
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
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rolling Alpha360 + TechAlpha158 JointTRA experiments.")
    parser.add_argument("command", choices=["run", "run_from_config", "train_task"])
    parser.add_argument("--config", dest="config_path")
    parser.add_argument("--step", type=int, default=240)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--n-drop", dest="n_drop", type=int, default=1)
    parser.add_argument("--window-key", dest="window_key", default="w1")
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
    parser.add_argument("--rolling-task-limit", dest="rolling_task_limit", type=int)
    parser.add_argument(
        "--legacy-runtime",
        action="store_true",
        help="disable forced deterministic runtime to mimic older JointTRA runs",
    )
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
            exp_suffix=args.exp_suffix,
            account=args.account,
            risk_degree=args.risk_degree,
            gpu_slots=args.gpu_slots,
            eval_segment=args.eval_segment,
            backtest_segment=args.backtest_segment,
            rolling_task_limit=args.rolling_task_limit,
            deterministic_runtime=not args.legacy_runtime,
        )
