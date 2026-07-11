"""Global TRA Alpha360 tuner.

Goal: rerun the broadened global training flow from ``tune_alstm_alpha360_global.py``
with TRA replacing ALSTM, while keeping the model search fixed to a single
TRA h5 base configuration. Search happens only on the strategy layer.

Selection target: max stable validation score based on yearly valid subsegments.
"""

from __future__ import annotations

import argparse
import copy
import csv
import multiprocessing as mp
import os
import queue as queue_module
import subprocess
import time
from ast import literal_eval
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.model.ens.ensemble import RollingEnsemble
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord
from qlib.workflow.task.collect import RecorderCollector
from qlib.workflow.task.gen import RollingGen, task_generator

try:
    from rolling_tra_alpha360_latest import (
        PROVIDER_URI,
        WINDOWS,
        _deep_merge,
        _metric_value,
        _normalize_gpu_slots,
        _prepare_handler_cache,
        _preload_fork_handlers as _preload_tra_fork_handlers,
        _run_task_worker as _run_tra_task_worker,
        build_base_task,
        run,
    )
except ModuleNotFoundError:
    from examples.my_strategy.rolling_tra_alpha360_latest import (
        PROVIDER_URI,
        WINDOWS,
        _deep_merge,
        _metric_value,
        _normalize_gpu_slots,
        _prepare_handler_cache,
        _preload_fork_handlers as _preload_tra_fork_handlers,
        _run_task_worker as _run_tra_task_worker,
        build_base_task,
        run,
    )


def _joint_model_config():
    try:
        from joint_tra_alpha360_tech158 import joint_model_config
    except ModuleNotFoundError:
        from examples.my_strategy.joint_tra_alpha360_tech158 import joint_model_config
    return joint_model_config()


def _run_joint(**kwargs):
    try:
        from rolling_joint_tra_alpha360_tech158_latest import run as run_joint
    except ModuleNotFoundError:
        from examples.my_strategy.rolling_joint_tra_alpha360_tech158_latest import run as run_joint
    return run_joint(**kwargs)


BASE_DIR = Path(__file__).resolve().parent
TRA_CACHE_DIR = BASE_DIR / "tra_cache"
VALIDATION_RESULT_PATH = BASE_DIR / "tra_global_validation_results.csv"
VALIDATION_BEST_PATH = BASE_DIR / "tra_global_validation_best.txt"
FINAL_TEST_PATH = BASE_DIR / "tra_global_final_test.txt"
JOINT_VALIDATION_RESULT_PATH = BASE_DIR / "joint_tra_alpha360_tech158_global_validation_results.csv"
JOINT_VALIDATION_BEST_PATH = BASE_DIR / "joint_tra_alpha360_tech158_global_validation_best.txt"
JOINT_FINAL_TEST_PATH = BASE_DIR / "joint_tra_alpha360_tech158_global_final_test.txt"
DUAL_SEED_GRID_PATH = BASE_DIR / "tra_global_dual_seed_valid_test_grid.csv"
DUAL_SEED_SUMMARY_PATH = BASE_DIR / "tra_global_dual_seed_pattern_summary.txt"
DUAL_SEED_RESOURCE_LOG_PATH = BASE_DIR / "tra_global_dual_seed_resource_log.csv"

ACCOUNT = 150000
TOPK = 5
WINDOW_KEY = "w1"
BENCHMARK = "SH000300"
INSTRUMENTS = "csi300"
GPU_SLOTS = "0,1"
DATASET_KWARGS = {"batch_size": 16384}
DUAL_SEED_BATCH_SIZE = 13312
DUAL_SEED_SMOKE_EPOCHS = 2
DUAL_SEED_ROLLING_JOBS_PER_GPU = 2
DUAL_SEED_DEFAULT_SEEDS = (2026, 3407)
DUAL_SEED_MIN_AVAILABLE_RAM_GB = 6.0
DUAL_SEED_MAX_EXTRA_SWAP_GB = 48.0
DUAL_SEED_RESOURCE_POLL_SECONDS = 2.0
DEFAULT_MODEL_FAMILY = "tra"

H5_LABEL = {"label": ["Ref($close, -6) / Ref($close, -1) - 1"]}

BASE_MODEL_KWARGS_OVERRIDE = {
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
    "update_freq": 1,
    "eval_freq": 5,
    "lamb": 1.0,
    "rho": 0.99,
    "alpha": 0.5,
    "pretrain": True,
    "transport_method": "router",
    "memory_mode": "sample",
}
SANITY_STRATEGY_TRIAL = "prac_m050_hold3_r085"
BASE_SANITY_MIN_ANN_RETURN = 0.0
VALIDATION_SCORE_MODE = "valid_yearly_stability_v1"
# First-pass stability weights: reward strong per-year averages, then penalize
# year-to-year dispersion and any negative worst-year annualized return.
VALIDATION_STABILITY_WEIGHTS = {
    "ann_std": 400.0,
    "ann_spread": 300.0,
    "negative_worst_ann": 2000.0,
}

MODEL_TRIALS = [
    {
        "trial_name": "strict_tra_lstm_s240_h5",
        "model_key": "tra_lstm_base",
        "step": 240,
        "handler_kwargs_extra": H5_LABEL,
    },
]

JOINT_MODEL_TRIALS = [
    {
        "trial_name": "joint_tra_alpha360_tech158_v1_s240_h5",
        "model_key": "joint_tra_alpha360_tech158_v1",
        "step": 240,
        "handler_kwargs_extra": H5_LABEL,
    },
]

_PRAC_CLASS = "PracticalTopkDropoutStrategy"
_PRAC_MODULE = "qlib.contrib.strategy.custom_signal_strategy"
_TOPK_CLASS = "TopkDropoutStrategy"
_TOPK_MODULE = "qlib.contrib.strategy"


def _prac(name, n_drop, risk_degree, score_margin, hold_thresh, slot_budget_ratio=1.0):
    return {
        "trial_name": name,
        "strategy_class": _PRAC_CLASS,
        "strategy_module_path": _PRAC_MODULE,
        "n_drop": n_drop,
        "risk_degree": risk_degree,
        "kwargs_extra": {
            "score_margin": score_margin,
            "hold_thresh": hold_thresh,
            "slot_budget_ratio": slot_budget_ratio,
        },
    }


def _topk(name, n_drop, risk_degree):
    return {
        "trial_name": name,
        "strategy_class": _TOPK_CLASS,
        "strategy_module_path": _TOPK_MODULE,
        "n_drop": n_drop,
        "risk_degree": risk_degree,
        "kwargs_extra": {},
    }


_SEARCH_HOLDS = (3, 4, 5)
_SEARCH_N_DROPS = tuple(range(1, TOPK + 1))
_SEARCH_RISKS = (0.70, 0.85, 0.95)
_SEARCH_MARGINS = (
    ("m000", 0.0),
    ("m025", 0.0025),
    ("m050", 0.005),
    ("m100", 0.010),
)


def _risk_tag(risk_degree: float) -> str:
    if risk_degree == 1.0:
        return "r10"
    return f"r{str(risk_degree).replace('.', '')}"


def _n_drop_tag(n_drop: int) -> str:
    return "" if n_drop == 1 else f"_drop{n_drop}"


def _build_stage1_strategy_trials() -> list[dict]:
    trials: list[dict] = []
    for hold_thresh in _SEARCH_HOLDS:
        for n_drop in _SEARCH_N_DROPS:
            for risk_degree in _SEARCH_RISKS:
                for margin_tag, score_margin in _SEARCH_MARGINS:
                    trials.append(
                        _prac(
                            f"prac_{margin_tag}{_n_drop_tag(n_drop)}_hold{hold_thresh}_{_risk_tag(risk_degree)}",
                            n_drop,
                            risk_degree,
                            score_margin,
                            hold_thresh,
                        )
                    )
    return trials


STRATEGY_TRIALS = _build_stage1_strategy_trials()

STRATEGY_TRIAL_BY_NAME = {trial["trial_name"]: trial for trial in STRATEGY_TRIALS}

STRATEGY_PROFILE_FULL = "full"
STRATEGY_PROFILE_TRA_BEST_72 = "tra_best_72"


def _strategy_trials_for_profile(strategy_profile: str) -> list[dict]:
    if strategy_profile == STRATEGY_PROFILE_FULL:
        return STRATEGY_TRIALS
    if strategy_profile == STRATEGY_PROFILE_TRA_BEST_72:
        allowed_holds = {3, 4, 5}
        allowed_n_drops = {1, 2, 3}
        allowed_risks = {0.85, 0.95}
        return [
            trial
            for trial in STRATEGY_TRIALS
            if trial["kwargs_extra"].get("hold_thresh") in allowed_holds
            and trial["n_drop"] in allowed_n_drops
            and float(trial["risk_degree"]) in allowed_risks
        ]
    raise ValueError(f"unsupported strategy_profile `{strategy_profile}`")


def _validation_result_path_for_family(model_family: str) -> Path:
    if model_family == "tra":
        return VALIDATION_RESULT_PATH
    if model_family == "joint_tra_tech158":
        return JOINT_VALIDATION_RESULT_PATH
    raise ValueError(f"unsupported model_family `{model_family}`")


def _validation_best_path_for_family(model_family: str) -> Path:
    if model_family == "tra":
        return VALIDATION_BEST_PATH
    if model_family == "joint_tra_tech158":
        return JOINT_VALIDATION_BEST_PATH
    raise ValueError(f"unsupported model_family `{model_family}`")


def _final_test_path_for_family(model_family: str) -> Path:
    if model_family == "tra":
        return FINAL_TEST_PATH
    if model_family == "joint_tra_tech158":
        return JOINT_FINAL_TEST_PATH
    raise ValueError(f"unsupported model_family `{model_family}`")


def _model_trials_for_family(model_family: str) -> list[dict]:
    if model_family == "tra":
        return MODEL_TRIALS
    if model_family == "joint_tra_tech158":
        return JOINT_MODEL_TRIALS
    raise ValueError(f"unsupported model_family `{model_family}`")


def _base_model_kwargs_override_for_family(model_family: str) -> dict:
    if model_family == "tra":
        return BASE_MODEL_KWARGS_OVERRIDE
    if model_family == "joint_tra_tech158":
        return _joint_model_config()
    raise ValueError(f"unsupported model_family `{model_family}`")


def _objective(metrics: dict) -> float:
    ann = float(metrics["with_cost_ann_return"])
    ir = float(metrics["with_cost_ir"])
    mdd = float(metrics["with_cost_mdd"])
    return ann * 1000 + ir * 50 + mdd * 20


def _row_matches_selection_config(row: dict) -> bool:
    return row.get("score_mode") == VALIDATION_SCORE_MODE


def _yearly_subsegments(date_range: list[str] | tuple[str, str], *, prefix: str) -> list[tuple[str, list[str]]]:
    start_ts = pd.Timestamp(date_range[0])
    end_ts = pd.Timestamp(date_range[1])
    segments: list[tuple[str, list[str]]] = []
    for year in range(start_ts.year, end_ts.year + 1):
        segment_start = max(start_ts, pd.Timestamp(year=year, month=1, day=1))
        segment_end = min(end_ts, pd.Timestamp(year=year, month=12, day=31))
        if segment_start <= segment_end:
            segments.append(
                (
                    f"{prefix}_{year}",
                    [segment_start.strftime("%Y-%m-%d"), segment_end.strftime("%Y-%m-%d")],
                )
            )
    return segments


def _validation_subsegments(window_key: str) -> list[tuple[str, list[str]]]:
    return _yearly_subsegments(WINDOWS[window_key]["valid"], prefix="valid")


def _suffix_path(path: Path, result_suffix: str) -> Path:
    if not result_suffix:
        return path
    return path.with_name(f"{path.stem}_{result_suffix}{path.suffix}")


def _trial_suffix(trial_name: str, result_suffix: str) -> str:
    if not result_suffix:
        return trial_name
    return f"{trial_name}_{result_suffix}"


def _parse_summary_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def _persist(rows: list[dict], path: Path):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    if "score" not in df.columns:
        df["score"] = pd.NA
    df.sort_values(by=["status", "score"], ascending=[True, False], na_position="last").to_csv(path, index=False)


def _row_key(model_trial: str, strategy_trial: str) -> tuple[str, str]:
    return model_trial, strategy_trial


def _load_existing_rows(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if df.empty:
        return {}

    row_map = {}
    for row in df.to_dict(orient="records"):
        row_map[_row_key(row["model_trial"], row["strategy_trial"])] = row
    return row_map


def _load_signal(cache_path: str | Path):
    cache = pd.read_pickle(cache_path)
    pred = cache["pred"].sort_index()
    label = cache["label"].sort_index()
    common_index = pred.index.intersection(label.index)
    return pred.loc[common_index].sort_index(), label.loc[common_index].sort_index()


def _slice_signal_to_range(frame: pd.DataFrame, date_range: list[str] | tuple[str, str]) -> pd.DataFrame:
    start_ts = pd.Timestamp(date_range[0])
    end_ts = pd.Timestamp(date_range[1])
    date_index = pd.Index(frame.index.get_level_values(0))
    mask = (date_index >= start_ts) & (date_index <= end_ts)
    return frame.loc[mask].sort_index()


def _parse_optional_literal(value: str):
    if value in ("None", "", None):
        return None
    return literal_eval(value)


def _resolve_reuse_artifact(cache_path: str | None, result_path: str | None) -> dict[str, str] | None:
    if not cache_path:
        return None

    resolved_cache_path = Path(cache_path).expanduser().resolve()
    if not resolved_cache_path.exists():
        raise FileNotFoundError(f"reuse cache path does not exist: {resolved_cache_path}")

    resolved_result_path = resolved_cache_path
    if result_path:
        resolved_result_path = Path(result_path).expanduser().resolve()
        if not resolved_result_path.exists():
            raise FileNotFoundError(f"reuse result path does not exist: {resolved_result_path}")

    return {
        "result_path": str(resolved_result_path),
        "cache_path": str(resolved_cache_path),
    }


def _model_kwargs_for_seed(seed: int, model_family: str, model_trial: dict | None = None) -> dict:
    if model_trial is not None and model_trial.get("model_kwargs_override") is not None:
        model_kwargs = copy.deepcopy(model_trial["model_kwargs_override"])
    else:
        model_kwargs = copy.deepcopy(_base_model_kwargs_override_for_family(model_family))
    model_kwargs["seed"] = seed
    return model_kwargs


def _run_family_trial(
    model_family: str,
    model_trial: dict,
    *,
    seed: int,
    gpu_slots: str,
    result_suffix: str,
    deterministic_runtime: bool,
    window_key: str,
    n_drop: int,
    risk_degree: float,
    strategy_class: str | None = None,
    strategy_module_path: str | None = None,
    strategy_kwargs_extra: dict | None = None,
    dataset_kwargs_extra: dict | None = None,
    eval_segment: str,
    backtest_segment: str,
    rolling_task_limit: int | None = None,
    parallel_mode: str = "fork_readonly",
):
    model_kwargs = _model_kwargs_for_seed(seed, model_family, model_trial)

    common_kwargs = {
        "step": model_trial["step"],
        "topk": TOPK,
        "n_drop": n_drop,
        "window_key": window_key,
        "exp_suffix": result_suffix,
        "account": ACCOUNT,
        "risk_degree": risk_degree,
        "provider_uri": PROVIDER_URI,
        "benchmark": BENCHMARK,
        "instruments": INSTRUMENTS,
        "model_kwargs_override": model_kwargs,
        "handler_kwargs_extra": model_trial["handler_kwargs_extra"],
        "dataset_kwargs_extra": dataset_kwargs_extra or DATASET_KWARGS,
        "gpu_slots": gpu_slots,
        "eval_segment": eval_segment,
        "backtest_segment": backtest_segment,
        "deterministic_runtime": deterministic_runtime,
    }
    if strategy_class is not None:
        common_kwargs["strategy_class"] = strategy_class
    if strategy_module_path is not None:
        common_kwargs["strategy_module_path"] = strategy_module_path
    if strategy_kwargs_extra is not None:
        common_kwargs["strategy_kwargs_extra"] = strategy_kwargs_extra

    if model_family == "tra":
        common_kwargs["rolling_task_limit"] = rolling_task_limit
        common_kwargs["parallel_mode"] = parallel_mode
        return run(
            model_key=model_trial["model_key"],
            handler_class="Alpha360",
            handler_module_path="qlib.contrib.data.handler",
            **common_kwargs,
        )
    if model_family == "joint_tra_tech158":
        return _run_joint(**common_kwargs)
    raise ValueError(f"unsupported model_family `{model_family}`")


def _row_matches_expected_config(row: dict, expected_model_kwargs: str, expected_dataset_kwargs: str) -> bool:
    return row.get("model_kwargs_override") == expected_model_kwargs and row.get("dataset_kwargs_extra") == expected_dataset_kwargs


def _expected_model_kwargs_by_trial(seed: int, model_family: str) -> dict[str, str]:
    return {
        trial["trial_name"]: str(_model_kwargs_for_seed(seed, model_family, trial))
        for trial in _model_trials_for_family(model_family)
    }


def _ordered_strategy_trials(strategy_trials: list[dict]) -> list[dict]:
    sanity_trial = STRATEGY_TRIAL_BY_NAME.get(SANITY_STRATEGY_TRIAL)
    if sanity_trial is None:
        return strategy_trials
    if all(trial["trial_name"] != SANITY_STRATEGY_TRIAL for trial in strategy_trials):
        return strategy_trials
    return [sanity_trial] + [trial for trial in strategy_trials if trial["trial_name"] != SANITY_STRATEGY_TRIAL]


def _evaluate_strategy(
    model_trial_name: str,
    strategy_trial: dict,
    pred: pd.DataFrame,
    label: pd.DataFrame,
    *,
    window_key: str,
) -> dict:
    result = _evaluate_signal(
        model_trial_name,
        strategy_trial,
        pred,
        label,
        window_key=window_key,
        eval_segment="valid",
        backtest_segment="valid",
        recorder_name=f"{model_trial_name}__{strategy_trial['trial_name']}__valid",
    )
    result["validation_recorder_id"] = result.pop("recorder_id")
    result["base_score"] = _objective(result)
    result.update(
        _evaluate_validation_stability(
            model_trial_name,
            strategy_trial,
            pred,
            label,
            window_key=window_key,
        )
    )
    return result


def _evaluate_validation_stability(
    model_trial_name: str,
    strategy_trial: dict,
    pred: pd.DataFrame,
    label: pd.DataFrame,
    *,
    window_key: str,
) -> dict:
    segment_results = []
    for segment_name, segment_range in _validation_subsegments(window_key):
        segment_result = _evaluate_signal(
            model_trial_name,
            strategy_trial,
            pred,
            label,
            window_key=window_key,
            eval_segment="valid",
            backtest_segment="valid",
            eval_range=segment_range,
            backtest_range=segment_range,
            recorder_name=f"{model_trial_name}__{strategy_trial['trial_name']}__{segment_name}",
        )
        segment_results.append({
            "name": segment_name,
            "range": segment_range,
            **segment_result,
        })

    ann_values = pd.Series([float(item["with_cost_ann_return"]) for item in segment_results], dtype="float64")
    ir_values = pd.Series([float(item["with_cost_ir"]) for item in segment_results], dtype="float64")
    mdd_values = pd.Series([float(item["with_cost_mdd"]) for item in segment_results], dtype="float64")

    segment_mean_metrics = {
        "with_cost_ann_return": float(ann_values.mean()),
        "with_cost_ir": float(ir_values.mean()),
        "with_cost_mdd": float(mdd_values.mean()),
    }
    segment_mean_score = _objective(segment_mean_metrics)
    ann_std = float(ann_values.std(ddof=0))
    ann_spread = float(ann_values.max() - ann_values.min())
    worst_ann = float(ann_values.min())
    stability_penalty = (
        ann_std * VALIDATION_STABILITY_WEIGHTS["ann_std"]
        + ann_spread * VALIDATION_STABILITY_WEIGHTS["ann_spread"]
        + max(0.0, -worst_ann) * VALIDATION_STABILITY_WEIGHTS["negative_worst_ann"]
    )
    segment_summary = "; ".join(
        f"{item['name']}[{item['range'][0]}:{item['range'][1]}]"
        f":ann={float(item['with_cost_ann_return']):.6f},ir={float(item['with_cost_ir']):.6f},mdd={float(item['with_cost_mdd']):.6f}"
        for item in segment_results
    )

    return {
        "score_mode": VALIDATION_SCORE_MODE,
        "segment_count": len(segment_results),
        "segment_mean_score": segment_mean_score,
        "stable_penalty": stability_penalty,
        "segment_ann_std": ann_std,
        "segment_ann_spread": ann_spread,
        "segment_worst_ann_return": worst_ann,
        "segment_details": segment_summary,
        "score": segment_mean_score - stability_penalty,
    }


def _evaluate_signal(
    model_trial_name: str,
    strategy_trial: dict,
    pred: pd.DataFrame,
    label: pd.DataFrame,
    *,
    window_key: str,
    eval_segment: str,
    backtest_segment: str,
    recorder_name: str,
    eval_range: list[str] | tuple[str, str] | None = None,
    backtest_range: list[str] | tuple[str, str] | None = None,
) -> dict:
    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

    task = build_base_task(
        topk=TOPK,
        n_drop=strategy_trial["n_drop"],
        window_key=window_key,
        model_key="tra_lstm_base",
        handler_class="Alpha360",
        handler_module_path="qlib.contrib.data.handler",
        account=ACCOUNT,
        risk_degree=strategy_trial["risk_degree"],
        benchmark=BENCHMARK,
        instruments=INSTRUMENTS,
        handler_kwargs_extra=H5_LABEL,
        strategy_class=strategy_trial["strategy_class"],
        strategy_module_path=strategy_trial["strategy_module_path"],
        strategy_kwargs_extra=strategy_trial["kwargs_extra"],
        eval_segment=eval_segment,
        backtest_segment=backtest_segment,
    )
    port_record = task["record"][2]
    if eval_range is not None:
        task["dataset"]["kwargs"]["segments"]["test"] = list(eval_range)
        pred = _slice_signal_to_range(pred, eval_range)
        label = _slice_signal_to_range(label, eval_range)
    if backtest_range is not None:
        port_record["kwargs"]["config"]["backtest"]["start_time"] = backtest_range[0]
        port_record["kwargs"]["config"]["backtest"]["end_time"] = backtest_range[1]

    with R.start(experiment_name="tra_global_validation_strategy_eval", recorder_name=recorder_name):
        rec = R.get_recorder()
        rec.save_objects(**{"pred.pkl": pred, "label.pkl": label})
        SigAnaRecord(recorder=rec, ana_long_short=False, ann_scaler=252).generate()
        PortAnaRecord(recorder=rec, **port_record["kwargs"]).generate()
        metrics = rec.list_metrics()

    return {
        "recorder_id": rec.id,
        "IC": metrics.get("IC"),
        "Rank IC": metrics.get("Rank IC"),
        "with_cost_ann_return": _metric_value(metrics, rec.id, "1day.excess_return_with_cost.annualized_return"),
        "with_cost_ir": _metric_value(metrics, rec.id, "1day.excess_return_with_cost.information_ratio"),
        "with_cost_mdd": _metric_value(metrics, rec.id, "1day.excess_return_with_cost.max_drawdown"),
    }


def _strategy_trial_from_row(row: dict) -> dict:
    return {
        "trial_name": row["strategy_trial"],
        "strategy_class": row["strategy_class"],
        "strategy_module_path": row["strategy_module_path"],
        "n_drop": int(row["n_drop"]),
        "risk_degree": float(row["risk_degree"]),
        "kwargs_extra": _parse_optional_literal(row["kwargs_extra"]) or {},
    }


def _write_best_validation(
    best_row: dict,
    *,
    output_path: Path,
    window_key: str,
    seed: int,
    deterministic_runtime: bool,
):
    window = WINDOWS[window_key]
    lines = [
        f"window_key={window_key}",
        f"train={window['train']}",
        f"valid={window['valid']}",
        f"test={window['test']}",
        f"seed={seed}",
        f"deterministic_runtime={deterministic_runtime}",
        f"dataset_kwargs_extra={best_row['dataset_kwargs_extra']}",
        f"model_kwargs_override={best_row['model_kwargs_override']}",
        f"model_trial={best_row['model_trial']}",
        f"model_key={best_row['model_key']}",
        f"step={best_row['step']}",
        f"handler_kwargs_extra={best_row['handler_kwargs_extra']}",
        f"strategy_trial={best_row['strategy_trial']}",
        f"strategy_class={best_row['strategy_class']}",
        f"strategy_module_path={best_row['strategy_module_path']}",
        f"n_drop={best_row['n_drop']}",
        f"risk_degree={best_row['risk_degree']}",
        f"kwargs_extra={best_row['kwargs_extra']}",
        f"validation_score_mode={best_row['score_mode']}",
        f"validation_base_score={best_row['base_score']}",
        f"validation_segment_mean_score={best_row['segment_mean_score']}",
        f"validation_stable_penalty={best_row['stable_penalty']}",
        f"validation_segment_count={best_row['segment_count']}",
        f"validation_segment_ann_std={best_row['segment_ann_std']}",
        f"validation_segment_ann_spread={best_row['segment_ann_spread']}",
        f"validation_segment_worst_ann_return={best_row['segment_worst_ann_return']}",
        f"validation_segment_details={best_row['segment_details']}",
        f"validation_IC={best_row['IC']}",
        f"validation_Rank_IC={best_row['Rank IC']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"validation_score={best_row['score']}",
        f"validation_cache_path={best_row['cache_path']}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_test(
    best_row: dict,
    test_result: dict,
    *,
    output_path: Path,
    window_key: str,
    seed: int,
    deterministic_runtime: bool,
):
    window = WINDOWS[window_key]
    lines = [
        f"window_key={window_key}",
        f"train={window['train']}",
        f"valid={window['valid']}",
        f"test={window['test']}",
        f"seed={seed}",
        f"deterministic_runtime={deterministic_runtime}",
        f"selection_source=validation_only",
        f"dataset_kwargs_extra={best_row['dataset_kwargs_extra']}",
        f"model_kwargs_override={best_row['model_kwargs_override']}",
        f"model_trial={best_row['model_trial']}",
        f"model_key={best_row['model_key']}",
        f"step={best_row['step']}",
        f"handler_kwargs_extra={best_row['handler_kwargs_extra']}",
        f"strategy_trial={best_row['strategy_trial']}",
        f"strategy_class={best_row['strategy_class']}",
        f"strategy_module_path={best_row['strategy_module_path']}",
        f"n_drop={best_row['n_drop']}",
        f"risk_degree={best_row['risk_degree']}",
        f"kwargs_extra={best_row['kwargs_extra']}",
        f"validation_score_mode={best_row['score_mode']}",
        f"validation_score={best_row['score']}",
        f"validation_base_score={best_row['base_score']}",
        f"validation_segment_mean_score={best_row['segment_mean_score']}",
        f"validation_stable_penalty={best_row['stable_penalty']}",
        f"validation_segment_count={best_row['segment_count']}",
        f"validation_segment_ann_std={best_row['segment_ann_std']}",
        f"validation_segment_ann_spread={best_row['segment_ann_spread']}",
        f"validation_segment_worst_ann_return={best_row['segment_worst_ann_return']}",
        f"validation_segment_details={best_row['segment_details']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"test_result_path={test_result['result_path']}",
        f"test_cache_path={test_result['cache_path']}",
        f"test_recorder_id={test_result['recorder_id']}",
        f"test_IC={test_result['IC']}",
        f"test_Rank_IC={test_result['Rank IC']}",
        f"test_with_cost_ann_return={test_result['with_cost_ann_return']}",
        f"test_with_cost_ir={test_result['with_cost_ir']}",
        f"test_with_cost_mdd={test_result['with_cost_mdd']}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_seeds(seed_text: str) -> list[int]:
    seeds = [int(item.strip()) for item in seed_text.split(",") if item.strip()]
    if len(seeds) < 2:
        raise ValueError("dual-seed study requires at least two seeds")
    return seeds


def _gpu_map_for_seeds(seeds: list[int], gpu_slots: str) -> dict[int, str]:
    slots = [slot.strip() for slot in gpu_slots.split(",") if slot.strip()]
    if not slots:
        slots = ["0"]
    return {seed: slots[index % len(slots)] for index, seed in enumerate(seeds)}


def _seed_run_suffix(result_suffix: str, seed: int, segment: str) -> str:
    parts = ["dualseed", f"s{seed}", segment]
    if result_suffix:
        parts.append(result_suffix)
    return "_".join(parts)


def _dual_seed_path(path: Path, result_suffix: str) -> Path:
    return _suffix_path(path, result_suffix)


def _meminfo_gb() -> dict[str, float]:
    values = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw_value = line.split(":", 1)
        values[key] = float(raw_value.strip().split()[0]) / 1024.0 / 1024.0
    swap_total = values.get("SwapTotal", 0.0)
    swap_free = values.get("SwapFree", 0.0)
    return {
        "mem_total_gb": values.get("MemTotal", 0.0),
        "mem_available_gb": values.get("MemAvailable", 0.0),
        "mem_used_gb": values.get("MemTotal", 0.0) - values.get("MemAvailable", 0.0),
        "swap_total_gb": swap_total,
        "swap_free_gb": swap_free,
        "swap_used_gb": max(0.0, swap_total - swap_free),
    }


def _descendant_rss_gb(processes: dict[int, mp.Process]) -> float:
    root_pids = {process.pid for process in processes.values() if process.pid is not None}
    if not root_pids:
        return 0.0
    completed = subprocess.run(["ps", "-eo", "pid=,ppid=,rss="], check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return 0.0
    children_by_parent: dict[int, list[int]] = {}
    rss_by_pid: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        pid, ppid, rss_kb = (int(parts[0]), int(parts[1]), int(parts[2]))
        children_by_parent.setdefault(ppid, []).append(pid)
        rss_by_pid[pid] = rss_kb
    stack = list(root_pids)
    all_pids = set(root_pids)
    while stack:
        pid = stack.pop()
        for child_pid in children_by_parent.get(pid, []):
            if child_pid not in all_pids:
                all_pids.add(child_pid)
                stack.append(child_pid)
    return sum(rss_by_pid.get(pid, 0) for pid in all_pids) / 1024.0 / 1024.0


def _gpu_resource_snapshot(gpu_slots: str) -> list[dict[str, float | int]]:
    requested_slots = {int(slot.strip()) for slot in gpu_slots.split(",") if slot.strip()}
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return []
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        gpu_index = int(parts[0])
        if requested_slots and gpu_index not in requested_slots:
            continue
        used_gb = float(parts[1]) / 1024.0
        total_gb = float(parts[2]) / 1024.0
        rows.append({"index": gpu_index, "used_gb": used_gb, "total_gb": total_gb, "free_gb": total_gb - used_gb})
    return rows


def _resource_snapshot(gpu_slots: str, processes: dict[int, mp.Process]) -> dict[str, object]:
    snapshot: dict[str, object] = _meminfo_gb()
    snapshot["child_rss_gb"] = _descendant_rss_gb(processes)
    snapshot["gpus"] = _gpu_resource_snapshot(gpu_slots)
    return snapshot


def _append_resource_snapshot(
    path: Path,
    *,
    phase: str,
    running_seeds: list[int],
    snapshot: dict[str, object],
    baseline_swap_used_gb: float,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    gpus = snapshot.get("gpus", [])
    gpu_summary = ";".join(
        f"gpu{gpu['index']}:used={float(gpu['used_gb']):.3f},free={float(gpu['free_gb']):.3f},total={float(gpu['total_gb']):.3f}"
        for gpu in gpus
    )
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        if write_header:
            writer.writerow([
                "timestamp",
                "phase",
                "running_seeds",
                "mem_used_gb",
                "mem_available_gb",
                "swap_used_gb",
                "swap_extra_gb",
                "child_rss_gb",
                "gpu_summary",
            ])
        writer.writerow([
            f"{time.time():.3f}",
            phase,
            ",".join(str(seed) for seed in running_seeds),
            f"{float(snapshot['mem_used_gb']):.3f}",
            f"{float(snapshot['mem_available_gb']):.3f}",
            f"{float(snapshot['swap_used_gb']):.3f}",
            f"{float(snapshot['swap_used_gb']) - baseline_swap_used_gb:.3f}",
            f"{float(snapshot['child_rss_gb']):.3f}",
            gpu_summary,
        ])


def _resource_limit_error(
    *,
    snapshot: dict[str, object],
    baseline_swap_used_gb: float,
    min_available_ram_gb: float,
    max_extra_swap_gb: float,
) -> str | None:
    available_ram_gb = float(snapshot["mem_available_gb"])
    swap_total_gb = float(snapshot["swap_total_gb"])
    swap_free_gb = float(snapshot["swap_free_gb"])
    swap_used_gb = float(snapshot["swap_used_gb"])
    extra_swap_gb = swap_used_gb - baseline_swap_used_gb
    combined_available_gb = available_ram_gb + swap_free_gb
    if extra_swap_gb > max_extra_swap_gb:
        return f"extra swap usage reached {extra_swap_gb:.2f} GiB, above limit {max_extra_swap_gb:.2f} GiB"
    if available_ram_gb < 1.5 and combined_available_gb < min_available_ram_gb:
        return (
            f"combined RAM+swap headroom dropped to {combined_available_gb:.2f} GiB "
            f"with MemAvailable {available_ram_gb:.2f} GiB"
        )
    if swap_total_gb > 0 and swap_used_gb / swap_total_gb > 0.90 and combined_available_gb < min_available_ram_gb * 2:
        return (
            f"RAM+swap pressure is too high: swap {swap_used_gb:.2f}/{swap_total_gb:.2f} GiB, "
            f"MemAvailable {available_ram_gb:.2f} GiB, combined headroom {combined_available_gb:.2f} GiB"
        )
    for gpu in snapshot.get("gpus", []):
        if float(gpu["free_gb"]) < 0.5:
            return f"GPU {gpu['index']} free memory dropped to {float(gpu['free_gb']):.2f} GiB"
    return None


def _terminate_processes(processes: dict[int, mp.Process]):
    for process in processes.values():
        if process.is_alive():
            process.terminate()
    for process in processes.values():
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def _drain_dual_seed_queue(result_queue, results: dict[int, dict], errors: list[str]) -> int:
    drained = 0
    while True:
        try:
            payload = result_queue.get_nowait()
        except queue_module.Empty:
            break
        drained += 1
        if payload["ok"]:
            results[payload["seed"]] = payload["result"]
        else:
            errors.append(f"seed {payload['seed']}: {payload['error']}")
    return drained


def _dual_seed_train_worker(
    result_queue,
    model_family: str,
    model_trial: dict,
    *,
    seed: int,
    gpu_slot: str,
    result_suffix: str,
    deterministic_runtime: bool,
    window_key: str,
    dataset_kwargs_extra: dict,
    eval_segment: str,
    backtest_segment: str,
    rolling_task_limit: int | None,
):
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_slot)
        os.environ["QLIB_TRA_PHYSICAL_GPU_SLOT"] = str(gpu_slot)
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
        result = _run_family_trial(
            model_family,
            model_trial,
            seed=seed,
            gpu_slots="0",
            result_suffix=result_suffix,
            deterministic_runtime=deterministic_runtime,
            window_key=window_key,
            n_drop=1,
            risk_degree=0.70,
            dataset_kwargs_extra=dataset_kwargs_extra,
            eval_segment=eval_segment,
            backtest_segment=backtest_segment,
            rolling_task_limit=rolling_task_limit,
            parallel_mode="trainer",
        )
        result_queue.put({"seed": seed, "ok": True, "result": result})
    except Exception as exc:
        result_queue.put({"seed": seed, "ok": False, "error": repr(exc)})


def _run_dual_seed_base_trials(
    model_family: str,
    model_trial: dict,
    *,
    seeds: list[int],
    gpu_slots: str,
    result_suffix: str,
    deterministic_runtime: bool,
    window_key: str,
    dataset_kwargs_extra: dict,
    eval_segment: str,
    backtest_segment: str,
    rolling_task_limit: int | None,
    max_concurrent_seeds: int,
    min_available_ram_gb: float,
    max_extra_swap_gb: float,
    resource_poll_seconds: float,
    resource_log_path: Path,
    baseline_swap_used_gb: float | None = None,
) -> dict[int, dict]:
    if model_family != "tra":
        raise ValueError("dual-seed shared training currently supports only model_family='tra'")

    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)
    step = model_trial["step"]
    model_key = model_trial["model_key"]
    handler_class = "Alpha360"
    risk_degree = 0.70
    risk_tag = str(risk_degree).replace(".", "")
    handler_uri = _prepare_handler_cache(
        window_key=window_key,
        handler_class=handler_class,
        handler_module_path="qlib.contrib.data.handler",
        instruments=INSTRUMENTS,
        handler_kwargs_extra=model_trial["handler_kwargs_extra"],
    )
    _preload_tra_fork_handlers([{"dataset": {"kwargs": {"handler": handler_uri}}}])
    print(f"preloaded shared TRA handler before seed x rolling-task fork for {eval_segment}: {handler_uri}")

    seed_task_info: dict[int, dict] = {}
    for seed in seeds:
        seed_suffix = _seed_run_suffix(result_suffix, seed, eval_segment)
        model_kwargs = _model_kwargs_for_seed(seed, model_family, model_trial)
        base_task = build_base_task(
            topk=TOPK,
            n_drop=1,
            window_key=window_key,
            model_key=model_key,
            handler_class=handler_class,
            handler_module_path="qlib.contrib.data.handler",
            account=ACCOUNT,
            risk_degree=risk_degree,
            benchmark=BENCHMARK,
            instruments=INSTRUMENTS,
            model_kwargs_override=model_kwargs,
            handler_kwargs_extra=model_trial["handler_kwargs_extra"],
            dataset_kwargs_extra=dataset_kwargs_extra,
            eval_segment=eval_segment,
            backtest_segment=backtest_segment,
            handler_override=handler_uri,
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
        rolling_models_exp = (
            f"rolling_models_tra_latest_{window_key}_{model_key}_{handler_class}_s{step}"
            f"_k{TOPK}_d1_a{ACCOUNT}_r{risk_tag}_{seed_suffix}"
        )
        final_exp = (
            f"rolling_eval_tra_latest_{window_key}_{model_key}_{handler_class}_s{step}"
            f"_k{TOPK}_d1_a{ACCOUNT}_r{risk_tag}_{seed_suffix}"
        )
        seed_task_info[seed] = {
            "seed_suffix": seed_suffix,
            "model_kwargs": model_kwargs,
            "task_list": task_list,
            "total_task_count": total_task_count,
            "rolling_models_exp": rolling_models_exp,
            "final_exp": final_exp,
        }
        R.get_exp(experiment_name=rolling_models_exp)
        print(
            f"prepared dual-seed rolling tasks seed={seed} eval_segment={eval_segment} "
            f"tasks={len(task_list)}/{total_task_count} exp={rolling_models_exp}"
        )

    ctx = mp.get_context("fork")
    slots = _normalize_gpu_slots(gpu_slots)
    if not slots:
        slots = [0]
    gpu_capacity = max(1, int(max_concurrent_seeds))
    # Round-robin capacities so an initial worker wave reaches every GPU.
    available_slots = slots * gpu_capacity
    max_task_count = max(len(seed_task_info[seed]["task_list"]) for seed in seeds)
    pending = [
        {"seed": seed, "task_idx": task_idx, "task": seed_task_info[seed]["task_list"][task_idx], "experiment_name": seed_task_info[seed]["rolling_models_exp"]}
        for task_idx in range(max_task_count)
        for seed in seeds
        if task_idx < len(seed_task_info[seed]["task_list"])
    ]
    running: dict[int, mp.Process] = {}
    results: dict[int, dict] = {}
    errors: list[str] = []
    completed_tasks: set[tuple[int, int]] = set()

    baseline_snapshot = _resource_snapshot(gpu_slots, running)
    if baseline_swap_used_gb is None:
        baseline_swap_used_gb = float(baseline_snapshot["swap_used_gb"])
    _append_resource_snapshot(
        resource_log_path,
        phase=f"{eval_segment}:baseline",
        running_seeds=[],
        snapshot=baseline_snapshot,
        baseline_swap_used_gb=baseline_swap_used_gb,
    )

    next_sample_at = time.monotonic()
    while pending or running:
        if errors:
            break
        while pending and available_slots:
            prelaunch_snapshot = _resource_snapshot(gpu_slots, running)
            _append_resource_snapshot(
                resource_log_path,
                phase=f"{eval_segment}:prelaunch",
                running_seeds=sorted({task_key[0] for task_key in running}),
                snapshot=prelaunch_snapshot,
                baseline_swap_used_gb=baseline_swap_used_gb,
            )
            limit_error = _resource_limit_error(
                snapshot=prelaunch_snapshot,
                baseline_swap_used_gb=baseline_swap_used_gb,
                min_available_ram_gb=min_available_ram_gb,
                max_extra_swap_gb=max_extra_swap_gb,
            )
            if limit_error is not None:
                errors.append(f"resource guard blocked {eval_segment} launch: {limit_error}")
                break
            task_item = pending.pop(0)
            seed = int(task_item["seed"])
            task_idx = int(task_item["task_idx"])
            gpu_slot = available_slots.pop(0)
            process = ctx.Process(
                target=_run_tra_task_worker,
                args=(task_item["task"], task_item["experiment_name"], PROVIDER_URI, gpu_slot, deterministic_runtime, 1),
            )
            process.start()
            process._dual_seed_gpu_slot = gpu_slot
            running[(seed, task_idx)] = process
            print(
                f"launch dual-seed rolling {eval_segment} seed={seed} task={task_idx + 1}/"
                f"{len(seed_task_info[seed]['task_list'])} physical_gpu={gpu_slot} pid={process.pid}"
            )

        for task_key, process in list(running.items()):
            if process.is_alive():
                continue
            process.join(timeout=1)
            if process.exitcode not in (0, None):
                errors.append(f"seed {task_key[0]} task {task_key[1] + 1} process {process.pid} exitcode={process.exitcode}")
            else:
                completed_tasks.add(task_key)
            running.pop(task_key, None)
            try:
                gpu_slot = getattr(process, "_dual_seed_gpu_slot")
            except AttributeError:
                gpu_slot = None
            if gpu_slot is not None:
                available_slots.append(gpu_slot)

        if errors:
            break
        current_time = time.monotonic()
        if running and current_time >= next_sample_at:
            snapshot = _resource_snapshot(gpu_slots, running)
            _append_resource_snapshot(
                resource_log_path,
                phase=f"{eval_segment}:monitor",
                running_seeds=sorted({task_key[0] for task_key in running}),
                snapshot=snapshot,
                baseline_swap_used_gb=baseline_swap_used_gb,
            )
            limit_error = _resource_limit_error(
                snapshot=snapshot,
                baseline_swap_used_gb=baseline_swap_used_gb,
                min_available_ram_gb=min_available_ram_gb,
                max_extra_swap_gb=max_extra_swap_gb,
            )
            if limit_error is not None:
                errors.append(f"resource guard triggered during {eval_segment}: {limit_error}")
                _terminate_processes(running)
                running.clear()
                break
            next_sample_at = current_time + max(1.0, resource_poll_seconds)

        if running:
            time.sleep(1)

    final_snapshot = _resource_snapshot(gpu_slots, running)
    _append_resource_snapshot(
        resource_log_path,
        phase=f"{eval_segment}:final",
        running_seeds=[],
        snapshot=final_snapshot,
        baseline_swap_used_gb=baseline_swap_used_gb,
    )
    if errors:
        _terminate_processes(running)
        raise RuntimeError("dual-seed base training failed: " + "; ".join(errors))
    expected_tasks = {
        (seed, task_idx)
        for seed in seeds
        for task_idx in range(len(seed_task_info[seed]["task_list"]))
    }
    missing_tasks = sorted(expected_tasks - completed_tasks)
    if missing_tasks:
        raise RuntimeError(f"dual-seed base training missing rolling tasks {missing_tasks}")

    for seed in seeds:
        info = seed_task_info[seed]
        collector = RecorderCollector(
            experiment=info["rolling_models_exp"],
            artifacts_key=["pred", "label"],
            process_list=[RollingEnsemble()],
            artifacts_path={"pred": "pred.pkl", "label": "label.pkl"},
        )
        res = collector()
        with R.start(experiment_name=info["final_exp"], recorder_name=f"rolling_step_{step}"):
            rec = R.get_recorder()
            rec.save_objects(**{"pred.pkl": res["pred"], "label.pkl": res["label"]})
            for record in build_base_task(
                topk=TOPK,
                n_drop=1,
                window_key=window_key,
                model_key=model_key,
                handler_class=handler_class,
                handler_module_path="qlib.contrib.data.handler",
                account=ACCOUNT,
                risk_degree=risk_degree,
                benchmark=BENCHMARK,
                instruments=INSTRUMENTS,
                model_kwargs_override=info["model_kwargs"],
                handler_kwargs_extra=model_trial["handler_kwargs_extra"],
                dataset_kwargs_extra=dataset_kwargs_extra,
                eval_segment=eval_segment,
                backtest_segment=backtest_segment,
                handler_override=handler_uri,
            )["record"]:
                if record["class"] == "SignalRecord":
                    continue
                if record["class"] == "SigAnaRecord":
                    SigAnaRecord(recorder=rec, **record["kwargs"]).generate()
                elif record["class"] == "PortAnaRecord":
                    PortAnaRecord(recorder=rec, **record["kwargs"]).generate()
            metrics = rec.list_metrics()

        TRA_CACHE_DIR.mkdir(exist_ok=True)
        cache_path = TRA_CACHE_DIR / (
            f"rolling_cache_tra_{window_key}_{model_key}_{handler_class}_step{step}"
            f"_k{TOPK}_d1_a{ACCOUNT}_r{risk_tag}_{info['seed_suffix']}.pkl"
        )
        pd.to_pickle({"pred": res["pred"], "label": res["label"]}, cache_path)
        result_path = BASE_DIR / (
            f"rolling_result_tra_{window_key}_{model_key}_{handler_class}_step{step}"
            f"_k{TOPK}_d1_a{ACCOUNT}_r{risk_tag}_{info['seed_suffix']}.txt"
        )
        with_cost_ann_return = _metric_value(metrics, rec.id, "1day.excess_return_with_cost.annualized_return")
        with_cost_ir = _metric_value(metrics, rec.id, "1day.excess_return_with_cost.information_ratio")
        with_cost_mdd = _metric_value(metrics, rec.id, "1day.excess_return_with_cost.max_drawdown")
        result_path.write_text(
            "\n".join(
                [
                    f"rolling_models_exp={info['rolling_models_exp']}",
                    f"final_exp={info['final_exp']}",
                    f"recorder_id={rec.id}",
                    f"window_key={window_key}",
                    f"model_key={model_key}",
                    f"eval_segment={eval_segment}",
                    f"backtest_segment={backtest_segment}",
                    f"rolling_task_limit={rolling_task_limit}",
                    f"rolling_task_count={len(info['task_list'])}",
                    f"rolling_task_total={info['total_task_count']}",
                    f"handler_kwargs_extra={model_trial['handler_kwargs_extra']}",
                    f"dataset_kwargs_extra={dataset_kwargs_extra}",
                    f"model_kwargs_override={info['model_kwargs']}",
                    f"deterministic_runtime={deterministic_runtime}",
                    f"scheduler=seed_x_rolling_task_top_level_fork",
                    f"jobs_per_gpu={gpu_capacity}",
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
        results[seed] = {
            "result_path": str(result_path),
            "cache_path": str(cache_path),
            "recorder_id": rec.id,
            "IC": metrics.get("IC"),
            "Rank IC": metrics.get("Rank IC"),
            "with_cost_ann_return": with_cost_ann_return,
            "with_cost_ir": with_cost_ir,
            "with_cost_mdd": with_cost_mdd,
            "eval_segment": eval_segment,
            "backtest_segment": backtest_segment,
            "rolling_task_limit": rolling_task_limit,
            "rolling_task_count": len(info["task_list"]),
            "rolling_task_total": info["total_task_count"],
        }

    return results


def _attach_metrics(row: dict, prefix: str, result: dict):
    for key in ["recorder_id", "IC", "Rank IC", "with_cost_ann_return", "with_cost_ir", "with_cost_mdd"]:
        if key in result:
            row[f"{prefix}_{key}"] = result[key]
    for key in [
        "score",
        "base_score",
        "segment_mean_score",
        "stable_penalty",
        "segment_ann_std",
        "segment_ann_spread",
        "segment_worst_ann_return",
        "segment_details",
    ]:
        if key in result:
            row[f"{prefix}_{key}"] = result[key]


def _build_dual_seed_grid(
    model_trial: dict,
    *,
    seeds: list[int],
    valid_base_runs: dict[int, dict],
    test_base_runs: dict[int, dict],
    window_key: str,
    output_path: Path,
) -> list[dict]:
    valid_signals = {seed: _load_signal(valid_base_runs[seed]["cache_path"]) for seed in seeds}
    test_signals = {seed: _load_signal(test_base_runs[seed]["cache_path"]) for seed in seeds}
    rows: list[dict] = []

    for strategy_index, strategy_trial in enumerate(STRATEGY_TRIALS, start=1):
        for seed in seeds:
            row = {
                "seed": seed,
                "model_trial": model_trial["trial_name"],
                "model_key": model_trial["model_key"],
                "strategy_trial": strategy_trial["trial_name"],
                "strategy_class": strategy_trial["strategy_class"],
                "strategy_module_path": strategy_trial["strategy_module_path"],
                "n_drop": strategy_trial["n_drop"],
                "risk_degree": strategy_trial["risk_degree"],
                "kwargs_extra": str(strategy_trial["kwargs_extra"]),
                "score_mode": VALIDATION_SCORE_MODE,
                "valid_cache_path": valid_base_runs[seed]["cache_path"],
                "test_cache_path": test_base_runs[seed]["cache_path"],
            }
            valid_pred, valid_label = valid_signals[seed]
            validation_result = _evaluate_strategy(
                f"{model_trial['trial_name']}__s{seed}",
                strategy_trial,
                valid_pred,
                valid_label,
                window_key=window_key,
            )
            _attach_metrics(row, "validation", validation_result)

            test_pred, test_label = test_signals[seed]
            test_result = _evaluate_signal(
                f"{model_trial['trial_name']}__s{seed}",
                strategy_trial,
                test_pred,
                test_label,
                window_key=window_key,
                eval_segment="test",
                backtest_segment="test",
                recorder_name=f"{model_trial['trial_name']}__{strategy_trial['trial_name']}__s{seed}__test",
            )
            _attach_metrics(row, "test", test_result)
            rows.append(row)

        pd.DataFrame(rows).to_csv(output_path, index=False)
        if strategy_index == 1 or strategy_index % 10 == 0 or strategy_index == len(STRATEGY_TRIALS):
            print(f"dual-seed grid progress: {strategy_index}/{len(STRATEGY_TRIALS)} strategies, rows={len(rows)}")
    return rows


def _write_dual_seed_summary(
    rows: list[dict],
    *,
    seeds: list[int],
    valid_base_runs: dict[int, dict],
    test_base_runs: dict[int, dict],
    output_path: Path,
):
    lines = [
        f"seeds={seeds}",
        f"seed_count={len(seeds)}",
        f"strategy_count={len(STRATEGY_TRIALS) if rows else 0}",
        f"row_count={len(rows)}",
        f"valid_base_runs={valid_base_runs}",
        f"test_base_runs={test_base_runs}",
        "top_validation_score_min=",
    ]
    if not rows:
        lines.extend(["top_test_ann_min=", "top_joint_rank_score="])
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    df = pd.DataFrame(rows)
    grouped = (
        df.groupby("strategy_trial", as_index=False)
        .agg(
            validation_score_min=("validation_score", "min"),
            validation_score_mean=("validation_score", "mean"),
            validation_ann_min=("validation_with_cost_ann_return", "min"),
            validation_ann_mean=("validation_with_cost_ann_return", "mean"),
            test_ann_min=("test_with_cost_ann_return", "min"),
            test_ann_mean=("test_with_cost_ann_return", "mean"),
            test_ir_min=("test_with_cost_ir", "min"),
            test_mdd_min=("test_with_cost_mdd", "min"),
        )
        .assign(joint_rank_score=lambda frame: frame["validation_score_min"] + frame["test_ann_min"] * 1000.0)
    )
    robust_valid = grouped.sort_values(["validation_score_min", "test_ann_min"], ascending=[False, False]).head(10)
    robust_test = grouped.sort_values(["test_ann_min", "validation_score_min"], ascending=[False, False]).head(10)
    joint_best = grouped.sort_values("joint_rank_score", ascending=False).head(10)

    lines.extend(
        f"{row.strategy_trial}|validation_score_min={row.validation_score_min}|test_ann_min={row.test_ann_min}|test_ann_mean={row.test_ann_mean}"
        for row in robust_valid.itertuples(index=False)
    )
    lines.append("top_test_ann_min=")
    lines.extend(
        f"{row.strategy_trial}|test_ann_min={row.test_ann_min}|test_ann_mean={row.test_ann_mean}|validation_score_min={row.validation_score_min}"
        for row in robust_test.itertuples(index=False)
    )
    lines.append("top_joint_rank_score=")
    lines.extend(
        f"{row.strategy_trial}|joint_rank_score={row.joint_rank_score}|validation_score_min={row.validation_score_min}|test_ann_min={row.test_ann_min}"
        for row in joint_best.itertuples(index=False)
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dual_seed_study(
    *,
    seeds: list[int],
    gpu_slots: str,
    result_suffix: str,
    dual_seed_segment: str,
    deterministic_runtime: bool,
    window_key: str,
    model_family: str,
    batch_size: int,
    smoke_only: bool,
    rolling_task_limit: int | None,
    max_concurrent_seeds: int,
    min_available_ram_gb: float,
    max_extra_swap_gb: float,
    resource_poll_seconds: float,
    smoke_epochs: int,
    base_only: bool,
):
    if model_family != "tra":
        raise ValueError("dual-seed study currently supports only model_family='tra'")
    if dual_seed_segment not in {"both", "valid", "test"}:
        raise ValueError(f"unknown dual-seed segment: {dual_seed_segment}")
    if dual_seed_segment != "both" and not base_only:
        raise ValueError("--dual-seed-segment valid/test is only supported with --dual-seed-base-only")
    model_trials = _model_trials_for_family(model_family)
    if len(model_trials) != 1:
        raise RuntimeError("dual-seed study expects exactly one model trial")
    model_trial = copy.deepcopy(model_trials[0])
    if smoke_only:
        model_trial["model_kwargs_override"] = _deep_merge(
            _base_model_kwargs_override_for_family(model_family),
            {
                "n_epochs": max(1, int(smoke_epochs)),
                "early_stop": max(1, min(2, int(smoke_epochs))),
                "eval_freq": 1,
            },
        )
    dataset_kwargs_extra = {"batch_size": batch_size}
    resource_log_path = _dual_seed_path(DUAL_SEED_RESOURCE_LOG_PATH, result_suffix)
    if resource_log_path.exists():
        resource_log_path.unlink()
    startup_snapshot = _resource_snapshot(gpu_slots, {})
    baseline_swap_used_gb = float(startup_snapshot["swap_used_gb"])
    startup_limit_error = _resource_limit_error(
        snapshot=startup_snapshot,
        baseline_swap_used_gb=baseline_swap_used_gb,
        min_available_ram_gb=min_available_ram_gb,
        max_extra_swap_gb=max_extra_swap_gb,
    )
    if startup_limit_error is not None:
        raise RuntimeError(f"dual-seed study refused to start: {startup_limit_error}")

    print(
        f"dual-seed TRA study seeds={seeds} gpu_slots={gpu_slots} batch_size={batch_size} "
        f"window_key={window_key} result_suffix={result_suffix or '<default>'} "
        f"dual_seed_segment={dual_seed_segment} "
        f"rolling_task_limit={rolling_task_limit} smoke_only={smoke_only} "
        f"smoke_epochs={smoke_epochs if smoke_only else '<formal>'} "
        f"jobs_per_gpu={max_concurrent_seeds} min_available_ram_gb={min_available_ram_gb} "
        f"max_extra_swap_gb={max_extra_swap_gb} resource_log_path={resource_log_path}"
    )

    valid_base_runs = {}
    if dual_seed_segment in {"both", "valid"}:
        valid_base_runs = _run_dual_seed_base_trials(
            model_family,
            model_trial,
            seeds=seeds,
            gpu_slots=gpu_slots,
            result_suffix=result_suffix,
            deterministic_runtime=deterministic_runtime,
            window_key=window_key,
            dataset_kwargs_extra=dataset_kwargs_extra,
            eval_segment="valid",
            backtest_segment="valid",
            rolling_task_limit=rolling_task_limit,
            max_concurrent_seeds=max_concurrent_seeds,
            min_available_ram_gb=min_available_ram_gb,
            max_extra_swap_gb=max_extra_swap_gb,
            resource_poll_seconds=resource_poll_seconds,
            resource_log_path=resource_log_path,
            baseline_swap_used_gb=baseline_swap_used_gb,
        )
    if smoke_only:
        print(f"dual-seed smoke completed: {valid_base_runs}")
        print(f"dual-seed resource log saved to {resource_log_path}")
        return

    test_base_runs = {}
    if dual_seed_segment in {"both", "test"}:
        test_base_runs = _run_dual_seed_base_trials(
            model_family,
            model_trial,
            seeds=seeds,
            gpu_slots=gpu_slots,
            result_suffix=result_suffix,
            deterministic_runtime=deterministic_runtime,
            window_key=window_key,
            dataset_kwargs_extra=dataset_kwargs_extra,
            eval_segment="test",
            backtest_segment="test",
            rolling_task_limit=rolling_task_limit,
            max_concurrent_seeds=max_concurrent_seeds,
            min_available_ram_gb=min_available_ram_gb,
            max_extra_swap_gb=max_extra_swap_gb,
            resource_poll_seconds=resource_poll_seconds,
            resource_log_path=resource_log_path,
            baseline_swap_used_gb=baseline_swap_used_gb,
        )

    summary_path = _dual_seed_path(DUAL_SEED_SUMMARY_PATH, result_suffix)
    if base_only:
        _write_dual_seed_summary(
            [],
            seeds=seeds,
            valid_base_runs=valid_base_runs,
            test_base_runs=test_base_runs,
            output_path=summary_path,
        )
        print(f"dual-seed base-only summary saved to {summary_path}")
        print(f"dual-seed resource log saved to {resource_log_path}")
        return

    grid_path = _dual_seed_path(DUAL_SEED_GRID_PATH, result_suffix)
    rows = _build_dual_seed_grid(
        model_trial,
        seeds=seeds,
        valid_base_runs=valid_base_runs,
        test_base_runs=test_base_runs,
        window_key=window_key,
        output_path=grid_path,
    )
    _write_dual_seed_summary(
        rows,
        seeds=seeds,
        valid_base_runs=valid_base_runs,
        test_base_runs=test_base_runs,
        output_path=summary_path,
    )
    print(f"dual-seed grid saved to {grid_path}")
    print(f"dual-seed summary saved to {summary_path}")
    print(f"dual-seed resource log saved to {resource_log_path}")


def main(
    skip_final_if_exists: bool = True,
    *,
    seed: int = 42,
    gpu_slots: str = GPU_SLOTS,
    result_suffix: str = "",
    deterministic_runtime: bool = True,
    window_key: str = WINDOW_KEY,
    run_final_test: bool = True,
    model_family: str = DEFAULT_MODEL_FAMILY,
    strategy_profile: str = STRATEGY_PROFILE_FULL,
    reuse_validation_cache: str = "",
    reuse_validation_result_path: str = "",
    reuse_test_cache: str = "",
    reuse_test_result_path: str = "",
):
    model_trials = _model_trials_for_family(model_family)
    strategy_trials = _strategy_trials_for_profile(strategy_profile)
    strategy_trial_names = {trial["trial_name"] for trial in strategy_trials}
    validation_result_path = _suffix_path(_validation_result_path_for_family(model_family), result_suffix)
    validation_best_path = _suffix_path(_validation_best_path_for_family(model_family), result_suffix)
    final_test_path = _suffix_path(_final_test_path_for_family(model_family), result_suffix)

    print(
        f"global tuning model_family={model_family} seed={seed} window_key={window_key} gpu_slots={gpu_slots} "
        f"deterministic_runtime={deterministic_runtime} result_suffix={result_suffix or '<default>'} "
        f"dataset_kwargs={DATASET_KWARGS} strategy_profile={strategy_profile} "
        f"sanity_strategy={SANITY_STRATEGY_TRIAL} #model_trials={len(model_trials)} "
        f"#strategy_trials={len(strategy_trials)}"
    )

    expected_model_kwargs_by_trial = _expected_model_kwargs_by_trial(seed, model_family)
    expected_dataset_kwargs = str(DATASET_KWARGS)

    row_map = _load_existing_rows(validation_result_path)
    if row_map:
        completed_count = sum(
            1
            for row in row_map.values()
            if row.get("status") == "success"
            and row.get("strategy_trial") in strategy_trial_names
            and _row_matches_selection_config(row)
            and row.get("model_trial") in expected_model_kwargs_by_trial
            and _row_matches_expected_config(
                row,
                expected_model_kwargs_by_trial[row["model_trial"]],
                expected_dataset_kwargs,
            )
        )
        print(f"resume existing validation file: {validation_result_path} success_rows={completed_count}/{len(row_map)}")

    model_result_by_trial = {}
    for row in row_map.values():
        if row.get("model_trial") not in expected_model_kwargs_by_trial:
            continue
        if not _row_matches_expected_config(
            row,
            expected_model_kwargs_by_trial[row["model_trial"]],
            expected_dataset_kwargs,
        ):
            continue
        result_path = row.get("result_path")
        cache_path = row.get("cache_path")
        if not result_path or not cache_path:
            continue
        if not Path(str(result_path)).exists() or not Path(str(cache_path)).exists():
            continue
        model_result_by_trial.setdefault(
            row["model_trial"],
            {"result_path": str(result_path), "cache_path": str(cache_path)},
        )

    explicit_reuse_validation = _resolve_reuse_artifact(reuse_validation_cache or None, reuse_validation_result_path or None)
    if explicit_reuse_validation is not None:
        if len(model_trials) != 1:
            raise RuntimeError("explicit validation cache reuse currently requires exactly one model trial")
        reused_trial_name = model_trials[0]["trial_name"]
        model_result_by_trial[reused_trial_name] = explicit_reuse_validation
        print(f"reuse validation base cache for {reused_trial_name}: {explicit_reuse_validation['cache_path']}")

    for model_trial in model_trials:
        success_keys = {
            key
            for key, row in row_map.items()
            if row.get("status") == "success"
            and key[0] == model_trial["trial_name"]
            and key[1] in strategy_trial_names
            and _row_matches_selection_config(row)
            and _row_matches_expected_config(
                row,
                expected_model_kwargs_by_trial[model_trial["trial_name"]],
                expected_dataset_kwargs,
            )
        }
        if len(success_keys) == len(strategy_trials):
            print(f"skip completed model trial {model_trial['trial_name']}: all strategy rows already successful")
            continue

        print(
            f"\n=== validation model trial {model_trial['trial_name']}: "
            f"model={model_trial['model_key']} step={model_trial['step']} ==="
        )
        model_result = model_result_by_trial.get(model_trial["trial_name"])
        if model_result is None:
            model_result = _run_family_trial(
                model_family,
                model_trial,
                seed=seed,
                gpu_slots=gpu_slots,
                result_suffix=_trial_suffix(model_trial["trial_name"], result_suffix),
                deterministic_runtime=deterministic_runtime,
                window_key=window_key,
                n_drop=1,
                risk_degree=0.70,
                eval_segment="valid",
                backtest_segment="valid",
            )
            model_result_by_trial[model_trial["trial_name"]] = {
                "result_path": model_result["result_path"],
                "cache_path": model_result["cache_path"],
            }
        else:
            print(f"reuse completed model signal for {model_trial['trial_name']}: {model_result['cache_path']}")

        pred, label = _load_signal(model_result["cache_path"])

        for strategy_trial in _ordered_strategy_trials(strategy_trials):
            row_key = _row_key(model_trial["trial_name"], strategy_trial["trial_name"])
            existing_row = row_map.get(row_key)
            if (
                existing_row
                and existing_row.get("status") == "success"
                and _row_matches_selection_config(existing_row)
                and _row_matches_expected_config(
                    existing_row,
                    expected_model_kwargs_by_trial[model_trial["trial_name"]],
                    expected_dataset_kwargs,
                )
            ):
                print(f"skip completed strategy eval {model_trial['trial_name']} + {strategy_trial['trial_name']}")
                continue

            row = {
                "model_trial": model_trial["trial_name"],
                "model_key": model_trial["model_key"],
                "step": model_trial["step"],
                "handler_kwargs_extra": str(model_trial["handler_kwargs_extra"]),
                "strategy_trial": strategy_trial["trial_name"],
                "strategy_class": strategy_trial["strategy_class"],
                "strategy_module_path": strategy_trial["strategy_module_path"],
                "n_drop": strategy_trial["n_drop"],
                "risk_degree": strategy_trial["risk_degree"],
                "kwargs_extra": str(strategy_trial["kwargs_extra"]),
                "model_kwargs_override": expected_model_kwargs_by_trial[model_trial["trial_name"]],
                "dataset_kwargs_extra": expected_dataset_kwargs,
                "seed": seed,
                "deterministic_runtime": deterministic_runtime,
                "score_mode": VALIDATION_SCORE_MODE,
                "status": "running",
                "result_path": model_result["result_path"],
                "cache_path": model_result["cache_path"],
            }
            try:
                eval_result = _evaluate_strategy(
                    model_trial["trial_name"],
                    strategy_trial,
                    pred,
                    label,
                    window_key=window_key,
                )
                row.update(eval_result)
                row["status"] = "success"
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
            row_map[row_key] = row
            _persist(list(row_map.values()), validation_result_path)
            if strategy_trial["trial_name"] == SANITY_STRATEGY_TRIAL:
                if row["status"] != "success":
                    raise RuntimeError(
                        f"base sanity strategy {SANITY_STRATEGY_TRIAL} failed before broader strategy search"
                    )
                if float(row["with_cost_ann_return"]) <= BASE_SANITY_MIN_ANN_RETURN:
                    raise RuntimeError(
                        f"base sanity strategy {SANITY_STRATEGY_TRIAL} ann_return={row['with_cost_ann_return']} "
                        f"did not exceed {BASE_SANITY_MIN_ANN_RETURN}; stop strategy search"
                    )

    rows = list(row_map.values())
    if not rows:
        raise RuntimeError("no validation rows produced")

    success_rows = [
        row
        for row in rows
        if row["status"] == "success"
        and row.get("strategy_trial") in strategy_trial_names
        and _row_matches_selection_config(row)
        and row.get("model_trial") in expected_model_kwargs_by_trial
        and _row_matches_expected_config(
            row,
            expected_model_kwargs_by_trial[row["model_trial"]],
            expected_dataset_kwargs,
        )
    ]
    if not success_rows:
        raise RuntimeError("all validation trials failed for the expected base_ref training configuration")

    best_row = max(success_rows, key=lambda item: item["score"])
    _write_best_validation(
        best_row,
        output_path=validation_best_path,
        window_key=window_key,
        seed=seed,
        deterministic_runtime=deterministic_runtime,
    )

    if not run_final_test:
        print("skip final test by request; validation winner has been written")
        return

    if skip_final_if_exists and final_test_path.exists():
        print(f"final test summary already exists at {final_test_path}, skip rerunning final test")
        return

    print(
        f"\n=== final test with validation-selected config: "
        f"{best_row['model_trial']} + {best_row['strategy_trial']} ==="
    )
    model_trial = next(trial for trial in model_trials if trial["trial_name"] == best_row["model_trial"])
    explicit_reuse_test = _resolve_reuse_artifact(reuse_test_cache or None, reuse_test_result_path or None)
    if explicit_reuse_test is not None:
        test_pred, test_label = _load_signal(explicit_reuse_test["cache_path"])
        test_metrics = _evaluate_signal(
            model_trial["trial_name"],
            _strategy_trial_from_row(best_row),
            test_pred,
            test_label,
            window_key=window_key,
            eval_segment="test",
            backtest_segment="test",
            recorder_name=f"{model_trial['trial_name']}__{best_row['strategy_trial']}__test_reuse",
        )
        test_result = {
            "result_path": explicit_reuse_test["result_path"],
            "cache_path": explicit_reuse_test["cache_path"],
            "recorder_id": test_metrics["recorder_id"],
            "IC": test_metrics["IC"],
            "Rank IC": test_metrics["Rank IC"],
            "with_cost_ann_return": test_metrics["with_cost_ann_return"],
            "with_cost_ir": test_metrics["with_cost_ir"],
            "with_cost_mdd": test_metrics["with_cost_mdd"],
        }
        print(f"reuse test base cache for {model_trial['trial_name']}: {explicit_reuse_test['cache_path']}")
    else:
        test_result = _run_family_trial(
            model_family,
            model_trial,
            seed=seed,
            gpu_slots=gpu_slots,
            result_suffix=_trial_suffix(f"{best_row['model_trial']}_final_test", result_suffix),
            deterministic_runtime=deterministic_runtime,
            window_key=window_key,
            n_drop=int(best_row["n_drop"]),
            risk_degree=float(best_row["risk_degree"]),
            strategy_class=best_row["strategy_class"],
            strategy_module_path=best_row["strategy_module_path"],
            strategy_kwargs_extra=_parse_optional_literal(best_row["kwargs_extra"]),
            eval_segment="test",
            backtest_segment="test",
        )
    _write_final_test(
        best_row,
        test_result,
        output_path=final_test_path,
        window_key=window_key,
        seed=seed,
        deterministic_runtime=deterministic_runtime,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Global Alpha360 tuning: supports TRA and JointTRA+TechAlpha158, and runs through final test."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-slots", default=GPU_SLOTS)
    parser.add_argument("--result-suffix", default="")
    parser.add_argument("--window-key", default=WINDOW_KEY, choices=sorted(WINDOWS))
    parser.add_argument(
        "--model-family",
        default=DEFAULT_MODEL_FAMILY,
        choices=["joint_tra_tech158", "tra"],
        help="Model family to tune end-to-end. Default is the TRA baseline path.",
    )
    parser.add_argument(
        "--strategy-profile",
        default=STRATEGY_PROFILE_FULL,
        choices=[STRATEGY_PROFILE_FULL, STRATEGY_PROFILE_TRA_BEST_72],
        help="Validation strategy subspace. `tra_best_72` restricts Stage-1 search to the documented 72-strategy TRA baseline subset.",
    )
    parser.add_argument("--reuse-validation-cache", default="", help="Existing validation-segment base cache to reuse instead of rerunning rolling base training.")
    parser.add_argument("--reuse-validation-result-path", default="", help="Optional existing validation rolling result summary path recorded alongside --reuse-validation-cache.")
    parser.add_argument("--reuse-test-cache", default="", help="Existing test-segment base cache to reuse instead of rerunning final-test rolling base evaluation.")
    parser.add_argument("--reuse-test-result-path", default="", help="Optional existing final-test rolling result summary path recorded alongside --reuse-test-cache.")
    parser.add_argument("--dual-seed-study", action="store_true", help="Run dual-seed TRA base training and enumerate each seed's 180 validation/test strategy rows.")
    parser.add_argument("--dual-seed-base-only", action="store_true", help="For --dual-seed-study, stop after fresh validation/test base caches and write a base-run summary without the slow strategy grid.")
    parser.add_argument("--dual-seed-segment", choices=["both", "valid", "test"], default="both", help="For --dual-seed-base-only, run both segments or only one segment.")
    parser.add_argument("--smoke-only", action="store_true", help="For dual-seed study, run only the shared-handler base-training smoke test and stop early.")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DUAL_SEED_DEFAULT_SEEDS), help="Comma-separated seeds for dual-seed study.")
    parser.add_argument("--batch-size", type=int, default=DUAL_SEED_BATCH_SIZE, help="Batch size used by dual-seed base training.")
    parser.add_argument("--smoke-epochs", type=int, default=DUAL_SEED_SMOKE_EPOCHS, help="Temporary epoch count for --smoke-only; formal dual-seed training keeps the model config epochs.")
    parser.add_argument("--rolling-task-limit", type=int, default=None, help="Optional rolling task limit, useful for smoke testing.")
    parser.add_argument("--max-concurrent-seeds", type=int, default=DUAL_SEED_ROLLING_JOBS_PER_GPU, help="For dual-seed study, number of rolling-task workers allowed per physical GPU.")
    parser.add_argument("--min-available-ram-gb", type=float, default=DUAL_SEED_MIN_AVAILABLE_RAM_GB, help="Abort if MemAvailable drops below this threshold.")
    parser.add_argument("--max-extra-swap-gb", type=float, default=DUAL_SEED_MAX_EXTRA_SWAP_GB, help="Abort if swap usage grows beyond this run-baseline delta.")
    parser.add_argument("--resource-poll-seconds", type=float, default=DUAL_SEED_RESOURCE_POLL_SECONDS, help="Polling interval for RAM/swap/GPU monitoring.")
    parser.add_argument("--rerun-final", action="store_true", help="rerun final test even if a final summary already exists")
    parser.add_argument("--skip-final", action="store_true", help="Run validation tuning only and skip test evaluation.")
    parser.add_argument("--legacy-runtime", action="store_true", help="Disable deterministic TRA runtime controls.")
    args = parser.parse_args()
    if args.dual_seed_study and args.strategy_profile != STRATEGY_PROFILE_FULL:
        raise ValueError("--strategy-profile is currently only supported for the single-seed validation/final-test path")
    if args.dual_seed_study:
        run_dual_seed_study(
            seeds=_parse_seeds(args.seeds),
            gpu_slots=args.gpu_slots,
            result_suffix=args.result_suffix,
            dual_seed_segment=args.dual_seed_segment,
            deterministic_runtime=not args.legacy_runtime,
            window_key=args.window_key,
            model_family=args.model_family,
            batch_size=args.batch_size,
            smoke_only=args.smoke_only,
            rolling_task_limit=args.rolling_task_limit,
            max_concurrent_seeds=args.max_concurrent_seeds,
            min_available_ram_gb=args.min_available_ram_gb,
            max_extra_swap_gb=args.max_extra_swap_gb,
            resource_poll_seconds=args.resource_poll_seconds,
            smoke_epochs=args.smoke_epochs,
            base_only=args.dual_seed_base_only,
        )
    else:
        main(
            skip_final_if_exists=not args.rerun_final,
            seed=args.seed,
            gpu_slots=args.gpu_slots,
            result_suffix=args.result_suffix,
            deterministic_runtime=not args.legacy_runtime,
            window_key=args.window_key,
            run_final_test=not args.skip_final,
            model_family=args.model_family,
            strategy_profile=args.strategy_profile,
            reuse_validation_cache=args.reuse_validation_cache,
            reuse_validation_result_path=args.reuse_validation_result_path,
            reuse_test_cache=args.reuse_test_cache,
            reuse_test_result_path=args.reuse_test_result_path,
        )
