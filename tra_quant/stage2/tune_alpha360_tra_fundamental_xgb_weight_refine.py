from __future__ import annotations

import argparse
import copy
import json
import importlib.util
from ast import literal_eval
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.trainer import task_train
from qlib.utils import hash_args, init_instance_by_config

from extreme_event_mask import apply_train_label_mask, compose_exp_suffix, suffix_path
from rolling_xgboost_alpha158_latest import (
    run as run_xgb,
    build_base_task as build_xgb_base_task,
    DEFAULT_HANDLER_KWARGS_EXTRA as XGB_DEFAULT_HANDLER_KWARGS_EXTRA,
    HANDLER_CACHE_DIR as XGB_HANDLER_CACHE_DIR,
    CACHE_DIR as XGB_CACHE_DIR,
    PROVIDER_URI as XGB_PROVIDER_URI,
    WINDOWS as XGB_WINDOWS,
    _build_handler_config as _build_xgb_handler_config,
    _metric_value as _xgb_metric_value,
    _normalize_gpu_slots,
    _recommended_nthread,
)
from tune_alpha360_alpha158_dnn_strict_fusion import (
    _align_signals,
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

DEFAULT_WINDOW_KEY = "w1"
CURRENT_BASELINE_MANIFEST = BASE_DIR / "current_best_baseline_manifest.json"
RANK_ENSEMBLE_RUNNER_PATH = BASE_DIR / "run_rank_ensemble_tra_alpha360_once.py"
TRA_BEST_REPRO_JSON_PATH = BASE_DIR / "tmp" / "repro_rank_ensemble_single_strategy_v2.json"
TRA_BEST_VALIDATION_SUMMARY_PATH = BASE_DIR / "tmp" / "rank_ensemble_old6_validation_summary_20260517.txt"
TRA_BEST_FINAL_TEST_SUMMARY_PATH = BASE_DIR / "tmp" / "rank_ensemble_old6_final_test_summary_20260517.txt"

ACCOUNT = 150000
TOPK = 5
BENCHMARK = "SH000300"
INSTRUMENTS = "csi300"
GPU_SLOTS = "0,1"

FUNDAMENTAL_FEATURE_PATH = str(
    BASE_DIR.parent.parent
    / "training_data"
    / "processed"
    / "training"
    / "csi300_daily_fundamental_features_rankpct_qlib.parquet"
)

SELECTED_TRIAL = {
    "trial_name": "xgb03_techA3d_fund_rankpct_base_s240",
    "handler_class": "TechAlpha158A3DFundamental",
    "handler_module_path": "qlib.contrib.data.custom_handler",
    "handler_kwargs_extra": {
        "fundamental_feature_path": FUNDAMENTAL_FEATURE_PATH,
    },
    "model_key": "xgb_base",
    # Freeze the archived current-best baseline against qlib's XGBModel wrapper drift.
    # The archived result was produced when fit-level defaults effectively used 1000/50/20.
    "model_kwargs_override": {
        "num_boost_round": 1000,
        "early_stopping_rounds": 50,
        "verbose_eval": 20,
    },
    "step": 240,
}

FUSION_MODE = "zscore"
TAB_WEIGHTS = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
TARGETED_TAB_WEIGHTS = [0.03, 0.05]
FILTER_SPECS = [
    {"low_q": 0.15, "penalty": 0.25, "high_q": pd.NA, "bonus": 0.0},
    {"low_q": 0.20, "penalty": 0.25, "high_q": pd.NA, "bonus": 0.0},
    {"low_q": 0.20, "penalty": 0.50, "high_q": pd.NA, "bonus": 0.0},
    {"low_q": 0.30, "penalty": 0.50, "high_q": pd.NA, "bonus": 0.0},
    {"low_q": 0.20, "penalty": 0.50, "high_q": 0.80, "bonus": 0.15},
    {"low_q": 0.30, "penalty": 0.50, "high_q": 0.80, "bonus": 0.15},
]
XGB_VALID_STAGE_SUFFIX = "tra_fund_rankpct_fusion_valid"
XGB_TEST_STAGE_SUFFIX = "tra_fund_rankpct_fusion_test"
XGB_ANCHORED_VALID_STAGE_SUFFIX = "tra_fund_rankpct_anchor_valid"
XGB_ANCHORED_TEST_STAGE_SUFFIX = "tra_fund_rankpct_anchor_test"

TRA_RANK_ENSEMBLE_HANDLER_KWARGS = {"label": ["Ref($close, -6) / Ref($close, -1) - 1"]}

_QLIB_INITIALIZED = False

_PRAC_CLASS = "PracticalTopkDropoutStrategy"
_PRAC_MODULE = "qlib.contrib.strategy.custom_signal_strategy"


def _prac(name: str, risk_degree: float, score_margin: float, hold_thresh: int, n_drop: int = 1) -> dict:
    return {
        "trial_name": name,
        "strategy_class": _PRAC_CLASS,
        "strategy_module_path": _PRAC_MODULE,
        "n_drop": n_drop,
        "risk_degree": risk_degree,
        "kwargs_extra": {
            "score_margin": score_margin,
            "hold_thresh": hold_thresh,
            "slot_budget_ratio": 1.0,
        },
    }


def _risk_tag(risk_degree: float) -> str:
    if risk_degree == 1.0:
        return "r10"
    return f"r{str(risk_degree).replace('.', '')}"


def _n_drop_tag(n_drop: int) -> str:
    return "" if n_drop == 1 else f"_drop{n_drop}"


def _build_stage2_strategy_trials() -> list[dict]:
    trials = []
    margins = (("m000", 0.0), ("m025", 0.0025), ("m050", 0.0050))
    for margin_tag, score_margin in margins:
        for hold_thresh in (3, 4, 5, 7):
            for risk_degree in (0.80, 0.85, 0.90, 0.95):
                for n_drop in (1, 2):
                    trial_name = (
                        f"prac_{margin_tag}{_n_drop_tag(n_drop)}_hold{hold_thresh}_{_risk_tag(risk_degree)}"
                    )
                    trials.append(_prac(trial_name, risk_degree, score_margin, hold_thresh, n_drop=n_drop))
    return trials


STAGE2_AWARE_STRATEGY_TRIALS = _build_stage2_strategy_trials()


def _ensure_qlib_initialized():
    global _QLIB_INITIALIZED
    if not _QLIB_INITIALIZED:
        qlib.init(provider_uri=XGB_PROVIDER_URI, region=REG_CN)
        _QLIB_INITIALIZED = True


def _load_rank_ensemble_runner():
    spec = importlib.util.spec_from_file_location("run_rank_ensemble_tra_alpha360_stage2", RANK_ENSEMBLE_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rank_ensemble_strategy_from_payload(payload: dict) -> dict:
    return _prac(
        payload["strategy_trial"],
        float(payload["risk_degree"]),
        float(payload["score_margin"]),
        int(payload["hold_thresh"]),
    )


def _metric_block_from_rank_payload(metrics: dict, prefix: str) -> dict:
    return {
        f"{prefix}_IC": metrics["IC"],
        f"{prefix}_Rank_IC": metrics["Rank IC"],
        f"{prefix}_with_cost_ann_return": metrics["ann"],
        f"{prefix}_with_cost_ir": metrics["ir"],
        f"{prefix}_with_cost_mdd": metrics["mdd"],
    }


def _load_rank_ensemble_stage1(
    *,
    validation_summary_path: str | Path,
    final_test_summary_path: str | Path,
    repro_json_path: str | Path,
    result_suffix: str,
) -> tuple[dict, dict]:
    runner = _load_rank_ensemble_runner()
    tra_mod = runner.load_module()
    validation_summary_path = Path(validation_summary_path).resolve()
    final_test_summary_path = Path(final_test_summary_path).resolve()
    repro_json_path = Path(repro_json_path).resolve()

    payload = json.loads(repro_json_path.read_text(encoding="utf-8"))
    validation_wrapper = _parse_summary_file(validation_summary_path)
    final_wrapper = _parse_summary_file(final_test_summary_path)
    valid_cache_paths = runner._load_seed_cache_paths(validation_summary_path, "seed_runs_valid")
    test_cache_paths = runner._load_seed_cache_paths(final_test_summary_path, "seed_runs_test")

    valid_pred, valid_label, valid_common_rows = runner.rank_ensemble_signal(tra_mod, valid_cache_paths)
    test_pred, test_label, test_common_rows = runner.rank_ensemble_signal(tra_mod, test_cache_paths)
    if int(valid_common_rows) != int(payload["valid_common_rows"]):
        raise RuntimeError(f"rank-ensemble valid rows changed: {valid_common_rows} != {payload['valid_common_rows']}")
    if int(test_common_rows) != int(payload["test_common_rows"]):
        raise RuntimeError(f"rank-ensemble test rows changed: {test_common_rows} != {payload['test_common_rows']}")

    valid_cache_path = _write_fused_cache(
        compose_exp_suffix("tra_best_rank_ensemble_stage1_valid", result_suffix),
        valid_pred,
        valid_label,
    )
    test_cache_path = _write_fused_cache(
        compose_exp_suffix("tra_best_rank_ensemble_stage1_test", result_suffix),
        test_pred,
        test_label,
    )

    strategy_trial = _rank_ensemble_strategy_from_payload(payload)
    base_fields = {
        "window_key": validation_wrapper["window_key"],
        "train": validation_wrapper["train"],
        "valid": validation_wrapper["valid"],
        "test": validation_wrapper["test"],
        "model_trial": "rank_ensemble_3seed_tra72_old6",
        "model_key": "tra_rank_ensemble",
        "handler_kwargs_extra": TRA_RANK_ENSEMBLE_HANDLER_KWARGS,
        "strategy_trial": strategy_trial["trial_name"],
        "strategy_class": strategy_trial["strategy_class"],
        "strategy_module_path": strategy_trial["strategy_module_path"],
        "n_drop": strategy_trial["n_drop"],
        "risk_degree": strategy_trial["risk_degree"],
        "kwargs_extra": strategy_trial["kwargs_extra"],
        "stage1_source": "rank_ensemble",
        "stage1_repro_json_path": str(repro_json_path),
        "stage1_validation_summary_path": str(validation_summary_path),
        "stage1_final_test_summary_path": str(final_test_summary_path),
        "stage1_validation_seed_cache_paths": [str(path) for path in valid_cache_paths],
        "stage1_test_seed_cache_paths": [str(path) for path in test_cache_paths],
        "stage1_valid_common_rows": valid_common_rows,
        "stage1_test_common_rows": test_common_rows,
    }
    tra_validation = {
        **base_fields,
        **_metric_block_from_rank_payload(payload["validation"], "validation"),
        "validation_score": payload["validation"]["score"],
        "validation_cache_path": str(valid_cache_path),
        "validation_recorder_id": "rank_ensemble_old6_valid_cache",
    }
    tra_final = {
        **base_fields,
        **_metric_block_from_rank_payload(payload["test"], "test"),
        "test_cache_path": str(test_cache_path),
        "test_recorder_id": "rank_ensemble_old6_test_cache",
    }
    return tra_validation, tra_final


def _load_current_best_final(current_best_final_path: str | None = None) -> dict:
    if current_best_final_path:
        return _parse_summary_file(Path(current_best_final_path).resolve())
    if CURRENT_BASELINE_MANIFEST.exists():
        data = json.loads(CURRENT_BASELINE_MANIFEST.read_text(encoding="utf-8"))
        summary_path = data.get("reproduction", {}).get("stage2_final_test_summary")
        if summary_path:
            return _parse_summary_file(Path(summary_path))
    return _parse_summary_file(BASE_DIR / "alpha360_tra_fundamental_xgb_weight_refine_final_test.txt")


def _strategy_hold_thresh(strategy_trial: dict) -> int | None:
    kwargs_extra = strategy_trial.get("kwargs_extra") or {}
    hold_thresh = kwargs_extra.get("hold_thresh")
    if hold_thresh is None:
        return None
    return int(hold_thresh)


def _filtered_stage2_strategy_trials(*, hold_min: int, risk_max: float) -> list[dict]:
    filtered = []
    for strategy_trial in STAGE2_AWARE_STRATEGY_TRIALS:
        hold_thresh = _strategy_hold_thresh(strategy_trial)
        if hold_thresh is None or hold_thresh < hold_min:
            continue
        if float(strategy_trial["risk_degree"]) > risk_max:
            continue
        filtered.append(copy.deepcopy(strategy_trial))
    if not filtered:
        raise RuntimeError(
            f"no stage2-aware strategy trials remain after filtering hold>={hold_min} risk<={risk_max}"
        )
    return filtered


def _strategy_cfg_from_trial(validation_summary: dict, strategy_trial: dict, eval_segment: str) -> dict:
    return {
        "handler_kwargs_extra": validation_summary["handler_kwargs_extra"],
        "strategy_class": strategy_trial["strategy_class"],
        "strategy_module_path": strategy_trial["strategy_module_path"],
        "kwargs_extra": strategy_trial["kwargs_extra"],
        "n_drop": strategy_trial["n_drop"],
        "risk_degree": strategy_trial["risk_degree"],
        "eval_segment": eval_segment,
        "backtest_segment": eval_segment,
    }


def _paths_for_window(window_key: str, result_suffix: str = ""):
    if window_key == "w1":
        return {
            "tra_validation_best": suffix_path(BASE_DIR / "tra_strict_validation_best.txt", result_suffix),
            "tra_final_test": suffix_path(BASE_DIR / "tra_strict_final_test.txt", result_suffix),
            "validation_results": suffix_path(
                BASE_DIR / "alpha360_tra_fundamental_xgb_weight_refine_validation_results.csv", result_suffix
            ),
            "validation_best": suffix_path(
                BASE_DIR / "alpha360_tra_fundamental_xgb_weight_refine_validation_best.txt", result_suffix
            ),
            "final_test": suffix_path(
                BASE_DIR / "alpha360_tra_fundamental_xgb_weight_refine_final_test.txt", result_suffix
            ),
        }
    return {
        "tra_validation_best": suffix_path(BASE_DIR / f"tra_strict_{window_key}_validation_best.txt", result_suffix),
        "tra_final_test": suffix_path(BASE_DIR / f"tra_strict_{window_key}_final_test.txt", result_suffix),
        "validation_results": suffix_path(
            BASE_DIR / f"alpha360_tra_fundamental_xgb_weight_refine_{window_key}_validation_results.csv", result_suffix
        ),
        "validation_best": suffix_path(
            BASE_DIR / f"alpha360_tra_fundamental_xgb_weight_refine_{window_key}_validation_best.txt", result_suffix
        ),
        "final_test": suffix_path(
            BASE_DIR / f"alpha360_tra_fundamental_xgb_weight_refine_{window_key}_final_test.txt", result_suffix
        ),
    }


def _result_path_for_xgb(trial: dict, stage_suffix: str, *, window_key: str, result_suffix: str = "") -> Path:
    stage_token = compose_exp_suffix(stage_suffix, result_suffix)
    return BASE_DIR / (
        f"rolling_result_xgb_{window_key}_{trial['model_key']}_{trial['handler_class']}_step{int(trial['step'])}_k{TOPK}"
        f"_d1_a{ACCOUNT}_r07_{trial['trial_name']}_{stage_token}.txt"
    )


def _direct_result_path_for_xgb(trial: dict, stage_suffix: str, *, window_key: str, result_suffix: str = "") -> Path:
    stage_token = compose_exp_suffix(stage_suffix, result_suffix)
    return BASE_DIR / (
        f"direct_result_xgb_{window_key}_{trial['model_key']}_{trial['handler_class']}_k{TOPK}"
        f"_d1_a{ACCOUNT}_r07_{trial['trial_name']}_{stage_token}.txt"
    )


def _prepare_direct_handler_cache(
    *,
    window_key: str,
    handler_class: str,
    handler_module_path: str,
    instruments: str,
    handler_kwargs_extra: dict | None,
    fit_start_time: str,
    fit_end_time: str,
) -> str:
    XGB_HANDLER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    handler_config = _build_xgb_handler_config(
        window_key=window_key,
        handler_class=handler_class,
        handler_module_path=handler_module_path,
        instruments=instruments,
        handler_kwargs_extra=handler_kwargs_extra,
    )
    handler_config = copy.deepcopy(handler_config)
    handler_config["kwargs"]["fit_start_time"] = fit_start_time
    handler_config["kwargs"]["fit_end_time"] = fit_end_time
    handler_hash = hash_args(handler_config)
    handler_path = XGB_HANDLER_CACHE_DIR / f"{handler_class}.{handler_hash[:10]}.cast.pkl"
    handler_uri = f"file://{handler_path}"
    if not handler_path.exists():
        handler = init_instance_by_config(handler_config, accept_types=DataHandlerLP)
        DataHandlerLP.cast(handler).to_pickle(handler_path, dump_all=True)
    try:
        init_instance_by_config(handler_uri, accept_types=DataHandlerLP)
    except Exception:
        handler = init_instance_by_config(handler_config, accept_types=DataHandlerLP)
        DataHandlerLP.cast(handler).to_pickle(handler_path, dump_all=True)
    return handler_uri


def _anchored_segments(window_key: str) -> dict:
    window = XGB_WINDOWS[window_key]
    train_start, train_end = window["train"]
    train_end_year = pd.Timestamp(train_end).year
    inner_train_end = f"{train_end_year - 1}-12-31"
    inner_valid_start = f"{train_end_year}-01-01"
    return {
        "selection_train": [train_start, inner_train_end],
        "selection_model_valid": [inner_valid_start, train_end],
        "selection_predict": list(window["valid"]),
        "final_train": list(window["train"]),
        "final_model_valid": list(window["valid"]),
        "final_predict": list(window["test"]),
    }


def _run_direct_xgb_trial(
    trial: dict,
    stage_suffix: str,
    *,
    window_key: str,
    result_suffix: str,
    train_segment: list[str],
    model_valid_segment: list[str],
    predict_segment: list[str],
    backtest_segment_key: str,
    train_label_mask_start: str | None = None,
    train_label_mask_end: str | None = None,
) -> dict:
    _ensure_qlib_initialized()
    handler_kwargs_extra = apply_train_label_mask(
        trial.get("handler_kwargs_extra"),
        train_label_mask_start,
        train_label_mask_end,
        base_learn_processors=XGB_DEFAULT_HANDLER_KWARGS_EXTRA["learn_processors"],
    )
    handler_uri = _prepare_direct_handler_cache(
        window_key=window_key,
        handler_class=trial["handler_class"],
        handler_module_path=trial["handler_module_path"],
        instruments=INSTRUMENTS,
        handler_kwargs_extra=handler_kwargs_extra,
        fit_start_time=train_segment[0],
        fit_end_time=train_segment[1],
    )
    gpu_slots = _normalize_gpu_slots(GPU_SLOTS)
    model_kwargs_override = copy.deepcopy(trial.get("model_kwargs_override") or {})
    model_kwargs_override.setdefault("nthread", _recommended_nthread(gpu_slots[:1] or gpu_slots))
    task = build_xgb_base_task(
        topk=TOPK,
        n_drop=1,
        window_key=window_key,
        model_key=trial["model_key"],
        handler_class=trial["handler_class"],
        handler_module_path=trial["handler_module_path"],
        account=ACCOUNT,
        risk_degree=0.70,
        benchmark=BENCHMARK,
        instruments=INSTRUMENTS,
        model_kwargs_override=model_kwargs_override,
        handler_kwargs_extra=handler_kwargs_extra,
        handler_override=handler_uri,
        eval_segment=backtest_segment_key,
        backtest_segment=backtest_segment_key,
    )
    task["dataset"]["kwargs"]["segments"] = {
        "train": train_segment,
        "valid": model_valid_segment,
        "test": predict_segment,
    }
    stage_token = compose_exp_suffix(stage_suffix, result_suffix)
    experiment_name = (
        f"direct_xgb_models_{window_key}_{trial['model_key']}_{trial['handler_class']}_k{TOPK}_d1_a{ACCOUNT}"
        f"_r07_{trial['trial_name']}_{stage_token}"
    )
    recorder = task_train(task, experiment_name)
    metrics = recorder.list_metrics()
    pred, label = _load_signal_from_recorder(recorder.id)
    cache_path = XGB_CACHE_DIR / (
        f"direct_cache_xgb_{window_key}_{trial['model_key']}_{trial['handler_class']}_k{TOPK}_d1_a{ACCOUNT}"
        f"_r07_{trial['trial_name']}_{stage_token}.pkl"
    )
    XGB_CACHE_DIR.mkdir(exist_ok=True)
    pd.to_pickle({"pred": pred, "label": label}, cache_path)
    result_path = _direct_result_path_for_xgb(
        trial,
        stage_suffix,
        window_key=window_key,
        result_suffix=result_suffix,
    )
    result_path.write_text(
        "\n".join(
            [
                f"experiment={experiment_name}",
                f"recorder_id={recorder.id}",
                f"window_key={window_key}",
                f"model_key={trial['model_key']}",
                f"handler_class={trial['handler_class']}",
                f"handler_module_path={trial['handler_module_path']}",
                f"account={ACCOUNT}",
                f"risk_degree=0.7",
                f"provider_uri={XGB_PROVIDER_URI}",
                f"benchmark={BENCHMARK}",
                f"instruments={INSTRUMENTS}",
                f"handler_cache_path={handler_uri}",
                f"cache_path={cache_path}",
                f"train_segment={train_segment}",
                f"model_valid_segment={model_valid_segment}",
                f"predict_segment={predict_segment}",
                f"backtest_segment={backtest_segment_key}",
                f"model_kwargs_override={model_kwargs_override}",
                f"IC={metrics.get('IC')}",
                f"Rank IC={metrics.get('Rank IC')}",
                f"with_cost_ann_return={_xgb_metric_value(metrics, recorder.id, '1day.excess_return_with_cost.annualized_return')}",
                f"with_cost_ir={_xgb_metric_value(metrics, recorder.id, '1day.excess_return_with_cost.information_ratio')}",
                f"with_cost_mdd={_xgb_metric_value(metrics, recorder.id, '1day.excess_return_with_cost.max_drawdown')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "result_path": str(result_path),
        "cache_path": str(cache_path),
        "recorder_id": recorder.id,
        "IC": metrics.get("IC"),
        "Rank IC": metrics.get("Rank IC"),
        "with_cost_ann_return": _xgb_metric_value(metrics, recorder.id, "1day.excess_return_with_cost.annualized_return"),
        "with_cost_ir": _xgb_metric_value(metrics, recorder.id, "1day.excess_return_with_cost.information_ratio"),
        "with_cost_mdd": _xgb_metric_value(metrics, recorder.id, "1day.excess_return_with_cost.max_drawdown"),
        "train_segment": train_segment,
        "model_valid_segment": model_valid_segment,
        "predict_segment": predict_segment,
        "backtest_segment": backtest_segment_key,
    }


def _run_xgb_trial(
    trial: dict,
    eval_segment: str,
    stage_suffix: str,
    *,
    window_key: str,
    result_suffix: str = "",
    train_label_mask_start: str | None = None,
    train_label_mask_end: str | None = None,
) -> dict:
    handler_kwargs_extra = apply_train_label_mask(
        trial.get("handler_kwargs_extra"),
        train_label_mask_start,
        train_label_mask_end,
        base_learn_processors=XGB_DEFAULT_HANDLER_KWARGS_EXTRA["learn_processors"],
    )
    result_path = _result_path_for_xgb(trial, stage_suffix, window_key=window_key, result_suffix=result_suffix)
    run_xgb(
        step=int(trial["step"]),
        topk=TOPK,
        n_drop=1,
        window_key=window_key,
        model_key=trial["model_key"],
        model_kwargs_override=trial.get("model_kwargs_override"),
        exp_suffix=compose_exp_suffix(f"{trial['trial_name']}_{stage_suffix}", result_suffix),
        handler_class=trial["handler_class"],
        handler_module_path=trial["handler_module_path"],
        handler_kwargs_extra=handler_kwargs_extra,
        account=ACCOUNT,
        risk_degree=0.70,
        benchmark=BENCHMARK,
        instruments=INSTRUMENTS,
        gpu_slots=GPU_SLOTS,
        eval_segment=eval_segment,
        backtest_segment=eval_segment,
    )
    return _parse_summary_file(result_path)


def _numeric_or_none(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _xgb_rank_pct(tab_pred: pd.DataFrame) -> pd.DataFrame:
    return _normalize_scores(tab_pred, "rank_pct")


def _build_stage2_pred_from_row(seq_pred: pd.DataFrame, tab_pred: pd.DataFrame, row: dict) -> pd.DataFrame:
    fusion_mode = row["fusion_mode"]
    if fusion_mode == "baseline":
        return seq_pred.copy()
    if fusion_mode in {"zscore", "rank_pct"}:
        seq_norm = _normalize_scores(seq_pred, fusion_mode)
        tab_norm = _normalize_scores(tab_pred, fusion_mode)
        return seq_norm.mul(float(row["seq_weight"])).add(tab_norm.mul(float(row["tab_weight"])), fill_value=0.0)
    if fusion_mode == "xgb_filter_zscore":
        seq_norm = _normalize_scores(seq_pred, "zscore")
        tab_rank = _xgb_rank_pct(tab_pred)
        adjusted = seq_norm.copy()
        score = adjusted["score"].copy()
        low_q = float(row["filter_low_quantile"])
        penalty = float(row["filter_penalty"])
        high_q = _numeric_or_none(row.get("filter_high_quantile"))
        bonus = float(row.get("filter_bonus", 0.0) or 0.0)
        score = score - (tab_rank["score"] <= low_q).astype(float) * penalty
        if high_q is not None and bonus:
            score = score + (tab_rank["score"] >= high_q).astype(float) * bonus
        adjusted["score"] = score
        return adjusted
    raise ValueError(f"unsupported fusion mode for stage2 rebuild: {fusion_mode}")


def _candidate_extra_fields(row: dict) -> dict:
    return {
        "signal_recipe": row.get("signal_recipe", pd.NA),
        "filter_low_quantile": row.get("filter_low_quantile", pd.NA),
        "filter_penalty": row.get("filter_penalty", pd.NA),
        "filter_high_quantile": row.get("filter_high_quantile", pd.NA),
        "filter_bonus": row.get("filter_bonus", pd.NA),
    }


def _label_alignment_report(
    seq_pred: pd.DataFrame,
    seq_label: pd.DataFrame,
    tab_pred: pd.DataFrame,
    tab_label: pd.DataFrame,
) -> dict:
    common_index = seq_pred.index.intersection(tab_pred.index).intersection(seq_label.index).intersection(tab_label.index)
    common_index = common_index.sort_values()
    seq_label_aligned = seq_label.loc[common_index].iloc[:, 0]
    tab_label_aligned = tab_label.loc[common_index].iloc[:, 0]
    label_diff = (seq_label_aligned - tab_label_aligned).abs()
    return {
        "alignment_common_rows": len(common_index),
        "alignment_common_days": int(
            common_index.get_level_values("datetime" if "datetime" in common_index.names else 0).nunique()
        )
        if len(common_index)
        else 0,
        "tab_label_max_abs_diff_vs_stage1_label": float(label_diff.max()) if len(label_diff) else pd.NA,
        "tab_label_mean_abs_diff_vs_stage1_label": float(label_diff.mean()) if len(label_diff) else pd.NA,
    }


def _write_validation_best(best_row: dict, *, tra_validation: dict, validation_best_path: Path):
    strict_validation = tra_validation
    lines = [
        f"window_key={strict_validation['window_key']}",
        f"train={strict_validation['train']}",
        f"valid={strict_validation['valid']}",
        f"test={strict_validation['test']}",
        f"selection_scope=local_weight_refine",
        f"base_model_trial={best_row['base_model_trial']}",
        f"base_strategy_trial={best_row['base_strategy_trial']}",
        f"strategy_trial={best_row.get('strategy_trial', best_row['base_strategy_trial'])}",
        f"strategy_class={best_row.get('strategy_class', pd.NA)}",
        f"strategy_module_path={best_row.get('strategy_module_path', pd.NA)}",
        f"n_drop={best_row.get('n_drop', pd.NA)}",
        f"risk_degree={best_row.get('risk_degree', pd.NA)}",
        f"kwargs_extra={best_row.get('kwargs_extra', pd.NA)}",
        f"tabular_trial={best_row['tabular_trial']}",
        f"tabular_handler_class={best_row['tabular_handler_class']}",
        f"tabular_model_key={best_row['tabular_model_key']}",
        f"fusion_mode={best_row['fusion_mode']}",
        f"seq_weight={best_row['seq_weight']}",
        f"tab_weight={best_row['tab_weight']}",
        f"validation_IC={best_row['IC']}",
        f"validation_Rank_IC={best_row['Rank IC']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"validation_cache_path={best_row['validation_cache_path']}",
        f"validation_recorder_id={best_row['validation_recorder_id']}",
        f"validation_score={best_row['score']}",
        f"tabular_validation_result_path={best_row['tabular_validation_result_path']}",
        f"tabular_validation_recorder_id={best_row['tabular_validation_recorder_id']}",
    ]
    for key in [
        "xgb_protocol",
        "selection_policy",
        "xgb_train_segment",
        "xgb_model_valid_segment",
        "xgb_predict_segment",
        "alignment_common_rows",
        "alignment_common_days",
        "tab_label_max_abs_diff_vs_stage1_label",
        "tab_label_mean_abs_diff_vs_stage1_label",
        "signal_recipe",
        "filter_low_quantile",
        "filter_penalty",
        "filter_high_quantile",
        "filter_bonus",
    ]:
        if key in best_row:
            lines.append(f"{key}={best_row[key]}")
    validation_best_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_test(
    best_row: dict,
    test_result: dict,
    tra_test: dict,
    current_best_test: dict,
    *,
    tra_validation: dict,
    final_test_path: Path,
):
    strict_validation = tra_validation
    improves_tra = float(test_result["with_cost_ann_return"]) > float(tra_test["test_with_cost_ann_return"])
    improves_current_best = (
        float(test_result["with_cost_ann_return"]) > float(current_best_test["test_with_cost_ann_return"])
    )
    lines = [
        f"window_key={strict_validation['window_key']}",
        f"train={strict_validation['train']}",
        f"valid={strict_validation['valid']}",
        f"test={strict_validation['test']}",
        f"selection_scope=pretest_fixed_protocol",
        f"test_improves_tra_incumbent={improves_tra}",
        f"test_improves_current_best_baseline={improves_current_best}",
        f"base_model_trial={best_row['base_model_trial']}",
        f"base_strategy_trial={best_row['base_strategy_trial']}",
        f"strategy_trial={best_row.get('strategy_trial', best_row['base_strategy_trial'])}",
        f"strategy_class={best_row.get('strategy_class', pd.NA)}",
        f"strategy_module_path={best_row.get('strategy_module_path', pd.NA)}",
        f"n_drop={best_row.get('n_drop', pd.NA)}",
        f"risk_degree={best_row.get('risk_degree', pd.NA)}",
        f"kwargs_extra={best_row.get('kwargs_extra', pd.NA)}",
        f"tabular_trial={best_row['tabular_trial']}",
        f"tabular_handler_class={best_row['tabular_handler_class']}",
        f"tabular_model_key={best_row['tabular_model_key']}",
        f"fusion_mode={best_row['fusion_mode']}",
        f"seq_weight={best_row['seq_weight']}",
        f"tab_weight={best_row['tab_weight']}",
        f"validation_score={best_row['score']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"test_cache_path={test_result['cache_path']}",
        f"test_recorder_id={test_result['recorder_id']}",
        f"test_IC={test_result['IC']}",
        f"test_Rank_IC={test_result['Rank IC']}",
        f"test_with_cost_ann_return={test_result['with_cost_ann_return']}",
        f"test_with_cost_ir={test_result['with_cost_ir']}",
        f"test_with_cost_mdd={test_result['with_cost_mdd']}",
        f"tra_incumbent_test_with_cost_ann_return={tra_test['test_with_cost_ann_return']}",
        f"tra_incumbent_test_with_cost_ir={tra_test['test_with_cost_ir']}",
        f"tra_incumbent_test_with_cost_mdd={tra_test['test_with_cost_mdd']}",
        f"current_best_test_with_cost_ann_return={current_best_test['test_with_cost_ann_return']}",
        f"current_best_test_with_cost_ir={current_best_test['test_with_cost_ir']}",
        f"current_best_test_with_cost_mdd={current_best_test['test_with_cost_mdd']}",
        f"tabular_test_result_path={test_result['tabular_test_result_path']}",
        f"tabular_test_recorder_id={test_result['tabular_test_recorder_id']}",
    ]
    for key in [
        "xgb_protocol",
        "selection_policy",
        "xgb_train_segment",
        "xgb_model_valid_segment",
        "xgb_predict_segment",
        "alignment_common_rows",
        "alignment_common_days",
        "tab_label_max_abs_diff_vs_stage1_label",
        "tab_label_mean_abs_diff_vs_stage1_label",
        "signal_recipe",
        "filter_low_quantile",
        "filter_penalty",
        "filter_high_quantile",
        "filter_bonus",
    ]:
        if key in best_row:
            lines.append(f"{key}={best_row[key]}")
    for key in [
        "test_xgb_train_segment",
        "test_xgb_model_valid_segment",
        "test_xgb_predict_segment",
        "test_alignment_common_rows",
        "test_alignment_common_days",
        "test_tab_label_max_abs_diff_vs_stage1_label",
        "test_tab_label_mean_abs_diff_vs_stage1_label",
    ]:
        if key in test_result:
            lines.append(f"{key}={test_result[key]}")
    final_test_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(
    window_key: str = DEFAULT_WINDOW_KEY,
    result_suffix: str = "",
    train_label_mask_start: str | None = None,
    train_label_mask_end: str | None = None,
    tra_validation_best_path: str | None = None,
    tra_final_test_path: str | None = None,
    stage1_source: str = "summary",
    tra_rank_validation_summary_path: str | None = None,
    tra_rank_final_test_summary_path: str | None = None,
    tra_rank_repro_json_path: str | None = None,
    current_best_final_path: str | None = None,
    xgb_protocol: str = "anchored",
    selection_policy: str = "allow-baseline",
    stage2_aware_strategy_search: bool = False,
    stage2_signal_profile: str = "linear",
    validation_fast_mode: bool = False,
    skip_final_test: bool = False,
    strategy_hold_min: int = 1,
    strategy_risk_max: float = 1.0,
):
    paths = _paths_for_window(window_key, result_suffix=result_suffix)
    tra_validation_best_path = Path(tra_validation_best_path).resolve() if tra_validation_best_path else paths["tra_validation_best"]
    tra_final_test_path = Path(tra_final_test_path).resolve() if tra_final_test_path else paths["tra_final_test"]
    validation_results_path = paths["validation_results"]
    validation_best_path = paths["validation_best"]
    final_test_path = paths["final_test"]

    if stage1_source == "summary":
        tra_validation = _parse_summary_file(tra_validation_best_path)
        tra_final = _parse_summary_file(tra_final_test_path)
    elif stage1_source == "rank-ensemble":
        tra_validation, tra_final = _load_rank_ensemble_stage1(
            validation_summary_path=tra_rank_validation_summary_path or TRA_BEST_VALIDATION_SUMMARY_PATH,
            final_test_summary_path=tra_rank_final_test_summary_path or TRA_BEST_FINAL_TEST_SUMMARY_PATH,
            repro_json_path=tra_rank_repro_json_path or TRA_BEST_REPRO_JSON_PATH,
            result_suffix=result_suffix,
        )
    else:
        raise ValueError(f"unknown stage1_source `{stage1_source}`")
    current_best_final = _load_current_best_final(current_best_final_path=current_best_final_path)
    strategy_cfg_valid = _load_alstm_strategy_cfg(tra_validation, eval_segment="valid")
    strategy_cfg_test = _load_alstm_strategy_cfg(tra_validation, eval_segment="test")
    stage2_strategy_trials = None
    if stage2_aware_strategy_search:
        stage2_strategy_trials = _filtered_stage2_strategy_trials(
            hold_min=strategy_hold_min,
            risk_max=strategy_risk_max,
        )
        print(
            f"stage2-aware strategy search enabled: #strategy_trials={len(stage2_strategy_trials)} "
            f"hold>={strategy_hold_min} risk<={strategy_risk_max}"
        )

    seq_valid_pred, seq_valid_label = _load_signal_from_cache(tra_validation["validation_cache_path"])
    anchored_segments = _anchored_segments(window_key)
    if xgb_protocol == "anchored":
        xgb_validation_result = _run_direct_xgb_trial(
            SELECTED_TRIAL,
            stage_suffix=XGB_ANCHORED_VALID_STAGE_SUFFIX,
            window_key=window_key,
            result_suffix=result_suffix,
            train_segment=anchored_segments["selection_train"],
            model_valid_segment=anchored_segments["selection_model_valid"],
            predict_segment=anchored_segments["selection_predict"],
            backtest_segment_key="valid",
            train_label_mask_start=train_label_mask_start,
            train_label_mask_end=train_label_mask_end,
        )
        tabular_validation_result_path = xgb_validation_result["result_path"]
    elif xgb_protocol == "rolling":
        xgb_validation_result = _run_xgb_trial(
            SELECTED_TRIAL,
            eval_segment="valid",
            stage_suffix=XGB_VALID_STAGE_SUFFIX,
            window_key=window_key,
            result_suffix=result_suffix,
            train_label_mask_start=train_label_mask_start,
            train_label_mask_end=train_label_mask_end,
        )
        tabular_validation_result_path = str(
            _result_path_for_xgb(
                SELECTED_TRIAL,
                XGB_VALID_STAGE_SUFFIX,
                window_key=window_key,
                result_suffix=result_suffix,
            )
        )
    else:
        raise ValueError(f"unknown xgb_protocol `{xgb_protocol}`")
    tab_valid_pred, tab_valid_label = _load_signal_from_recorder(xgb_validation_result["recorder_id"])
    validation_alignment_report = _label_alignment_report(
        seq_valid_pred,
        seq_valid_label,
        tab_valid_pred,
        tab_valid_label,
    )
    seq_pred_aligned, tab_pred_aligned, label_aligned = _align_signals(
        seq_valid_pred, seq_valid_label, tab_valid_pred, tab_valid_label
    )
    seq_norm = _normalize_scores(seq_pred_aligned, FUSION_MODE)
    tab_norm = _normalize_scores(tab_pred_aligned, FUSION_MODE)
    aligned_baseline_cache_path = _write_fused_cache(
        compose_exp_suffix(f"{xgb_protocol}_weight_refine_baseline_tra_only_aligned", result_suffix),
        seq_pred_aligned,
        label_aligned,
    )

    signal_candidates = [
        {
            "signal_trial_name": "weight_refine_baseline_tra_only",
            "base_model_trial": tra_validation["model_trial"],
            "base_strategy_trial": tra_validation["strategy_trial"],
            "tabular_trial": "none",
            "tabular_handler_class": "none",
            "tabular_model_key": "none",
            "fusion_mode": "baseline",
            "seq_weight": 1.0,
            "tab_weight": 0.0,
            "signal_recipe": "baseline",
            "pred": seq_pred_aligned,
            "label": label_aligned,
            "validation_cache_path": str(aligned_baseline_cache_path),
            "tabular_validation_result_path": pd.NA,
            "tabular_validation_recorder_id": pd.NA,
            "xgb_protocol": xgb_protocol,
            "selection_policy": selection_policy,
            "xgb_train_segment": pd.NA,
            "xgb_model_valid_segment": pd.NA,
            "xgb_predict_segment": pd.NA,
            **validation_alignment_report,
        }
    ]

    if stage2_signal_profile == "linear":
        active_tab_weights = TAB_WEIGHTS
        include_filter_candidates = False
    elif stage2_signal_profile == "targeted":
        active_tab_weights = TARGETED_TAB_WEIGHTS
        include_filter_candidates = True
    elif stage2_signal_profile == "filter":
        active_tab_weights = []
        include_filter_candidates = True
    elif stage2_signal_profile == "all":
        active_tab_weights = TAB_WEIGHTS
        include_filter_candidates = True
    else:
        raise ValueError(f"unknown stage2_signal_profile `{stage2_signal_profile}`")

    for tab_weight in active_tab_weights:
        seq_weight = round(1.0 - tab_weight, 2)
        signal_trial_name = f"weight_refine_tra_{SELECTED_TRIAL['trial_name']}_{FUSION_MODE}_tw{str(tab_weight).replace('.', '')}"
        candidate = {
            "fusion_mode": FUSION_MODE,
            "seq_weight": seq_weight,
            "tab_weight": tab_weight,
            "signal_recipe": f"linear_{FUSION_MODE}",
        }
        fused_pred = _build_stage2_pred_from_row(seq_pred_aligned, tab_pred_aligned, candidate)
        validation_cache_path = _write_fused_cache(
            compose_exp_suffix(f"{xgb_protocol}_{signal_trial_name}", result_suffix),
            fused_pred,
            label_aligned,
        )
        signal_candidates.append(
            {
                "signal_trial_name": signal_trial_name,
                "base_model_trial": tra_validation["model_trial"],
                "base_strategy_trial": tra_validation["strategy_trial"],
                "tabular_trial": SELECTED_TRIAL["trial_name"],
                "tabular_handler_class": SELECTED_TRIAL["handler_class"],
                "tabular_model_key": SELECTED_TRIAL["model_key"],
                "fusion_mode": FUSION_MODE,
                "seq_weight": seq_weight,
                "tab_weight": tab_weight,
                "signal_recipe": candidate["signal_recipe"],
                "pred": fused_pred,
                "label": label_aligned,
                "validation_cache_path": str(validation_cache_path),
                "tabular_validation_result_path": tabular_validation_result_path,
                "tabular_validation_recorder_id": xgb_validation_result["recorder_id"],
                "xgb_protocol": xgb_protocol,
                "selection_policy": selection_policy,
                "xgb_train_segment": xgb_validation_result.get("train_segment", pd.NA),
                "xgb_model_valid_segment": xgb_validation_result.get("model_valid_segment", pd.NA),
                "xgb_predict_segment": xgb_validation_result.get("predict_segment", pd.NA),
                **validation_alignment_report,
            }
        )

    if include_filter_candidates:
        for spec in FILTER_SPECS:
            high_q = spec["high_q"]
            high_tag = "none" if pd.isna(high_q) else str(high_q).replace(".", "")
            signal_trial_name = (
                f"xgb_filter_lq{str(spec['low_q']).replace('.', '')}_p{str(spec['penalty']).replace('.', '')}"
                f"_hq{high_tag}_b{str(spec['bonus']).replace('.', '')}"
            )
            candidate = {
                "fusion_mode": "xgb_filter_zscore",
                "seq_weight": 1.0,
                "tab_weight": 0.0,
                "signal_recipe": "xgb_low_rank_penalty_zscore",
                "filter_low_quantile": spec["low_q"],
                "filter_penalty": spec["penalty"],
                "filter_high_quantile": high_q,
                "filter_bonus": spec["bonus"],
            }
            filtered_pred = _build_stage2_pred_from_row(seq_pred_aligned, tab_pred_aligned, candidate)
            validation_cache_path = _write_fused_cache(
                compose_exp_suffix(f"{xgb_protocol}_{signal_trial_name}", result_suffix),
                filtered_pred,
                label_aligned,
            )
            signal_candidates.append(
                {
                    "signal_trial_name": signal_trial_name,
                    "base_model_trial": tra_validation["model_trial"],
                    "base_strategy_trial": tra_validation["strategy_trial"],
                    "tabular_trial": SELECTED_TRIAL["trial_name"],
                    "tabular_handler_class": SELECTED_TRIAL["handler_class"],
                    "tabular_model_key": SELECTED_TRIAL["model_key"],
                    "fusion_mode": candidate["fusion_mode"],
                    "seq_weight": candidate["seq_weight"],
                    "tab_weight": candidate["tab_weight"],
                    "signal_recipe": candidate["signal_recipe"],
                    "filter_low_quantile": candidate["filter_low_quantile"],
                    "filter_penalty": candidate["filter_penalty"],
                    "filter_high_quantile": candidate["filter_high_quantile"],
                    "filter_bonus": candidate["filter_bonus"],
                    "pred": filtered_pred,
                    "label": label_aligned,
                    "validation_cache_path": str(validation_cache_path),
                    "tabular_validation_result_path": tabular_validation_result_path,
                    "tabular_validation_recorder_id": xgb_validation_result["recorder_id"],
                    "xgb_protocol": xgb_protocol,
                    "selection_policy": selection_policy,
                    "xgb_train_segment": xgb_validation_result.get("train_segment", pd.NA),
                    "xgb_model_valid_segment": xgb_validation_result.get("model_valid_segment", pd.NA),
                    "xgb_predict_segment": xgb_validation_result.get("predict_segment", pd.NA),
                    **validation_alignment_report,
                }
            )

    rows: list[dict] = []
    if stage2_aware_strategy_search:
        for signal_candidate in signal_candidates:
            for strategy_trial in stage2_strategy_trials:
                row = {
                    "trial_name": f"{signal_candidate['signal_trial_name']}__{strategy_trial['trial_name']}",
                    "signal_trial_name": signal_candidate["signal_trial_name"],
                    "base_model_trial": signal_candidate["base_model_trial"],
                    "base_strategy_trial": signal_candidate["base_strategy_trial"],
                    "strategy_trial": strategy_trial["trial_name"],
                    "strategy_class": strategy_trial["strategy_class"],
                    "strategy_module_path": strategy_trial["strategy_module_path"],
                    "n_drop": strategy_trial["n_drop"],
                    "risk_degree": strategy_trial["risk_degree"],
                    "kwargs_extra": str(strategy_trial["kwargs_extra"]),
                    "tabular_trial": signal_candidate["tabular_trial"],
                    "tabular_handler_class": signal_candidate["tabular_handler_class"],
                    "tabular_model_key": signal_candidate["tabular_model_key"],
                    "fusion_mode": signal_candidate["fusion_mode"],
                    "seq_weight": signal_candidate["seq_weight"],
                    "tab_weight": signal_candidate["tab_weight"],
                    "status": "running",
                    "validation_cache_path": signal_candidate["validation_cache_path"],
                    "tabular_validation_result_path": signal_candidate["tabular_validation_result_path"],
                    "tabular_validation_recorder_id": signal_candidate["tabular_validation_recorder_id"],
                    "xgb_protocol": signal_candidate["xgb_protocol"],
                    "selection_policy": signal_candidate["selection_policy"],
                    "xgb_train_segment": signal_candidate["xgb_train_segment"],
                    "xgb_model_valid_segment": signal_candidate["xgb_model_valid_segment"],
                    "xgb_predict_segment": signal_candidate["xgb_predict_segment"],
                    "alignment_common_rows": signal_candidate["alignment_common_rows"],
                    "alignment_common_days": signal_candidate["alignment_common_days"],
                    "tab_label_max_abs_diff_vs_stage1_label": signal_candidate[
                        "tab_label_max_abs_diff_vs_stage1_label"
                    ],
                    "tab_label_mean_abs_diff_vs_stage1_label": signal_candidate[
                        "tab_label_mean_abs_diff_vs_stage1_label"
                    ],
                    **_candidate_extra_fields(signal_candidate),
                }
                try:
                    metrics = _evaluate_signal(
                        recorder_name=row["trial_name"],
                        pred=signal_candidate["pred"],
                        label=signal_candidate["label"],
                        strategy_cfg=_strategy_cfg_from_trial(tra_validation, strategy_trial, eval_segment="valid"),
                        experiment_name="alpha360_tra_fundamental_xgb_weight_refine_validation_eval",
                        fast_mode=validation_fast_mode,
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
                            "validation_recorder_id": metrics["recorder_id"],
                        }
                    )
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = repr(exc)[:500]
                rows = [existing for existing in rows if existing.get("trial_name") != row["trial_name"]] + [row]
                _persist(rows, validation_results_path)
    else:
        baseline_candidate = signal_candidates[0]
        baseline_row = {
            "trial_name": "weight_refine_baseline_tra_only",
            "signal_trial_name": "weight_refine_baseline_tra_only",
            "base_model_trial": tra_validation["model_trial"],
            "base_strategy_trial": tra_validation["strategy_trial"],
            "strategy_trial": tra_validation["strategy_trial"],
            "strategy_class": tra_validation["strategy_class"],
            "strategy_module_path": tra_validation["strategy_module_path"],
            "n_drop": tra_validation["n_drop"],
            "risk_degree": tra_validation["risk_degree"],
            "kwargs_extra": str(tra_validation["kwargs_extra"]),
            "tabular_trial": "none",
            "tabular_handler_class": "none",
            "tabular_model_key": "none",
            "fusion_mode": "baseline",
            "seq_weight": 1.0,
            "tab_weight": 0.0,
            "status": "running",
            "validation_cache_path": baseline_candidate["validation_cache_path"],
            "tabular_validation_result_path": pd.NA,
            "tabular_validation_recorder_id": pd.NA,
            "xgb_protocol": xgb_protocol,
            "selection_policy": selection_policy,
            "xgb_train_segment": pd.NA,
            "xgb_model_valid_segment": pd.NA,
            "xgb_predict_segment": pd.NA,
            **validation_alignment_report,
            **_candidate_extra_fields(baseline_candidate),
        }
        try:
            metrics = _evaluate_signal(
                recorder_name=baseline_row["trial_name"],
                pred=baseline_candidate["pred"],
                label=baseline_candidate["label"],
                strategy_cfg=strategy_cfg_valid,
                experiment_name="alpha360_tra_fundamental_xgb_weight_refine_validation_eval",
                fast_mode=validation_fast_mode,
            )
            baseline_row.update(
                {
                    "status": "success",
                    "IC": metrics["IC"],
                    "Rank IC": metrics["Rank IC"],
                    "with_cost_ann_return": metrics["with_cost_ann_return"],
                    "with_cost_ir": metrics["with_cost_ir"],
                    "with_cost_mdd": metrics["with_cost_mdd"],
                    "score": metrics["score"],
                    "validation_recorder_id": metrics["recorder_id"],
                }
            )
        except Exception as exc:
            baseline_row["status"] = "failed"
            baseline_row["error"] = repr(exc)[:500]
        rows = [baseline_row]
        _persist(rows, validation_results_path)

        for signal_candidate in signal_candidates[1:]:
            row = {
                "trial_name": signal_candidate["signal_trial_name"],
                "signal_trial_name": signal_candidate["signal_trial_name"],
                "base_model_trial": signal_candidate["base_model_trial"],
                "base_strategy_trial": signal_candidate["base_strategy_trial"],
                "strategy_trial": tra_validation["strategy_trial"],
                "strategy_class": tra_validation["strategy_class"],
                "strategy_module_path": tra_validation["strategy_module_path"],
                "n_drop": tra_validation["n_drop"],
                "risk_degree": tra_validation["risk_degree"],
                "kwargs_extra": str(tra_validation["kwargs_extra"]),
                "tabular_trial": signal_candidate["tabular_trial"],
                "tabular_handler_class": signal_candidate["tabular_handler_class"],
                "tabular_model_key": signal_candidate["tabular_model_key"],
                "fusion_mode": signal_candidate["fusion_mode"],
                "seq_weight": signal_candidate["seq_weight"],
                "tab_weight": signal_candidate["tab_weight"],
                "status": "running",
                "validation_cache_path": signal_candidate["validation_cache_path"],
                "tabular_validation_result_path": signal_candidate["tabular_validation_result_path"],
                "tabular_validation_recorder_id": signal_candidate["tabular_validation_recorder_id"],
                "xgb_protocol": signal_candidate["xgb_protocol"],
                "selection_policy": signal_candidate["selection_policy"],
                "xgb_train_segment": signal_candidate["xgb_train_segment"],
                "xgb_model_valid_segment": signal_candidate["xgb_model_valid_segment"],
                "xgb_predict_segment": signal_candidate["xgb_predict_segment"],
                "alignment_common_rows": signal_candidate["alignment_common_rows"],
                "alignment_common_days": signal_candidate["alignment_common_days"],
                "tab_label_max_abs_diff_vs_stage1_label": signal_candidate[
                    "tab_label_max_abs_diff_vs_stage1_label"
                ],
                "tab_label_mean_abs_diff_vs_stage1_label": signal_candidate[
                    "tab_label_mean_abs_diff_vs_stage1_label"
                ],
                **_candidate_extra_fields(signal_candidate),
            }
            try:
                metrics = _evaluate_signal(
                    recorder_name=row["trial_name"],
                    pred=signal_candidate["pred"],
                    label=signal_candidate["label"],
                    strategy_cfg=strategy_cfg_valid,
                    experiment_name="alpha360_tra_fundamental_xgb_weight_refine_validation_eval",
                    fast_mode=validation_fast_mode,
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
                        "validation_recorder_id": metrics["recorder_id"],
                    }
                )
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = repr(exc)[:500]
            rows = [existing for existing in rows if existing.get("trial_name") != row["trial_name"]] + [row]
            _persist(rows, validation_results_path)

    success_rows = [row for row in rows if row["status"] == "success"]
    if not success_rows:
        raise RuntimeError("no successful local weight-refine rows")

    candidate_rows = success_rows
    if selection_policy == "best-fusion":
        candidate_rows = [row for row in success_rows if row["tabular_trial"] != "none"]
    elif selection_policy != "allow-baseline":
        raise ValueError(f"unknown selection_policy `{selection_policy}`")
    if not candidate_rows:
        raise RuntimeError(f"no successful rows for selection policy `{selection_policy}`")

    best_row = max(candidate_rows, key=lambda item: item["score"])
    _write_validation_best(
        best_row,
        tra_validation=tra_validation,
        validation_best_path=validation_best_path,
    )
    if skip_final_test:
        return

    selected_strategy_cfg_test = strategy_cfg_test
    if stage2_aware_strategy_search:
        selected_strategy_cfg_test = _strategy_cfg_from_trial(
            tra_validation,
            {
                "trial_name": best_row["strategy_trial"],
                "strategy_class": best_row["strategy_class"],
                "strategy_module_path": best_row["strategy_module_path"],
                "n_drop": int(best_row["n_drop"]),
                "risk_degree": float(best_row["risk_degree"]),
                "kwargs_extra": literal_eval(best_row["kwargs_extra"]),
            },
            eval_segment="test",
        )

    if best_row["tabular_trial"] == "none":
        if stage2_aware_strategy_search:
            seq_test_pred, seq_test_label = _load_signal_from_cache(tra_final["test_cache_path"])
            test_cache_path = _write_fused_cache(
                compose_exp_suffix(f"{xgb_protocol}_{best_row['trial_name']}_final_test", result_suffix),
                seq_test_pred,
                seq_test_label,
            )
            test_metrics = _evaluate_signal(
                recorder_name=f"{best_row['trial_name']}_final_test",
                pred=seq_test_pred,
                label=seq_test_label,
                strategy_cfg=selected_strategy_cfg_test,
                experiment_name="alpha360_tra_fundamental_xgb_weight_refine_final_test_eval",
            )
            test_metrics.update(
                {
                    "cache_path": str(test_cache_path),
                    "tabular_test_result_path": pd.NA,
                    "tabular_test_recorder_id": pd.NA,
                }
            )
            _write_final_test(
                best_row,
                test_metrics,
                tra_final,
                current_best_final,
                tra_validation=tra_validation,
                final_test_path=final_test_path,
            )
            return
        test_result = {
            "cache_path": tra_final["test_cache_path"],
            "recorder_id": tra_final["test_recorder_id"],
            "IC": tra_final["test_IC"],
            "Rank IC": tra_final["test_Rank_IC"],
            "with_cost_ann_return": tra_final["test_with_cost_ann_return"],
            "with_cost_ir": tra_final["test_with_cost_ir"],
            "with_cost_mdd": tra_final["test_with_cost_mdd"],
            "tabular_test_result_path": pd.NA,
            "tabular_test_recorder_id": pd.NA,
        }
        _write_final_test(
            best_row,
            test_result,
            tra_final,
            current_best_final,
            tra_validation=tra_validation,
            final_test_path=final_test_path,
        )
        return

    if xgb_protocol == "anchored":
        xgb_test_result = _run_direct_xgb_trial(
            SELECTED_TRIAL,
            stage_suffix=XGB_ANCHORED_TEST_STAGE_SUFFIX,
            window_key=window_key,
            result_suffix=result_suffix,
            train_segment=anchored_segments["final_train"],
            model_valid_segment=anchored_segments["final_model_valid"],
            predict_segment=anchored_segments["final_predict"],
            backtest_segment_key="test",
            train_label_mask_start=train_label_mask_start,
            train_label_mask_end=train_label_mask_end,
        )
        tabular_test_result_path = xgb_test_result["result_path"]
    else:
        xgb_test_result = _run_xgb_trial(
            SELECTED_TRIAL,
            eval_segment="test",
            stage_suffix=XGB_TEST_STAGE_SUFFIX,
            window_key=window_key,
            result_suffix=result_suffix,
            train_label_mask_start=train_label_mask_start,
            train_label_mask_end=train_label_mask_end,
        )
        tabular_test_result_path = str(
            _result_path_for_xgb(
                SELECTED_TRIAL,
                XGB_TEST_STAGE_SUFFIX,
                window_key=window_key,
                result_suffix=result_suffix,
            )
        )
    seq_test_pred, seq_test_label = _load_signal_from_cache(tra_final["test_cache_path"])
    tab_test_pred, tab_test_label = _load_signal_from_recorder(xgb_test_result["recorder_id"])
    test_alignment_report = _label_alignment_report(
        seq_test_pred,
        seq_test_label,
        tab_test_pred,
        tab_test_label,
    )
    seq_test_aligned, tab_test_aligned, test_label_aligned = _align_signals(
        seq_test_pred, seq_test_label, tab_test_pred, tab_test_label
    )
    fused_test_pred = _build_stage2_pred_from_row(seq_test_aligned, tab_test_aligned, best_row)
    test_cache_path = _write_fused_cache(
        compose_exp_suffix(f"{xgb_protocol}_{best_row['trial_name']}_final_test", result_suffix),
        fused_test_pred,
        test_label_aligned,
    )
    test_metrics = _evaluate_signal(
        recorder_name=f"{best_row['trial_name']}_final_test",
        pred=fused_test_pred,
        label=test_label_aligned,
        strategy_cfg=selected_strategy_cfg_test,
        experiment_name="alpha360_tra_fundamental_xgb_weight_refine_final_test_eval",
    )
    test_metrics.update(
        {
            "cache_path": str(test_cache_path),
            "tabular_test_result_path": tabular_test_result_path,
            "tabular_test_recorder_id": xgb_test_result["recorder_id"],
            "test_xgb_train_segment": xgb_test_result.get("train_segment", pd.NA),
            "test_xgb_model_valid_segment": xgb_test_result.get("model_valid_segment", pd.NA),
            "test_xgb_predict_segment": xgb_test_result.get("predict_segment", pd.NA),
            "test_alignment_common_rows": test_alignment_report["alignment_common_rows"],
            "test_alignment_common_days": test_alignment_report["alignment_common_days"],
            "test_tab_label_max_abs_diff_vs_stage1_label": test_alignment_report[
                "tab_label_max_abs_diff_vs_stage1_label"
            ],
            "test_tab_label_mean_abs_diff_vs_stage1_label": test_alignment_report[
                "tab_label_mean_abs_diff_vs_stage1_label"
            ],
        }
    )
    _write_final_test(
        best_row,
        test_metrics,
        tra_final,
        current_best_final,
        tra_validation=tra_validation,
        final_test_path=final_test_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local weight refine for Alpha360(TRA) + fundamental XGBoost fusion.")
    parser.add_argument("--window-key", default=DEFAULT_WINDOW_KEY, choices=["w1", "w2", "w3"])
    parser.add_argument("--result-suffix", default="", help="suffix appended to output summaries/caches")
    parser.add_argument("--train-label-mask-start", default=None, help="drop train labels on/after this date")
    parser.add_argument("--train-label-mask-end", default=None, help="drop train labels on/before this date")
    parser.add_argument("--tra-validation-best-path", default=None, help="override stage1 TRA validation summary path")
    parser.add_argument("--tra-final-test-path", default=None, help="override stage1 TRA final-test summary path")
    parser.add_argument(
        "--stage1-source",
        default="summary",
        choices=["summary", "rank-ensemble"],
        help="summary reads stage1 summaries; rank-ensemble rebuilds the authoritative 3-seed TRA rank ensemble",
    )
    parser.add_argument(
        "--tra-rank-validation-summary-path",
        default=str(TRA_BEST_VALIDATION_SUMMARY_PATH),
        help="old6 validation wrapper summary for --stage1-source rank-ensemble",
    )
    parser.add_argument(
        "--tra-rank-final-test-summary-path",
        default=str(TRA_BEST_FINAL_TEST_SUMMARY_PATH),
        help="old6 final-test wrapper summary for --stage1-source rank-ensemble",
    )
    parser.add_argument(
        "--tra-rank-repro-json-path",
        default=str(TRA_BEST_REPRO_JSON_PATH),
        help="authoritative single-strategy repro JSON for --stage1-source rank-ensemble",
    )
    parser.add_argument("--current-best-final-path", default=None, help="override archived current-best final-test summary path")
    parser.add_argument(
        "--xgb-protocol",
        default="anchored",
        choices=["anchored", "rolling"],
        help="anchored uses fixed pre-test splits; rolling preserves the previous rolling XGB path",
    )
    parser.add_argument(
        "--selection-policy",
        default="allow-baseline",
        choices=["best-fusion", "allow-baseline"],
        help="best-fusion chooses among XGB fusion weights only; allow-baseline may choose pure TRA",
    )
    parser.add_argument(
        "--stage2-aware-strategy-search",
        action="store_true",
        help="rerun strategy search on top of each stage2 fused signal instead of reusing the stage1 incumbent strategy",
    )
    parser.add_argument(
        "--stage2-signal-profile",
        default="linear",
        choices=["linear", "targeted", "filter", "all"],
        help="which stage2 signal recipes to validate",
    )
    parser.add_argument(
        "--validation-fast-mode",
        action="store_true",
        help="use in-memory validation backtests for large searches",
    )
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="run validation selection only and do not touch the final test set",
    )
    parser.add_argument("--strategy-hold-min", type=int, default=1, help="minimum hold_thresh allowed in stage2-aware strategy search")
    parser.add_argument("--strategy-risk-max", type=float, default=1.0, help="maximum risk_degree allowed in stage2-aware strategy search")
    args = parser.parse_args()
    main(
        window_key=args.window_key,
        result_suffix=args.result_suffix,
        train_label_mask_start=args.train_label_mask_start,
        train_label_mask_end=args.train_label_mask_end,
        tra_validation_best_path=args.tra_validation_best_path,
        tra_final_test_path=args.tra_final_test_path,
        stage1_source=args.stage1_source,
        tra_rank_validation_summary_path=args.tra_rank_validation_summary_path,
        tra_rank_final_test_summary_path=args.tra_rank_final_test_summary_path,
        tra_rank_repro_json_path=args.tra_rank_repro_json_path,
        current_best_final_path=args.current_best_final_path,
        xgb_protocol=args.xgb_protocol,
        selection_policy=args.selection_policy,
        stage2_aware_strategy_search=args.stage2_aware_strategy_search,
        stage2_signal_profile=args.stage2_signal_profile,
        validation_fast_mode=args.validation_fast_mode,
        skip_final_test=args.skip_final_test,
        strategy_hold_min=args.strategy_hold_min,
        strategy_risk_max=args.strategy_risk_max,
    )
