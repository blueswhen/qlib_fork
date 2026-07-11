from __future__ import annotations

import argparse
import json
from ast import literal_eval
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord

from extreme_event_mask import apply_train_label_mask, compose_exp_suffix, suffix_path
from rolling_tra_alpha360_latest import DEFAULT_HANDLER_KWARGS_EXTRA as TRA_DEFAULT_HANDLER_KWARGS_EXTRA
from rolling_tra_alpha360_latest import (
    PROVIDER_URI,
    WINDOWS,
    _metric_value,
    build_base_task,
    run,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WINDOW_KEY = "w1"
CURRENT_BASELINE_MANIFEST = BASE_DIR / "current_best_baseline_manifest.json"

ACCOUNT = 150000
TOPK = 5
BENCHMARK = "SH000300"
INSTRUMENTS = "csi300"
GPU_SLOTS = "0,1"

H3_LABEL = None
H5_LABEL = {"label": ["Ref($close, -6) / Ref($close, -1) - 1"]}

MODEL_TRIALS = [
    {
        "trial_name": "strict_tra_lstm_s240_h3",
        "model_key": "tra_lstm_base",
        "step": 240,
        "handler_kwargs_extra": H3_LABEL,
    },
    {
        "trial_name": "strict_tra_lstm_s240_h5",
        "model_key": "tra_lstm_base",
        "step": 240,
        "handler_kwargs_extra": H5_LABEL,
    },
    {
        "trial_name": "strict_tra_gru_s240_h3",
        "model_key": "tra_gru_base",
        "step": 240,
        "handler_kwargs_extra": H3_LABEL,
    },
    {
        "trial_name": "strict_tra_gru_s240_h5",
        "model_key": "tra_gru_base",
        "step": 240,
        "handler_kwargs_extra": H5_LABEL,
    },
]

STRATEGY_TRIALS = [
    {
        "trial_name": "base_topk",
        "strategy_class": "TopkDropoutStrategy",
        "strategy_module_path": "qlib.contrib.strategy",
        "n_drop": 1,
        "risk_degree": 0.70,
        "kwargs_extra": {},
    },
    {
        "trial_name": "prac_m000_h1_r07",
        "strategy_class": "PracticalTopkDropoutStrategy",
        "strategy_module_path": "qlib.contrib.strategy.custom_signal_strategy",
        "n_drop": 1,
        "risk_degree": 0.70,
        "kwargs_extra": {"score_margin": 0.0, "hold_thresh": 1, "slot_budget_ratio": 1.0},
    },
    {
        "trial_name": "prac_m025_h3_r07",
        "strategy_class": "PracticalTopkDropoutStrategy",
        "strategy_module_path": "qlib.contrib.strategy.custom_signal_strategy",
        "n_drop": 1,
        "risk_degree": 0.70,
        "kwargs_extra": {"score_margin": 0.0025, "hold_thresh": 3, "slot_budget_ratio": 1.0},
    },
    {
        "trial_name": "prac_m050_h3_r07",
        "strategy_class": "PracticalTopkDropoutStrategy",
        "strategy_module_path": "qlib.contrib.strategy.custom_signal_strategy",
        "n_drop": 1,
        "risk_degree": 0.70,
        "kwargs_extra": {"score_margin": 0.005, "hold_thresh": 3, "slot_budget_ratio": 1.0},
    },
    {
        "trial_name": "prac_m050_h5_r07",
        "strategy_class": "PracticalTopkDropoutStrategy",
        "strategy_module_path": "qlib.contrib.strategy.custom_signal_strategy",
        "n_drop": 1,
        "risk_degree": 0.70,
        "kwargs_extra": {"score_margin": 0.005, "hold_thresh": 5, "slot_budget_ratio": 1.0},
    },
    {
        "trial_name": "prac_m050_h3_r085",
        "strategy_class": "PracticalTopkDropoutStrategy",
        "strategy_module_path": "qlib.contrib.strategy.custom_signal_strategy",
        "n_drop": 1,
        "risk_degree": 0.85,
        "kwargs_extra": {"score_margin": 0.005, "hold_thresh": 3, "slot_budget_ratio": 1.0},
    },
    {
        "trial_name": "prac_m025_h1_r07_d2",
        "strategy_class": "PracticalTopkDropoutStrategy",
        "strategy_module_path": "qlib.contrib.strategy.custom_signal_strategy",
        "n_drop": 2,
        "risk_degree": 0.70,
        "kwargs_extra": {"score_margin": 0.0025, "hold_thresh": 1, "slot_budget_ratio": 1.0},
    },
]


def _objective(metrics: dict) -> float:
    ann = float(metrics["with_cost_ann_return"])
    ir = float(metrics["with_cost_ir"])
    mdd = float(metrics["with_cost_mdd"])
    return ann * 1000 + ir * 50 + mdd * 20


def _persist(rows: list[dict], path: Path):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    if "score" not in df.columns:
        df["score"] = pd.NA
    df.sort_values(by=["status", "score"], ascending=[True, False], na_position="last").to_csv(path, index=False)


def _load_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    return df.to_dict("records")


def _completed_model_trials(rows: list[dict]) -> set[str]:
    if not rows:
        return set()
    completed = set()
    expected = len(STRATEGY_TRIALS)
    by_trial = {}
    for row in rows:
        by_trial.setdefault(row["model_trial"], []).append(row)
    for model_trial, trial_rows in by_trial.items():
        if len(trial_rows) >= expected:
            completed.add(model_trial)
    return completed


def _load_signal(cache_path: str | Path):
    cache = pd.read_pickle(cache_path)
    pred = cache["pred"].sort_index()
    label = cache["label"].sort_index()
    common_index = pred.index.intersection(label.index)
    return pred.loc[common_index].sort_index(), label.loc[common_index].sort_index()


def _parse_optional_literal(value: str):
    if value in ("None", "", None):
        return None
    return literal_eval(value)


def _evaluate_strategy(
    model_trial: dict,
    strategy_trial: dict,
    pred: pd.DataFrame,
    label: pd.DataFrame,
    *,
    window_key: str,
) -> dict:
    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

    task = build_base_task(
        topk=TOPK,
        n_drop=strategy_trial["n_drop"],
        window_key=window_key,
        model_key=model_trial["model_key"],
        handler_class="Alpha360",
        handler_module_path="qlib.contrib.data.handler",
        account=ACCOUNT,
        risk_degree=strategy_trial["risk_degree"],
        benchmark=BENCHMARK,
        instruments=INSTRUMENTS,
        handler_kwargs_extra=model_trial["handler_kwargs_extra"],
        strategy_class=strategy_trial["strategy_class"],
        strategy_module_path=strategy_trial["strategy_module_path"],
        strategy_kwargs_extra=strategy_trial["kwargs_extra"],
        eval_segment="valid",
        backtest_segment="valid",
    )
    port_record = task["record"][2]

    recorder_name = f"{model_trial['trial_name']}__{strategy_trial['trial_name']}"
    with R.start(experiment_name="tra_strict_validation_strategy_eval", recorder_name=recorder_name):
        rec = R.get_recorder()
        rec.save_objects(**{"pred.pkl": pred, "label.pkl": label})
        SigAnaRecord(recorder=rec, ana_long_short=False, ann_scaler=252).generate()
        PortAnaRecord(recorder=rec, **port_record["kwargs"]).generate()
        metrics = rec.list_metrics()

    result = {
        "validation_recorder_id": rec.id,
        "IC": metrics.get("IC"),
        "Rank IC": metrics.get("Rank IC"),
        "with_cost_ann_return": _metric_value(metrics, rec.id, "1day.excess_return_with_cost.annualized_return"),
        "with_cost_ir": _metric_value(metrics, rec.id, "1day.excess_return_with_cost.information_ratio"),
        "with_cost_mdd": _metric_value(metrics, rec.id, "1day.excess_return_with_cost.max_drawdown"),
    }
    result["score"] = _objective(result)
    return result


def _current_baseline_test_metrics():
    if not CURRENT_BASELINE_MANIFEST.exists():
        return {}
    data = json.loads(CURRENT_BASELINE_MANIFEST.read_text(encoding="utf-8"))
    final_test = data.get("current_best_baseline", {}).get("final_test", {})
    if not isinstance(final_test, dict):
        return {}
    return {
        "with_cost_ann_return": final_test.get("with_cost_ann_return"),
        "with_cost_ir": final_test.get("with_cost_ir"),
        "with_cost_mdd": final_test.get("with_cost_mdd"),
    }


def _paths_for_window(window_key: str, result_suffix: str = ""):
    if window_key == "w1":
        return {
            "validation_result": suffix_path(BASE_DIR / "tra_strict_validation_results.csv", result_suffix),
            "validation_best": suffix_path(BASE_DIR / "tra_strict_validation_best.txt", result_suffix),
            "final_test": suffix_path(BASE_DIR / "tra_strict_final_test.txt", result_suffix),
        }
    return {
        "validation_result": suffix_path(BASE_DIR / f"tra_strict_{window_key}_validation_results.csv", result_suffix),
        "validation_best": suffix_path(BASE_DIR / f"tra_strict_{window_key}_validation_best.txt", result_suffix),
        "final_test": suffix_path(BASE_DIR / f"tra_strict_{window_key}_final_test.txt", result_suffix),
    }


def _write_best_validation(best_row: dict, *, window_key: str, validation_best_path: Path):
    window = WINDOWS[window_key]
    lines = [
        f"window_key={window_key}",
        f"train={window['train']}",
        f"valid={window['valid']}",
        f"test={window['test']}",
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
        f"validation_IC={best_row['IC']}",
        f"validation_Rank_IC={best_row['Rank IC']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"validation_score={best_row['score']}",
        f"validation_result_path={best_row['result_path']}",
        f"validation_cache_path={best_row['cache_path']}",
    ]
    validation_best_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_test(best_row: dict, test_result: dict, *, window_key: str, final_test_path: Path):
    window = WINDOWS[window_key]
    baseline = _current_baseline_test_metrics()
    baseline_ann = baseline.get("with_cost_ann_return")
    improves_baseline = None
    if baseline_ann is not None:
        improves_baseline = float(test_result["with_cost_ann_return"]) > float(baseline_ann)

    lines = [
        f"window_key={window_key}",
        f"train={window['train']}",
        f"valid={window['valid']}",
        f"test={window['test']}",
        f"selection_source=validation_only",
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
        f"validation_score={best_row['score']}",
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
        f"test_improves_current_baseline={improves_baseline}",
    ]
    if baseline:
        lines.extend(
            [
                f"current_baseline_with_cost_ann_return={baseline.get('with_cost_ann_return')}",
                f"current_baseline_with_cost_ir={baseline.get('with_cost_ir')}",
                f"current_baseline_with_cost_mdd={baseline.get('with_cost_mdd')}",
            ]
        )
    final_test_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(
    window_key: str = DEFAULT_WINDOW_KEY,
    skip_final_if_exists: bool = True,
    result_suffix: str = "",
    train_label_mask_start: str | None = None,
    train_label_mask_end: str | None = None,
    deterministic_runtime: bool = True,
):
    paths = _paths_for_window(window_key, result_suffix=result_suffix)
    validation_result_path = paths["validation_result"]
    validation_best_path = paths["validation_best"]
    final_test_path = paths["final_test"]
    rows = _load_existing_rows(validation_result_path)
    completed_trials = _completed_model_trials(rows)

    for model_trial in MODEL_TRIALS:
        if model_trial["trial_name"] in completed_trials:
            print(f"skip completed model trial {model_trial['trial_name']}")
            continue
        print(
            f"\n=== validation TRA model trial {model_trial['trial_name']}: "
            f"model={model_trial['model_key']} step={model_trial['step']} ==="
        )
        handler_kwargs_extra = apply_train_label_mask(
            model_trial["handler_kwargs_extra"],
            train_label_mask_start,
            train_label_mask_end,
            base_learn_processors=TRA_DEFAULT_HANDLER_KWARGS_EXTRA["learn_processors"],
        )
        model_result = run(
            step=model_trial["step"],
            topk=TOPK,
            n_drop=1,
            window_key=window_key,
            model_key=model_trial["model_key"],
            exp_suffix=compose_exp_suffix(model_trial["trial_name"], result_suffix),
            handler_class="Alpha360",
            handler_module_path="qlib.contrib.data.handler",
            account=ACCOUNT,
            risk_degree=0.70,
            provider_uri=PROVIDER_URI,
            benchmark=BENCHMARK,
            instruments=INSTRUMENTS,
            handler_kwargs_extra=handler_kwargs_extra,
            gpu_slots=GPU_SLOTS,
            eval_segment="valid",
            backtest_segment="valid",
            deterministic_runtime=deterministic_runtime,
        )

        pred, label = _load_signal(model_result["cache_path"])

        for strategy_trial in STRATEGY_TRIALS:
            row = {
                "model_trial": model_trial["trial_name"],
                "model_key": model_trial["model_key"],
                "step": model_trial["step"],
                "handler_kwargs_extra": str(handler_kwargs_extra),
                "strategy_trial": strategy_trial["trial_name"],
                "strategy_class": strategy_trial["strategy_class"],
                "strategy_module_path": strategy_trial["strategy_module_path"],
                "n_drop": strategy_trial["n_drop"],
                "risk_degree": strategy_trial["risk_degree"],
                "kwargs_extra": str(strategy_trial["kwargs_extra"]),
                "status": "running",
                "result_path": model_result["result_path"],
                "cache_path": model_result["cache_path"],
            }
            try:
                eval_result = _evaluate_strategy(
                    model_trial, strategy_trial, pred, label, window_key=window_key
                )
                row.update(eval_result)
                row["status"] = "success"
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
            rows.append(row)
            _persist(rows, validation_result_path)

    if not rows:
        raise RuntimeError("no validation rows produced")

    success_rows = [row for row in rows if row["status"] == "success"]
    if not success_rows:
        raise RuntimeError("all validation trials failed")

    best_row = max(success_rows, key=lambda item: item["score"])
    _write_best_validation(best_row, window_key=window_key, validation_best_path=validation_best_path)

    if skip_final_if_exists and final_test_path.exists():
        print(f"skip final test because {final_test_path} already exists")
        return

    test_result = run(
        step=int(best_row["step"]),
        topk=TOPK,
        n_drop=int(best_row["n_drop"]),
        window_key=window_key,
        model_key=best_row["model_key"],
        exp_suffix=compose_exp_suffix(f"{best_row['model_trial']}_final_test", result_suffix),
        handler_class="Alpha360",
        handler_module_path="qlib.contrib.data.handler",
        account=ACCOUNT,
        risk_degree=float(best_row["risk_degree"]),
        provider_uri=PROVIDER_URI,
        benchmark=BENCHMARK,
        instruments=INSTRUMENTS,
        handler_kwargs_extra=_parse_optional_literal(best_row["handler_kwargs_extra"]),
        strategy_class=best_row["strategy_class"],
        strategy_module_path=best_row["strategy_module_path"],
        strategy_kwargs_extra=literal_eval(best_row["kwargs_extra"]),
        gpu_slots=GPU_SLOTS,
        eval_segment="test",
        backtest_segment="test",
        deterministic_runtime=deterministic_runtime,
    )
    _write_final_test(best_row, test_result, window_key=window_key, final_test_path=final_test_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict validation/test selection for Alpha360 + TRA.")
    parser.add_argument("--window-key", default=DEFAULT_WINDOW_KEY, choices=sorted(WINDOWS.keys()))
    parser.add_argument("--force-final", action="store_true", help="rerun final test even if cached summary exists")
    parser.add_argument("--result-suffix", default="", help="suffix appended to output summaries/caches")
    parser.add_argument("--train-label-mask-start", default=None, help="drop train labels on/after this date")
    parser.add_argument("--train-label-mask-end", default=None, help="drop train labels on/before this date")
    parser.add_argument("--legacy-runtime", action="store_true", help="disable forced deterministic runtime to mimic older TRA runs")
    args = parser.parse_args()
    main(
        window_key=args.window_key,
        skip_final_if_exists=not args.force_final,
        result_suffix=args.result_suffix,
        train_label_mask_start=args.train_label_mask_start,
        train_label_mask_end=args.train_label_mask_end,
        deterministic_runtime=not args.legacy_runtime,
    )
