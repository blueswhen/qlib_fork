from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rolling_doubleensemble_alpha158_latest import run as run_de
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

TRA_VALIDATION_BEST_PATH = BASE_DIR / "tra_strict_validation_best.txt"
TRA_FINAL_TEST_PATH = BASE_DIR / "tra_strict_final_test.txt"
CURRENT_BEST_FINAL_PATH = BASE_DIR / "alpha360_tra_alpha158_xgb_strict_final_test.txt"

VALIDATION_RESULTS_PATH = BASE_DIR / "alpha360_tra_alpha158_de_strict_validation_results.csv"
VALIDATION_BEST_PATH = BASE_DIR / "alpha360_tra_alpha158_de_strict_validation_best.txt"
FINAL_TEST_PATH = BASE_DIR / "alpha360_tra_alpha158_de_strict_final_test.txt"

ACCOUNT = 150000
TOPK = 5
WINDOW_KEY = "w1"
BENCHMARK = "SH000300"
INSTRUMENTS = "csi300"
WORKER_SLOTS = "0,1"

FUSION_MODE = "zscore"
TAB_WEIGHTS = [0.05, 0.08, 0.10, 0.12, 0.15]

DE_TRIALS = [
    {
        "trial_name": "de01_techA_base_s240",
        "handler_class": "TechAlpha158A",
        "handler_module_path": "qlib.contrib.data.custom_handler",
        "handler_kwargs_extra": None,
        "model_key": "de_base",
        "step": 240,
    },
    {
        "trial_name": "de02_techA_safe_s240",
        "handler_class": "TechAlpha158A",
        "handler_module_path": "qlib.contrib.data.custom_handler",
        "handler_kwargs_extra": None,
        "model_key": "de_safe",
        "step": 240,
    },
]


def _objective(metrics: dict) -> float:
    ann = float(metrics["with_cost_ann_return"])
    ir = float(metrics["with_cost_ir"])
    mdd = float(metrics["with_cost_mdd"])
    return ann * 1000 + ir * 50 + mdd * 20


def _result_path_for_de(trial: dict, stage_suffix: str) -> Path:
    return BASE_DIR / (
        f"rolling_result_de_{WINDOW_KEY}_{trial['model_key']}_{trial['handler_class']}_step{int(trial['step'])}_k{TOPK}"
        f"_d1_a{ACCOUNT}_r07_{trial['trial_name']}_{stage_suffix}.txt"
    )


def _run_de_trial(trial: dict, eval_segment: str, stage_suffix: str) -> dict:
    result_path = _result_path_for_de(trial, stage_suffix)
    run_de(
        step=int(trial["step"]),
        topk=TOPK,
        n_drop=1,
        window_key=WINDOW_KEY,
        model_key=trial["model_key"],
        exp_suffix=f"{trial['trial_name']}_{stage_suffix}",
        handler_class=trial["handler_class"],
        handler_module_path=trial["handler_module_path"],
        handler_kwargs_extra=trial.get("handler_kwargs_extra"),
        account=ACCOUNT,
        risk_degree=0.70,
        benchmark=BENCHMARK,
        instruments=INSTRUMENTS,
        worker_slots=WORKER_SLOTS,
        eval_segment=eval_segment,
        backtest_segment=eval_segment,
    )
    return _parse_summary_file(result_path)


def _write_validation_best(best_row: dict):
    strict_validation = _parse_summary_file(TRA_VALIDATION_BEST_PATH)
    lines = [
        f"window_key={strict_validation['window_key']}",
        f"train={strict_validation['train']}",
        f"valid={strict_validation['valid']}",
        f"test={strict_validation['test']}",
        f"selection_scope=tabular_model_plus_local_weight_refine",
        f"base_model_trial={best_row['base_model_trial']}",
        f"base_strategy_trial={best_row['base_strategy_trial']}",
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
    VALIDATION_BEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_test(best_row: dict, test_result: dict, tra_test: dict, current_best_test: dict):
    strict_validation = _parse_summary_file(TRA_VALIDATION_BEST_PATH)
    improves_tra = float(test_result["with_cost_ann_return"]) > float(tra_test["test_with_cost_ann_return"])
    improves_current_best = float(test_result["with_cost_ann_return"]) > float(current_best_test["test_with_cost_ann_return"])
    lines = [
        f"window_key={strict_validation['window_key']}",
        f"train={strict_validation['train']}",
        f"valid={strict_validation['valid']}",
        f"test={strict_validation['test']}",
        f"selection_scope=validation_only",
        f"test_improves_tra_incumbent={improves_tra}",
        f"test_improves_current_best_baseline={improves_current_best}",
        f"base_model_trial={best_row['base_model_trial']}",
        f"base_strategy_trial={best_row['base_strategy_trial']}",
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
    FINAL_TEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    tra_validation = _parse_summary_file(TRA_VALIDATION_BEST_PATH)
    tra_final = _parse_summary_file(TRA_FINAL_TEST_PATH)
    current_best_final = _parse_summary_file(CURRENT_BEST_FINAL_PATH)
    strategy_cfg_valid = _load_alstm_strategy_cfg(tra_validation, eval_segment="valid")
    strategy_cfg_test = _load_alstm_strategy_cfg(tra_validation, eval_segment="test")

    seq_valid_pred, seq_valid_label = _load_signal_from_cache(tra_validation["validation_cache_path"])

    rows: list[dict] = [
        {
            "trial_name": "strict_de_fusion_baseline_tra_only",
            "base_model_trial": tra_validation["model_trial"],
            "base_strategy_trial": tra_validation["strategy_trial"],
            "tabular_trial": "none",
            "tabular_handler_class": "none",
            "tabular_model_key": "none",
            "fusion_mode": "baseline",
            "seq_weight": 1.0,
            "tab_weight": 0.0,
            "status": "success",
            "IC": float(tra_validation["validation_IC"]),
            "Rank IC": float(tra_validation["validation_Rank_IC"]),
            "with_cost_ann_return": float(tra_validation["validation_with_cost_ann_return"]),
            "with_cost_ir": float(tra_validation["validation_with_cost_ir"]),
            "with_cost_mdd": float(tra_validation["validation_with_cost_mdd"]),
            "score": float(tra_validation["validation_score"]),
            "validation_cache_path": str(tra_validation["validation_cache_path"]),
            "validation_recorder_id": pd.NA,
            "tabular_validation_result_path": pd.NA,
            "tabular_validation_recorder_id": pd.NA,
        }
    ]

    for trial in DE_TRIALS:
        de_validation_result = _run_de_trial(trial, eval_segment="valid", stage_suffix="strict_valid")
        tab_valid_pred, tab_valid_label = _load_signal_from_recorder(de_validation_result["recorder_id"])
        seq_pred_aligned, tab_pred_aligned, label_aligned = _align_signals(seq_valid_pred, seq_valid_label, tab_valid_pred, tab_valid_label)
        seq_norm = _normalize_scores(seq_pred_aligned, FUSION_MODE)
        tab_norm = _normalize_scores(tab_pred_aligned, FUSION_MODE)

        for tab_weight in TAB_WEIGHTS:
            seq_weight = round(1.0 - tab_weight, 2)
            row = {
                "trial_name": f"strict_de_fusion_{trial['trial_name']}_{FUSION_MODE}_tw{str(tab_weight).replace('.', '')}",
                "base_model_trial": tra_validation["model_trial"],
                "base_strategy_trial": tra_validation["strategy_trial"],
                "tabular_trial": trial["trial_name"],
                "tabular_handler_class": trial["handler_class"],
                "tabular_model_key": trial["model_key"],
                "fusion_mode": FUSION_MODE,
                "seq_weight": seq_weight,
                "tab_weight": tab_weight,
                "status": "running",
            }
            try:
                fused_pred = seq_norm.mul(seq_weight).add(tab_norm.mul(tab_weight), fill_value=0.0)
                validation_cache_path = _write_fused_cache(row["trial_name"], fused_pred, label_aligned)
                metrics = _evaluate_signal(
                    recorder_name=row["trial_name"],
                    pred=fused_pred,
                    label=label_aligned,
                    strategy_cfg=strategy_cfg_valid,
                    experiment_name="alpha360_tra_alpha158_de_strict_validation_eval",
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
                        "tabular_validation_result_path": str(_result_path_for_de(trial, "strict_valid")),
                        "tabular_validation_recorder_id": de_validation_result["recorder_id"],
                    }
                )
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = repr(exc)[:500]
            rows = [existing for existing in rows if existing.get("trial_name") != row["trial_name"]] + [row]
            _persist(rows, VALIDATION_RESULTS_PATH)

    success_rows = [row for row in rows if row["status"] == "success"]
    if not success_rows:
        raise RuntimeError("no successful de-fusion validation rows")

    best_row = max(success_rows, key=lambda item: item["score"])
    _write_validation_best(best_row)

    if best_row["tabular_trial"] == "none":
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
        _write_final_test(best_row, test_result, tra_final, current_best_final)
        return

    selected_trial = next(trial for trial in DE_TRIALS if trial["trial_name"] == best_row["tabular_trial"])
    de_test_result = _run_de_trial(selected_trial, eval_segment="test", stage_suffix="final_test")
    seq_test_pred, seq_test_label = _load_signal_from_cache(tra_final["test_cache_path"])
    tab_test_pred, tab_test_label = _load_signal_from_recorder(de_test_result["recorder_id"])
    seq_test_aligned, tab_test_aligned, test_label_aligned = _align_signals(seq_test_pred, seq_test_label, tab_test_pred, tab_test_label)
    seq_test_norm = _normalize_scores(seq_test_aligned, FUSION_MODE)
    tab_test_norm = _normalize_scores(tab_test_aligned, FUSION_MODE)
    fused_test_pred = seq_test_norm.mul(best_row["seq_weight"]).add(tab_test_norm.mul(best_row["tab_weight"]), fill_value=0.0)
    test_cache_path = _write_fused_cache(f"{best_row['trial_name']}_final_test", fused_test_pred, test_label_aligned)
    test_metrics = _evaluate_signal(
        recorder_name=f"{best_row['trial_name']}_final_test",
        pred=fused_test_pred,
        label=test_label_aligned,
        strategy_cfg=strategy_cfg_test,
        experiment_name="alpha360_tra_alpha158_de_strict_final_test_eval",
    )
    test_metrics.update(
        {
            "cache_path": str(test_cache_path),
            "tabular_test_result_path": str(_result_path_for_de(selected_trial, "final_test")),
            "tabular_test_recorder_id": de_test_result["recorder_id"],
        }
    )
    _write_final_test(best_row, test_metrics, tra_final, current_best_final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strict Alpha360(TRA) + Alpha158(DoubleEnsemble) fusion.")
    parser.parse_args()
    main()
