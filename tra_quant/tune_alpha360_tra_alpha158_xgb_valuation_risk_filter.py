"""Plan A on XGBoost baseline: valuation hard risk filter on top of TRA+XGB fusion.

XGBoost fusion baseline has worse MDD (test -21.56%) than DNN fusion (-15.34%).
The valuation trap filter primarily removes drawdown-driving names, so it is a
more natural fit for XGB. Same logic / same grid as the DNN version; only the
input baseline caches are swapped.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Reuse all helpers from the DNN version
from tune_alpha360_tra_alpha158_valuation_risk_filter import (
    EXPENSIVE_QS,
    QUALITY_QS,
    RISK_QS,
    FREEZE_FREQ,
    TOP_N,
    _apply_risk_filter,
    _flatten_columns,
    _prepare_components,
)
from tune_alpha360_alpha158_dnn_strict_fusion import (
    _evaluate_signal,
    _load_alstm_strategy_cfg,
    _load_signal_from_cache,
    _parse_summary_file,
    _persist,
    _write_fused_cache,
)


BASE_DIR = Path(__file__).resolve().parent

TRA_VALIDATION_BEST_PATH = BASE_DIR / "tra_strict_validation_best.txt"
# XGB fusion baseline summaries
BASELINE_VALIDATION_BEST_PATH = BASE_DIR / "alpha360_tra_alpha158_xgb_strict_validation_best.txt"
BASELINE_FINAL_TEST_PATH = BASE_DIR / "alpha360_tra_alpha158_xgb_strict_final_test.txt"

VALIDATION_RESULTS_PATH = BASE_DIR / "alpha360_tra_alpha158_xgb_valuation_risk_filter_validation_results.csv"
VALIDATION_BEST_PATH = BASE_DIR / "alpha360_tra_alpha158_xgb_valuation_risk_filter_validation_best.txt"
FINAL_TEST_PATH = BASE_DIR / "alpha360_tra_alpha158_xgb_valuation_risk_filter_final_test.txt"


def _suffix_path(path: Path, result_suffix: str) -> Path:
    if not result_suffix:
        return path
    return path.with_name(f"{path.stem}_{result_suffix}{path.suffix}")


def _write_validation_best(best_row: dict, baseline_validation_best_path: Path, validation_best_path: Path):
    base_valid = _parse_summary_file(baseline_validation_best_path)
    lines = [
        f"window_key={base_valid['window_key']}",
        f"train={base_valid['train']}",
        f"valid={base_valid['valid']}",
        f"test={base_valid['test']}",
        "selection_scope=valuation_risk_filter_search_xgb",
        f"base_model_trial={base_valid['base_model_trial']}",
        f"base_strategy_trial={base_valid['base_strategy_trial']}",
        f"base_tabular_trial={base_valid['tabular_trial']}",
        f"freeze_freq={best_row['freeze_freq']}",
        f"top_n={best_row['top_n']}",
        f"expensive_q={best_row['expensive_q']}",
        f"quality_q={best_row['quality_q']}",
        f"risk_q={best_row['risk_q']}",
        f"validation_IC={best_row['IC']}",
        f"validation_Rank_IC={best_row['Rank IC']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"validation_score={best_row['score']}",
        f"validation_cache_path={best_row['validation_cache_path']}",
        f"validation_recorder_id={best_row['validation_recorder_id']}",
        f"validation_demote_day_ratio={best_row.get('demote_day_ratio')}",
        f"validation_avg_demoted_per_day={best_row.get('avg_demoted_per_day')}",
        f"validation_demoted_total={best_row.get('demoted_total')}",
    ]
    validation_best_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_test(
    best_row: dict,
    test_result: dict,
    baseline_test: dict,
    *,
    baseline_validation_best_path: Path,
    final_test_path: Path,
):
    base_valid = _parse_summary_file(baseline_validation_best_path)
    improves_baseline = (
        float(test_result["with_cost_ann_return"]) > float(baseline_test["test_with_cost_ann_return"])
    )
    lines = [
        f"window_key={base_valid['window_key']}",
        f"train={base_valid['train']}",
        f"valid={base_valid['valid']}",
        f"test={base_valid['test']}",
        "selection_scope=validation_only_xgb",
        f"test_improves_baseline={improves_baseline}",
        f"base_model_trial={base_valid['base_model_trial']}",
        f"base_strategy_trial={base_valid['base_strategy_trial']}",
        f"base_tabular_trial={base_valid['tabular_trial']}",
        f"freeze_freq={best_row['freeze_freq']}",
        f"top_n={best_row['top_n']}",
        f"expensive_q={best_row['expensive_q']}",
        f"quality_q={best_row['quality_q']}",
        f"risk_q={best_row['risk_q']}",
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
        f"test_demote_day_ratio={test_result.get('demote_day_ratio')}",
        f"test_avg_demoted_per_day={test_result.get('avg_demoted_per_day')}",
        f"test_demoted_total={test_result.get('demoted_total')}",
        f"baseline_test_with_cost_ann_return={baseline_test['test_with_cost_ann_return']}",
        f"baseline_test_with_cost_ir={baseline_test['test_with_cost_ir']}",
        f"baseline_test_with_cost_mdd={baseline_test['test_with_cost_mdd']}",
    ]
    final_test_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args: argparse.Namespace):
    tra_validation_best_path = Path(args.tra_validation_best_path).resolve()
    baseline_validation_best_path = Path(args.baseline_validation_best_path).resolve()
    baseline_final_test_path = Path(args.baseline_final_test_path).resolve()
    validation_results_path = _suffix_path(VALIDATION_RESULTS_PATH, args.result_suffix)
    validation_best_path = _suffix_path(VALIDATION_BEST_PATH, args.result_suffix)
    final_test_path = _suffix_path(FINAL_TEST_PATH, args.result_suffix)

    tra_validation = _parse_summary_file(tra_validation_best_path)
    baseline_valid = _parse_summary_file(baseline_validation_best_path)
    baseline_test = _parse_summary_file(baseline_final_test_path)

    strategy_cfg_valid = _load_alstm_strategy_cfg(tra_validation, eval_segment="valid")
    strategy_cfg_test = _load_alstm_strategy_cfg(tra_validation, eval_segment="test")

    valid_pred, valid_label = _load_signal_from_cache(baseline_valid["validation_cache_path"])
    test_pred, test_label = _load_signal_from_cache(baseline_test["test_cache_path"])

    components = _prepare_components(valid_pred.index.union(test_pred.index))

    rows: list[dict] = [
        {
            "trial_name": "baseline_no_filter_xgb",
            "freeze_freq": "none",
            "top_n": TOP_N,
            "expensive_q": pd.NA,
            "quality_q": pd.NA,
            "risk_q": pd.NA,
            "status": "success",
            "IC": float(baseline_valid["validation_IC"]),
            "Rank IC": float(baseline_valid["validation_Rank_IC"]),
            "with_cost_ann_return": float(baseline_valid["validation_with_cost_ann_return"]),
            "with_cost_ir": float(baseline_valid["validation_with_cost_ir"]),
            "with_cost_mdd": float(baseline_valid["validation_with_cost_mdd"]),
            "score": float(baseline_valid["validation_score"]),
            "validation_cache_path": str(baseline_valid["validation_cache_path"]),
            "validation_recorder_id": pd.NA,
            "demote_day_ratio": 0.0,
            "avg_demoted_per_day": 0.0,
            "demoted_total": 0,
        }
    ]

    for expensive_q in EXPENSIVE_QS:
        for quality_q in QUALITY_QS:
            for risk_q in RISK_QS:
                trial_name = (
                    f"xgb_valuation_risk_filter_{FREEZE_FREQ}_n{TOP_N}"
                    f"_eq{int(expensive_q*100)}_qq{int(quality_q*100)}_rq{int(risk_q*100)}"
                )
                if args.result_suffix:
                    trial_name = f"{trial_name}_{args.result_suffix}"
                row = {
                    "trial_name": trial_name,
                    "freeze_freq": FREEZE_FREQ,
                    "top_n": TOP_N,
                    "expensive_q": expensive_q,
                    "quality_q": quality_q,
                    "risk_q": risk_q,
                    "status": "running",
                }
                try:
                    filtered_pred, stats = _apply_risk_filter(
                        base_pred=valid_pred,
                        components=components,
                        top_n=TOP_N,
                        expensive_q=expensive_q,
                        quality_q=quality_q,
                        risk_q=risk_q,
                        freeze_freq=FREEZE_FREQ,
                    )
                    common = filtered_pred.index.intersection(valid_label.index)
                    filtered_pred = filtered_pred.loc[common]
                    label = valid_label.loc[common]

                    cache_path = _write_fused_cache(trial_name, filtered_pred, label)
                    metrics = _evaluate_signal(
                        recorder_name=trial_name,
                        pred=filtered_pred,
                        label=label,
                        strategy_cfg=strategy_cfg_valid,
                        experiment_name="alpha360_tra_alpha158_xgb_valuation_risk_filter_validation_eval",
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
                            "validation_cache_path": str(cache_path),
                            "validation_recorder_id": metrics["recorder_id"],
                            "demote_day_ratio": stats["demote_day_ratio"],
                            "avg_demoted_per_day": stats["avg_demoted_per_day"],
                            "demoted_total": stats["demoted_total"],
                        }
                    )
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = repr(exc)[:500]

                rows = [r for r in rows if r.get("trial_name") != trial_name] + [row]
                _persist(rows, validation_results_path)

    success_rows = [row for row in rows if row["status"] == "success"]
    if not success_rows:
        raise RuntimeError("no successful xgb valuation-risk-filter rows")

    best_row = max(success_rows, key=lambda item: item["score"])
    _write_validation_best(best_row, baseline_validation_best_path, validation_best_path)

    if best_row["trial_name"] == "baseline_no_filter_xgb":
        test_result = {
            "cache_path": baseline_test["test_cache_path"],
            "recorder_id": baseline_test["test_recorder_id"],
            "IC": baseline_test["test_IC"],
            "Rank IC": baseline_test["test_Rank_IC"],
            "with_cost_ann_return": baseline_test["test_with_cost_ann_return"],
            "with_cost_ir": baseline_test["test_with_cost_ir"],
            "with_cost_mdd": baseline_test["test_with_cost_mdd"],
            "demote_day_ratio": 0.0,
            "avg_demoted_per_day": 0.0,
            "demoted_total": 0,
        }
        _write_final_test(
            best_row,
            test_result,
            baseline_test,
            baseline_validation_best_path=baseline_validation_best_path,
            final_test_path=final_test_path,
        )
        return

    filtered_test_pred, stats_test = _apply_risk_filter(
        base_pred=test_pred,
        components=components,
        top_n=int(best_row["top_n"]),
        expensive_q=float(best_row["expensive_q"]),
        quality_q=float(best_row["quality_q"]),
        risk_q=float(best_row["risk_q"]),
        freeze_freq=str(best_row["freeze_freq"]),
    )
    common_test = filtered_test_pred.index.intersection(test_label.index)
    filtered_test_pred = filtered_test_pred.loc[common_test]
    test_label = test_label.loc[common_test]

    test_cache_path = _write_fused_cache(f"{best_row['trial_name']}_final_test", filtered_test_pred, test_label)
    test_metrics = _evaluate_signal(
        recorder_name=f"{best_row['trial_name']}_final_test",
        pred=filtered_test_pred,
        label=test_label,
        strategy_cfg=strategy_cfg_test,
        experiment_name="alpha360_tra_alpha158_xgb_valuation_risk_filter_final_test_eval",
    )
    test_metrics["cache_path"] = str(test_cache_path)
    test_metrics.update(stats_test)
    _write_final_test(
        best_row,
        test_metrics,
        baseline_test,
        baseline_validation_best_path=baseline_validation_best_path,
        final_test_path=final_test_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valuation hard risk filter on TRA+XGB baseline.")
    parser.add_argument("--tra-validation-best-path", default=str(TRA_VALIDATION_BEST_PATH))
    parser.add_argument("--baseline-validation-best-path", default=str(BASELINE_VALIDATION_BEST_PATH))
    parser.add_argument("--baseline-final-test-path", default=str(BASELINE_FINAL_TEST_PATH))
    parser.add_argument("--result-suffix", default="")
    main(parser.parse_args())
