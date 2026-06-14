"""Triple-fusion: TRA (Alpha360) + Tech DNN (TechAlpha158A) + Fund DNN (rank-pct fundamentals).

Reuses already-trained signals (no retraining):
  * TRA: cached pred/label pickles from tra_strict_*.
  * Tech DNN: recorder ids stored in
    rolling_result_dnn_w1_dnn_wide_TechAlpha158A_step240_k5_d1_a150000_r07_dnn02_techA_wide_s240_tra_fusion_{valid,test}.txt
  * Fund DNN (rank-pct): recorder ids stored in
    rolling_result_dnn_w1_dnn_wide_TechAlpha158A3DFundamental_step240_k5_d1_a150000_r07_dnn03_techA3d_fund_rankpct_wide_s240_tra_fund_rankpct_fusion_{valid,test}.txt

Grid-searches (tech_weight, fund_weight) with seq_weight = 1 - tech_weight - fund_weight,
selects by validation score, then evaluates on test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

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

TRA_VALIDATION_BEST_PATH = BASE_DIR / "tra_strict_validation_best.txt"
TRA_FINAL_TEST_PATH = BASE_DIR / "tra_strict_final_test.txt"
CURRENT_BEST_FINAL_PATH = BASE_DIR / "alpha360_tra_alpha158_dnn_weight_refine_final_test.txt"

# Tech DNN single-branch results (already trained as part of the 2-way TRA+Tech fusion).
TECH_DNN_VALID_RESULT_PATH = (
    BASE_DIR
    / "rolling_result_dnn_w1_dnn_wide_TechAlpha158A_step240_k5_d1_a150000_r07_dnn02_techA_wide_s240_tra_fusion_valid.txt"
)
TECH_DNN_TEST_RESULT_PATH = (
    BASE_DIR
    / "rolling_result_dnn_w1_dnn_wide_TechAlpha158A_step240_k5_d1_a150000_r07_dnn02_techA_wide_s240_tra_fusion_test.txt"
)

# Fund DNN (rank-pct) single-branch results (already trained as part of the 2-way TRA+Fund fusion).
FUND_DNN_VALID_RESULT_PATH = (
    BASE_DIR
    / "rolling_result_dnn_w1_dnn_wide_TechAlpha158A3DFundamental_step240_k5_d1_a150000_r07"
    "_dnn03_techA3d_fund_rankpct_wide_s240_tra_fund_rankpct_fusion_valid.txt"
)
FUND_DNN_TEST_RESULT_PATH = (
    BASE_DIR
    / "rolling_result_dnn_w1_dnn_wide_TechAlpha158A3DFundamental_step240_k5_d1_a150000_r07"
    "_dnn03_techA3d_fund_rankpct_wide_s240_tra_fund_rankpct_fusion_test.txt"
)

VALIDATION_RESULTS_PATH = BASE_DIR / "alpha360_tra_tech_fund_triple_fusion_validation_results.csv"
VALIDATION_BEST_PATH = BASE_DIR / "alpha360_tra_tech_fund_triple_fusion_validation_best.txt"
FINAL_TEST_PATH = BASE_DIR / "alpha360_tra_tech_fund_triple_fusion_final_test.txt"

FUSION_MODE = "zscore"

# (tech_weight, fund_weight); seq_weight = 1 - tech - fund.
WEIGHT_GRID: list[tuple[float, float]] = [
    (0.00, 0.00),  # TRA only baseline
    (0.10, 0.00),  # TRA + Tech only (re-prove 2-way baseline)
    (0.00, 0.03),  # TRA + Fund only (re-prove 2-way baseline)
    (0.08, 0.02),
    (0.08, 0.03),
    (0.08, 0.05),
    (0.10, 0.02),
    (0.10, 0.03),
    (0.10, 0.05),
    (0.10, 0.08),
    (0.12, 0.02),
    (0.12, 0.03),
    (0.12, 0.05),
    (0.12, 0.08),
    (0.15, 0.02),
    (0.15, 0.05),
    (0.15, 0.10),
    (0.18, 0.05),
    (0.20, 0.05),
]


def _align_three(
    seq_pred, seq_label, tech_pred, tech_label, fund_pred, fund_label
):
    common = (
        seq_pred.index
        .intersection(tech_pred.index)
        .intersection(fund_pred.index)
        .intersection(seq_label.index)
        .intersection(tech_label.index)
        .intersection(fund_label.index)
        .sort_values()
    )
    return (
        seq_pred.loc[common].sort_index(),
        tech_pred.loc[common].sort_index(),
        fund_pred.loc[common].sort_index(),
        seq_label.loc[common].sort_index(),
    )


def _trial_name(seq_w: float, tech_w: float, fund_w: float) -> str:
    return (
        f"triple_seq{seq_w:.2f}_tech{tech_w:.2f}_fund{fund_w:.2f}".replace(".", "")
    )


def _write_validation_best(best_row: dict):
    strict_validation = _parse_summary_file(TRA_VALIDATION_BEST_PATH)
    lines = [
        f"window_key={strict_validation['window_key']}",
        f"train={strict_validation['train']}",
        f"valid={strict_validation['valid']}",
        f"test={strict_validation['test']}",
        f"selection_scope=triple_fusion_local_weight_search",
        f"base_model_trial={strict_validation['model_trial']}",
        f"base_strategy_trial={strict_validation['strategy_trial']}",
        f"tech_dnn_trial=dnn02_techA_wide_s240",
        f"fund_dnn_trial=dnn03_techA3d_fund_rankpct_wide_s240",
        f"fusion_mode={best_row['fusion_mode']}",
        f"seq_weight={best_row['seq_weight']}",
        f"tech_weight={best_row['tech_weight']}",
        f"fund_weight={best_row['fund_weight']}",
        f"validation_IC={best_row['IC']}",
        f"validation_Rank_IC={best_row['Rank IC']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"validation_score={best_row['score']}",
        f"validation_cache_path={best_row['validation_cache_path']}",
        f"validation_recorder_id={best_row['validation_recorder_id']}",
    ]
    VALIDATION_BEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_test(best_row: dict, test_result: dict, tra_test: dict, current_best_test: dict):
    strict_validation = _parse_summary_file(TRA_VALIDATION_BEST_PATH)
    improves_tra = float(test_result["with_cost_ann_return"]) > float(tra_test["test_with_cost_ann_return"])
    improves_current_best = (
        float(test_result["with_cost_ann_return"]) > float(current_best_test["test_with_cost_ann_return"])
    )
    lines = [
        f"window_key={strict_validation['window_key']}",
        f"train={strict_validation['train']}",
        f"valid={strict_validation['valid']}",
        f"test={strict_validation['test']}",
        f"selection_scope=validation_only",
        f"test_improves_tra_incumbent={improves_tra}",
        f"test_improves_current_best_baseline={improves_current_best}",
        f"base_model_trial={strict_validation['model_trial']}",
        f"base_strategy_trial={strict_validation['strategy_trial']}",
        f"tech_dnn_trial=dnn02_techA_wide_s240",
        f"fund_dnn_trial=dnn03_techA3d_fund_rankpct_wide_s240",
        f"fusion_mode={best_row['fusion_mode']}",
        f"seq_weight={best_row['seq_weight']}",
        f"tech_weight={best_row['tech_weight']}",
        f"fund_weight={best_row['fund_weight']}",
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
    ]
    FINAL_TEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    tra_validation = _parse_summary_file(TRA_VALIDATION_BEST_PATH)
    tra_final = _parse_summary_file(TRA_FINAL_TEST_PATH)
    current_best_final = _parse_summary_file(CURRENT_BEST_FINAL_PATH)
    tech_valid_meta = _parse_summary_file(TECH_DNN_VALID_RESULT_PATH)
    tech_test_meta = _parse_summary_file(TECH_DNN_TEST_RESULT_PATH)
    fund_valid_meta = _parse_summary_file(FUND_DNN_VALID_RESULT_PATH)
    fund_test_meta = _parse_summary_file(FUND_DNN_TEST_RESULT_PATH)

    strategy_cfg_valid = _load_alstm_strategy_cfg(tra_validation, eval_segment="valid")
    strategy_cfg_test = _load_alstm_strategy_cfg(tra_validation, eval_segment="test")

    # ------------------------- VALIDATION ---------------------------------------
    seq_valid_pred, seq_valid_label = _load_signal_from_cache(tra_validation["validation_cache_path"])
    tech_valid_pred, tech_valid_label = _load_signal_from_cache(tech_valid_meta["cache_path"])
    fund_valid_pred, fund_valid_label = _load_signal_from_cache(fund_valid_meta["cache_path"])

    seq_v, tech_v, fund_v, label_v = _align_three(
        seq_valid_pred, seq_valid_label,
        tech_valid_pred, tech_valid_label,
        fund_valid_pred, fund_valid_label,
    )
    seq_v_n = _normalize_scores(seq_v, FUSION_MODE)
    tech_v_n = _normalize_scores(tech_v, FUSION_MODE)
    fund_v_n = _normalize_scores(fund_v, FUSION_MODE)

    rows: list[dict] = []
    for tech_w, fund_w in WEIGHT_GRID:
        seq_w = round(1.0 - tech_w - fund_w, 4)
        if seq_w <= 0:
            continue
        trial = _trial_name(seq_w, tech_w, fund_w)
        row = {
            "trial_name": trial,
            "fusion_mode": FUSION_MODE,
            "seq_weight": seq_w,
            "tech_weight": tech_w,
            "fund_weight": fund_w,
            "status": "running",
        }
        try:
            fused = (
                seq_v_n.mul(seq_w)
                .add(tech_v_n.mul(tech_w), fill_value=0.0)
                .add(fund_v_n.mul(fund_w), fill_value=0.0)
            )
            cache_path = _write_fused_cache(trial, fused, label_v)
            metrics = _evaluate_signal(
                recorder_name=trial,
                pred=fused,
                label=label_v,
                strategy_cfg=strategy_cfg_valid,
                experiment_name="alpha360_tra_tech_fund_triple_fusion_validation_eval",
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
                }
            )
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)[:500]
        rows.append(row)
        _persist(rows, VALIDATION_RESULTS_PATH)

    success_rows = [r for r in rows if r["status"] == "success"]
    if not success_rows:
        raise RuntimeError("no successful triple-fusion rows")

    best_row = max(success_rows, key=lambda r: r["score"])
    _write_validation_best(best_row)

    # ------------------------- TEST ---------------------------------------------
    seq_test_pred, seq_test_label = _load_signal_from_cache(tra_final["test_cache_path"])
    tech_test_pred, tech_test_label = _load_signal_from_cache(tech_test_meta["cache_path"])
    fund_test_pred, fund_test_label = _load_signal_from_cache(fund_test_meta["cache_path"])

    seq_t, tech_t, fund_t, label_t = _align_three(
        seq_test_pred, seq_test_label,
        tech_test_pred, tech_test_label,
        fund_test_pred, fund_test_label,
    )
    seq_t_n = _normalize_scores(seq_t, FUSION_MODE)
    tech_t_n = _normalize_scores(tech_t, FUSION_MODE)
    fund_t_n = _normalize_scores(fund_t, FUSION_MODE)

    fused_test = (
        seq_t_n.mul(best_row["seq_weight"])
        .add(tech_t_n.mul(best_row["tech_weight"]), fill_value=0.0)
        .add(fund_t_n.mul(best_row["fund_weight"]), fill_value=0.0)
    )
    test_trial = best_row["trial_name"] + "_final_test"
    test_cache_path = _write_fused_cache(test_trial, fused_test, label_t)
    test_metrics = _evaluate_signal(
        recorder_name=test_trial,
        pred=fused_test,
        label=label_t,
        strategy_cfg=strategy_cfg_test,
        experiment_name="alpha360_tra_tech_fund_triple_fusion_final_test_eval",
    )
    test_metrics["cache_path"] = str(test_cache_path)
    _write_final_test(best_row, test_metrics, tra_final, current_best_final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triple fusion: TRA + Tech DNN + Fund DNN (rank-pct).")
    parser.parse_args()
    main()
