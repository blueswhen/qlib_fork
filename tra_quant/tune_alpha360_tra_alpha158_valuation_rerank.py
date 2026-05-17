from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
BASELINE_VALIDATION_BEST_PATH = BASE_DIR / "alpha360_tra_alpha158_dnn_weight_refine_validation_best.txt"
BASELINE_FINAL_TEST_PATH = BASE_DIR / "alpha360_tra_alpha158_dnn_weight_refine_final_test.txt"

VALIDATION_RESULTS_PATH = BASE_DIR / "alpha360_tra_alpha158_valuation_rerank_validation_results.csv"
VALIDATION_BEST_PATH = BASE_DIR / "alpha360_tra_alpha158_valuation_rerank_validation_best.txt"
FINAL_TEST_PATH = BASE_DIR / "alpha360_tra_alpha158_valuation_rerank_final_test.txt"

FUNDAMENTAL_RAW_PATH = (
    "/home/blueswhen/DL/qlib/tushare/processed/training/csi300_daily_fundamental_features_qlib.parquet"
)

CANDIDATE_TOPN_LIST = [10, 15, 20]
ALPHAS = [0.0]
BETAS = [0.005, 0.01, 0.02]
FREEZE_FREQS = ["month"]

EXPENSIVE_Q = 0.85
POOR_QUALITY_Q = 0.30
HIGH_RISK_Q = 0.70


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        out = df.copy()
        out.columns = [str(col[-1]) for col in out.columns]
        return out
    return df


def _rank_pct(series: pd.Series) -> pd.Series:
    return series.rank(pct=True)


def _compute_components_for_day(day_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "pe_ttm",
        "pb",
        "ps_ttm",
        "total_mv",
        "fi_q_sales_yoy",
        "fi_or_yoy",
        "fi_roe_dt",
        "fi_netprofit_margin",
        "fi_ocf_yoy",
        "fi_debt_to_assets",
    ]
    x = day_df.reindex(columns=cols).copy()

    y = np.log(x["pe_ttm"].where(x["pe_ttm"] > 0.1, np.nan))
    y = y.fillna(np.log(x["pb"].where(x["pb"] > 0.01, np.nan)))
    y = y.fillna(np.log(x["ps_ttm"].where(x["ps_ttm"] > 0.01, np.nan)))

    for c in cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan)
        med = x[c].median(skipna=True)
        if pd.isna(med):
            med = 0.0
        x[c] = x[c].fillna(med)

    reg_cols = [
        "total_mv",
        "fi_q_sales_yoy",
        "fi_or_yoy",
        "fi_roe_dt",
        "fi_netprofit_margin",
        "fi_ocf_yoy",
        "fi_debt_to_assets",
    ]
    reg_x = x[reg_cols].astype(float)
    reg_x = (reg_x - reg_x.mean()) / reg_x.std(ddof=0).replace(0, 1)

    valid = y.notna() & reg_x.notna().all(axis=1)
    resid = pd.Series(np.nan, index=day_df.index, dtype=float)

    if valid.sum() >= 30:
        X = np.column_stack([np.ones(valid.sum()), reg_x.loc[valid].values])
        yv = y.loc[valid].values
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid.loc[valid] = yv - (X @ beta)

    undervalue = _rank_pct(-resid).fillna(0.5)
    expensive = _rank_pct(resid).fillna(0.5)
    quality = pd.concat(
        [
            _rank_pct(x["fi_roe_dt"]),
            _rank_pct(x["fi_netprofit_margin"]),
            _rank_pct(x["fi_ocf_yoy"]),
        ],
        axis=1,
    ).mean(axis=1).fillna(0.5)
    risk = _rank_pct(x["fi_debt_to_assets"]).fillna(0.5)

    trap = (
        (expensive > EXPENSIVE_Q)
        & (quality < POOR_QUALITY_Q)
        & (risk > HIGH_RISK_Q)
    )

    return pd.DataFrame(
        {
            "undervalue": undervalue,
            "expensive": expensive,
            "quality": quality,
            "risk": risk,
            "trap": trap.astype(float),
        },
        index=day_df.index,
    )


def _compute_components(fund_df: pd.DataFrame) -> pd.DataFrame:
    return fund_df.groupby(level="datetime", group_keys=False).apply(_compute_components_for_day)


def _freeze_series(component: pd.Series, freeze_freq: str) -> pd.Series:
    wide = component.unstack("instrument").sort_index()
    dates = wide.index
    periods = dates.to_period("M" if freeze_freq == "month" else "Q")
    out = wide.copy()
    for period in periods.unique():
        period_dates = dates[periods == period]
        anchor = period_dates.min()
        out.loc[period_dates] = wide.loc[anchor].values
    frozen = out.stack(dropna=False)
    frozen.index.names = ["datetime", "instrument"]
    return frozen.sort_index()


def _apply_candidate_rerank(
    base_pred: pd.DataFrame,
    components: pd.DataFrame,
    candidate_topn: int,
    alpha: float,
    beta: float,
    freeze_freq: str,
) -> tuple[pd.DataFrame, dict]:
    und = _freeze_series(components["undervalue"], freeze_freq=freeze_freq)
    trap = _freeze_series(components["trap"], freeze_freq=freeze_freq)

    common = base_pred.index.intersection(und.index).intersection(trap.index)
    pred = base_pred.loc[common].copy()
    und = und.loc[common].astype(float)
    trap = trap.loc[common].astype(float)

    score = pred["score"].unstack("instrument")
    und_w = und.unstack("instrument").reindex_like(score)
    trap_w = trap.unstack("instrument").reindex_like(score)

    out_rows = []
    applied_counts = []
    for dt in score.index:
        s = score.loc[dt].astype(float).copy()
        u = und_w.loc[dt]
        t = trap_w.loc[dt]

        cand_idx = s.dropna().nlargest(candidate_topn).index
        if len(cand_idx) == 0:
            out_rows.append(s)
            applied_counts.append(0)
            continue

        u_centered = (u.loc[cand_idx].fillna(0.5) - 0.5)
        trap_penalty = t.loc[cand_idx].fillna(0.0)
        s.loc[cand_idx] = s.loc[cand_idx] + alpha * u_centered - beta * trap_penalty

        out_rows.append(s)
        applied_counts.append(len(cand_idx))

    out = pd.DataFrame(out_rows, index=score.index)
    reranked = out.stack(dropna=False).to_frame("score")
    reranked.index.names = ["datetime", "instrument"]

    stats = {
        "avg_candidates": float(np.mean(applied_counts)),
        "trap_ratio": float(np.nanmean(trap.values)),
        "avg_abs_alpha_term": float(np.nanmean(np.abs(alpha * (und.values - 0.5)))),
    }
    return reranked.sort_index(), stats


def _prepare_components(index_ref: pd.Index) -> pd.DataFrame:
    fund_df = pd.read_parquet(FUNDAMENTAL_RAW_PATH)
    fund_df = _flatten_columns(fund_df)
    fund_df = fund_df.loc[fund_df.index.intersection(index_ref)].sort_index()
    return _compute_components(fund_df)


def _write_validation_best(best_row: dict):
    base_valid = _parse_summary_file(BASELINE_VALIDATION_BEST_PATH)
    lines = [
        f"window_key={base_valid['window_key']}",
        f"train={base_valid['train']}",
        f"valid={base_valid['valid']}",
        f"test={base_valid['test']}",
        "selection_scope=valuation_candidate_rerank_search",
        f"base_model_trial={base_valid['base_model_trial']}",
        f"base_strategy_trial={base_valid['base_strategy_trial']}",
        f"base_tabular_trial={base_valid['tabular_trial']}",
        f"freeze_freq={best_row['freeze_freq']}",
        f"candidate_topn={best_row['candidate_topn']}",
        f"alpha={best_row['alpha']}",
        f"beta={best_row['beta']}",
        f"validation_IC={best_row['IC']}",
        f"validation_Rank_IC={best_row['Rank IC']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"validation_score={best_row['score']}",
        f"validation_cache_path={best_row['validation_cache_path']}",
        f"validation_recorder_id={best_row['validation_recorder_id']}",
        f"validation_avg_candidates={best_row.get('avg_candidates')}",
        f"validation_trap_ratio={best_row.get('trap_ratio')}",
        f"validation_avg_abs_alpha_term={best_row.get('avg_abs_alpha_term')}",
    ]
    VALIDATION_BEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_test(best_row: dict, test_result: dict, baseline_test: dict):
    base_valid = _parse_summary_file(BASELINE_VALIDATION_BEST_PATH)
    improves_baseline = (
        float(test_result["with_cost_ann_return"]) > float(baseline_test["test_with_cost_ann_return"])
    )
    lines = [
        f"window_key={base_valid['window_key']}",
        f"train={base_valid['train']}",
        f"valid={base_valid['valid']}",
        f"test={base_valid['test']}",
        "selection_scope=validation_only",
        f"test_improves_baseline={improves_baseline}",
        f"base_model_trial={base_valid['base_model_trial']}",
        f"base_strategy_trial={base_valid['base_strategy_trial']}",
        f"base_tabular_trial={base_valid['tabular_trial']}",
        f"freeze_freq={best_row['freeze_freq']}",
        f"candidate_topn={best_row['candidate_topn']}",
        f"alpha={best_row['alpha']}",
        f"beta={best_row['beta']}",
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
        f"test_avg_candidates={test_result.get('avg_candidates')}",
        f"test_trap_ratio={test_result.get('trap_ratio')}",
        f"test_avg_abs_alpha_term={test_result.get('avg_abs_alpha_term')}",
        f"baseline_test_with_cost_ann_return={baseline_test['test_with_cost_ann_return']}",
        f"baseline_test_with_cost_ir={baseline_test['test_with_cost_ir']}",
        f"baseline_test_with_cost_mdd={baseline_test['test_with_cost_mdd']}",
    ]
    FINAL_TEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    tra_validation = _parse_summary_file(TRA_VALIDATION_BEST_PATH)
    baseline_valid = _parse_summary_file(BASELINE_VALIDATION_BEST_PATH)
    baseline_test = _parse_summary_file(BASELINE_FINAL_TEST_PATH)

    strategy_cfg_valid = _load_alstm_strategy_cfg(tra_validation, eval_segment="valid")
    strategy_cfg_test = _load_alstm_strategy_cfg(tra_validation, eval_segment="test")

    valid_pred, valid_label = _load_signal_from_cache(baseline_valid["validation_cache_path"])
    test_pred, test_label = _load_signal_from_cache(baseline_test["test_cache_path"])

    components = _prepare_components(valid_pred.index.union(test_pred.index))

    rows: list[dict] = [
        {
            "trial_name": "baseline_no_rerank",
            "freeze_freq": "none",
            "candidate_topn": 0,
            "alpha": 0.0,
            "beta": 0.0,
            "status": "success",
            "IC": float(baseline_valid["validation_IC"]),
            "Rank IC": float(baseline_valid["validation_Rank_IC"]),
            "with_cost_ann_return": float(baseline_valid["validation_with_cost_ann_return"]),
            "with_cost_ir": float(baseline_valid["validation_with_cost_ir"]),
            "with_cost_mdd": float(baseline_valid["validation_with_cost_mdd"]),
            "score": float(baseline_valid["validation_score"]),
            "validation_cache_path": str(baseline_valid["validation_cache_path"]),
            "validation_recorder_id": pd.NA,
            "avg_candidates": 0.0,
            "trap_ratio": pd.NA,
            "avg_abs_alpha_term": 0.0,
        }
    ]

    for freeze_freq in FREEZE_FREQS:
        for candidate_topn in CANDIDATE_TOPN_LIST:
            for alpha in ALPHAS:
                for beta in BETAS:
                    trial_name = f"valuation_rerank_{freeze_freq}_n{candidate_topn}_a{str(alpha).replace('.', '')}_b{str(beta).replace('.', '')}"
                    row = {
                        "trial_name": trial_name,
                        "freeze_freq": freeze_freq,
                        "candidate_topn": candidate_topn,
                        "alpha": alpha,
                        "beta": beta,
                        "status": "running",
                    }
                    try:
                        reranked_pred, stats = _apply_candidate_rerank(
                            base_pred=valid_pred,
                            components=components,
                            candidate_topn=candidate_topn,
                            alpha=alpha,
                            beta=beta,
                            freeze_freq=freeze_freq,
                        )
                        common = reranked_pred.index.intersection(valid_label.index)
                        reranked_pred = reranked_pred.loc[common]
                        label = valid_label.loc[common]

                        cache_path = _write_fused_cache(trial_name, reranked_pred, label)
                        metrics = _evaluate_signal(
                            recorder_name=trial_name,
                            pred=reranked_pred,
                            label=label,
                            strategy_cfg=strategy_cfg_valid,
                            experiment_name="alpha360_tra_alpha158_valuation_rerank_validation_eval",
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
                                "avg_candidates": stats["avg_candidates"],
                                "trap_ratio": stats["trap_ratio"],
                                "avg_abs_alpha_term": stats["avg_abs_alpha_term"],
                            }
                        )
                    except Exception as exc:
                        row["status"] = "failed"
                        row["error"] = repr(exc)[:500]

                    rows = [existing for existing in rows if existing.get("trial_name") != trial_name] + [row]
                    _persist(rows, VALIDATION_RESULTS_PATH)

    success_rows = [row for row in rows if row["status"] == "success"]
    if not success_rows:
        raise RuntimeError("no successful valuation-rerank rows")

    best_row = max(success_rows, key=lambda item: item["score"])
    _write_validation_best(best_row)

    if best_row["trial_name"] == "baseline_no_rerank":
        test_result = {
            "cache_path": baseline_test["test_cache_path"],
            "recorder_id": baseline_test["test_recorder_id"],
            "IC": baseline_test["test_IC"],
            "Rank IC": baseline_test["test_Rank_IC"],
            "with_cost_ann_return": baseline_test["test_with_cost_ann_return"],
            "with_cost_ir": baseline_test["test_with_cost_ir"],
            "with_cost_mdd": baseline_test["test_with_cost_mdd"],
            "avg_candidates": 0.0,
            "trap_ratio": pd.NA,
            "avg_abs_alpha_term": 0.0,
        }
        _write_final_test(best_row, test_result, baseline_test)
        return

    reranked_test_pred, stats_test = _apply_candidate_rerank(
        base_pred=test_pred,
        components=components,
        candidate_topn=int(best_row["candidate_topn"]),
        alpha=float(best_row["alpha"]),
        beta=float(best_row["beta"]),
        freeze_freq=str(best_row["freeze_freq"]),
    )
    common_test = reranked_test_pred.index.intersection(test_label.index)
    reranked_test_pred = reranked_test_pred.loc[common_test]
    test_label = test_label.loc[common_test]

    test_cache_path = _write_fused_cache(f"{best_row['trial_name']}_final_test", reranked_test_pred, test_label)
    test_metrics = _evaluate_signal(
        recorder_name=f"{best_row['trial_name']}_final_test",
        pred=reranked_test_pred,
        label=test_label,
        strategy_cfg=strategy_cfg_test,
        experiment_name="alpha360_tra_alpha158_valuation_rerank_final_test_eval",
    )
    test_metrics["cache_path"] = str(test_cache_path)
    test_metrics.update(stats_test)
    _write_final_test(best_row, test_metrics, baseline_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valuation residual candidate-pool rerank over TRA+Tech baseline.")
    parser.parse_args()
    main()
