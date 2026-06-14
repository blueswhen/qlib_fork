"""Route 1: valuation-based hard risk filter on baseline top-N.

For each trading day:
  1. Take baseline score for all instruments.
  2. Identify the current top-N candidates (N = trade pool, e.g. 10).
  3. Within those top-N, detect "extreme valuation traps":
        expensive percentile > expensive_q
        AND quality percentile < quality_q
        AND leverage / risk percentile > risk_q
  4. Hard-demote trap stocks (score := global_min - 1) so that the next
     non-trap candidates back-fill the top-N naturally.

This is NOT a soft penalty and NOT a candidate-pool rerank. It only removes
clearly bad names; everything else is identical to baseline.
"""
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

VALIDATION_RESULTS_PATH = BASE_DIR / "alpha360_tra_alpha158_valuation_risk_filter_v2_validation_results.csv"
VALIDATION_BEST_PATH = BASE_DIR / "alpha360_tra_alpha158_valuation_risk_filter_v2_validation_best.txt"
FINAL_TEST_PATH = BASE_DIR / "alpha360_tra_alpha158_valuation_risk_filter_v2_final_test.txt"

FUNDAMENTAL_RAW_PATH = (
    "/home/blueswhen/DL/qlib/tushare/processed/training/csi300_daily_fundamental_features_qlib.parquet"
)

# Trade pool to filter inside.
TOP_N = 10
# Monthly freeze of fundamentals to avoid look-ahead noise.
FREEZE_FREQ = "month"

# Grid over trap thresholds. Relaxed (plan A): allow more trap hits in test period.
EXPENSIVE_QS = [0.80, 0.85, 0.90]
QUALITY_QS = [0.30, 0.35, 0.40]
RISK_QS = [0.65, 0.70, 0.80]


def _suffix_path(path: Path, result_suffix: str) -> Path:
    if not result_suffix:
        return path
    return path.with_name(f"{path.stem}_{result_suffix}{path.suffix}")


def _summary_value(summary: dict, key: str, fallback_key: str | None = None, default=pd.NA):
    if key in summary:
        return summary[key]
    if fallback_key is not None and fallback_key in summary:
        return summary[fallback_key]
    return default


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        out = df.copy()
        out.columns = [str(col[-1]) for col in out.columns]
        return out
    return df


def _rank_pct(series: pd.Series) -> pd.Series:
    return series.rank(pct=True)


def _components_for_day(day_df: pd.DataFrame) -> pd.DataFrame:
    """Compute expensive / quality / risk percentiles for a single date.

    expensive: high = overvalued (residual of log valuation vs quality+size).
    quality:   high = good profitability / cash-flow.
    risk:      high = high leverage.
    """
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

    return pd.DataFrame(
        {"expensive": expensive, "quality": quality, "risk": risk},
        index=day_df.index,
    )


def _compute_components(fund_df: pd.DataFrame) -> pd.DataFrame:
    return fund_df.groupby(level="datetime", group_keys=False).apply(_components_for_day)


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


def _apply_risk_filter(
    base_pred: pd.DataFrame,
    components: pd.DataFrame,
    top_n: int,
    expensive_q: float,
    quality_q: float,
    risk_q: float,
    freeze_freq: str,
) -> tuple[pd.DataFrame, dict]:
    exp_f = _freeze_series(components["expensive"], freeze_freq=freeze_freq)
    qua_f = _freeze_series(components["quality"], freeze_freq=freeze_freq)
    rsk_f = _freeze_series(components["risk"], freeze_freq=freeze_freq)

    common = base_pred.index.intersection(exp_f.index).intersection(qua_f.index).intersection(rsk_f.index)
    pred = base_pred.loc[common].copy()

    score = pred["score"].unstack("instrument").astype(float)
    expensive = exp_f.loc[common].unstack("instrument").reindex_like(score)
    quality = qua_f.loc[common].unstack("instrument").reindex_like(score)
    risk = rsk_f.loc[common].unstack("instrument").reindex_like(score)

    days_total = 0
    days_with_demote = 0
    demoted_total = 0

    out = score.copy()
    for dt in score.index:
        s = score.loc[dt]
        valid_idx = s.dropna().index
        if len(valid_idx) == 0:
            continue
        days_total += 1

        topn_idx = s.loc[valid_idx].nlargest(top_n).index

        e = expensive.loc[dt, topn_idx]
        q = quality.loc[dt, topn_idx]
        r = risk.loc[dt, topn_idx]

        trap_mask = (
            (e > expensive_q)
            & (q < quality_q)
            & (r > risk_q)
        )
        trap_mask = trap_mask.fillna(False)
        trap_ids = topn_idx[trap_mask.values]

        if len(trap_ids) > 0:
            global_min = float(s.loc[valid_idx].min())
            out.loc[dt, trap_ids] = global_min - 1.0
            days_with_demote += 1
            demoted_total += int(len(trap_ids))

    filtered = out.stack(dropna=False).to_frame("score")
    filtered.index.names = ["datetime", "instrument"]

    stats = {
        "days_total": days_total,
        "days_with_demote": days_with_demote,
        "demote_day_ratio": (days_with_demote / days_total) if days_total else 0.0,
        "avg_demoted_per_day": (demoted_total / days_total) if days_total else 0.0,
        "demoted_total": demoted_total,
    }
    return filtered.sort_index(), stats


def _prepare_components(index_ref: pd.Index) -> pd.DataFrame:
    fund_df = pd.read_parquet(FUNDAMENTAL_RAW_PATH)
    fund_df = _flatten_columns(fund_df)
    fund_df = fund_df.loc[fund_df.index.intersection(index_ref)].sort_index()
    return _compute_components(fund_df)


def _write_validation_best(best_row: dict, baseline_validation_best_path: Path, validation_best_path: Path):
    base_valid = _parse_summary_file(baseline_validation_best_path)
    lines = [
        f"window_key={base_valid['window_key']}",
        f"train={base_valid['train']}",
        f"valid={base_valid['valid']}",
        f"test={base_valid['test']}",
        "selection_scope=valuation_risk_filter_search",
        f"base_model_trial={_summary_value(base_valid, 'base_model_trial', 'model_trial')}",
        f"base_strategy_trial={_summary_value(base_valid, 'base_strategy_trial', 'strategy_trial')}",
        f"base_tabular_trial={_summary_value(base_valid, 'base_tabular_trial', 'tabular_trial', 'none')}",
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
        "selection_scope=validation_only",
        f"test_improves_baseline={improves_baseline}",
        f"base_model_trial={_summary_value(base_valid, 'base_model_trial', 'model_trial')}",
        f"base_strategy_trial={_summary_value(base_valid, 'base_strategy_trial', 'strategy_trial')}",
        f"base_tabular_trial={_summary_value(base_valid, 'base_tabular_trial', 'tabular_trial', 'none')}",
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
            "trial_name": "baseline_no_filter",
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
                    f"valuation_risk_filter_{FREEZE_FREQ}_n{TOP_N}"
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
                        experiment_name="alpha360_tra_alpha158_valuation_risk_filter_validation_eval",
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
        raise RuntimeError("no successful valuation-risk-filter rows")

    best_row = max(success_rows, key=lambda item: item["score"])
    _write_validation_best(best_row, baseline_validation_best_path, validation_best_path)

    if best_row["trial_name"] == "baseline_no_filter":
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
        experiment_name="alpha360_tra_alpha158_valuation_risk_filter_final_test_eval",
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
    parser = argparse.ArgumentParser(description="Valuation hard risk filter on baseline top-N.")
    parser.add_argument("--tra-validation-best-path", default=str(TRA_VALIDATION_BEST_PATH))
    parser.add_argument("--baseline-validation-best-path", default=str(BASELINE_VALIDATION_BEST_PATH))
    parser.add_argument("--baseline-final-test-path", default=str(BASELINE_FINAL_TEST_PATH))
    parser.add_argument("--result-suffix", default="")
    main(parser.parse_args())
