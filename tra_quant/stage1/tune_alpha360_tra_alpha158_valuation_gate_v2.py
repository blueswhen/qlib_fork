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

VALIDATION_RESULTS_PATH = BASE_DIR / "alpha360_tra_alpha158_valuation_gate_v2_validation_results.csv"
VALIDATION_BEST_PATH = BASE_DIR / "alpha360_tra_alpha158_valuation_gate_v2_validation_best.txt"
FINAL_TEST_PATH = BASE_DIR / "alpha360_tra_alpha158_valuation_gate_v2_final_test.txt"

FUNDAMENTAL_RAW_PATH = (
    "/home/blueswhen/DL/qlib/tushare/processed/training/csi300_daily_fundamental_features_qlib.parquet"
)

FREEZE_FREQS = ["month", "quarter"]
LAMBDAS = [0.005, 0.010, 0.020]
TAUS = [0.20, 0.30, 0.40]
TRAP_MODES = [False]
TRAP_EXTRA_PENALTY = 0.20


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        out = df.copy()
        out.columns = [str(col[-1]) for col in out.columns]
        return out
    return df


def _rank_pct(series: pd.Series) -> pd.Series:
    return series.rank(pct=True)


def _compute_gate_components_for_day(day_df: pd.DataFrame) -> pd.DataFrame:
    # Required columns are selected to express fair-value residual + quality trap.
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
    # Fallback where PE is unavailable/invalid.
    y = y.fillna(np.log(x["pb"].where(x["pb"] > 0.01, np.nan)))
    y = y.fillna(np.log(x["ps_ttm"].where(x["ps_ttm"] > 0.01, np.nan)))

    # Cross-sectional fill keeps sample size stable for daily regression.
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
        yhat = X @ beta
        resid.loc[valid] = yv - yhat

    undervalue = _rank_pct(-resid).fillna(0.5)
    quality = pd.concat(
        [
            _rank_pct(x["fi_roe_dt"]),
            _rank_pct(x["fi_netprofit_margin"]),
            _rank_pct(x["fi_ocf_yoy"]),
        ],
        axis=1,
    ).mean(axis=1).fillna(0.5)
    risk = _rank_pct(x["fi_debt_to_assets"]).fillna(0.5)

    gate_score = (0.55 * undervalue + 0.35 * quality - 0.20 * risk).clip(0.0, 1.0)
    trap = (undervalue > 0.80) & (quality < 0.30) & (risk > 0.70)

    return pd.DataFrame(
        {
            "gate_score": gate_score,
            "undervalue": undervalue,
            "quality": quality,
            "risk": risk,
            "trap": trap.astype(float),
        },
        index=day_df.index,
    )


def _compute_gate_components(fund_df: pd.DataFrame) -> pd.DataFrame:
    return fund_df.groupby(level="datetime", group_keys=False).apply(_compute_gate_components_for_day)


def _freeze_component(component: pd.Series, freeze_freq: str) -> pd.Series:
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


def _apply_soft_gate(
    base_pred: pd.DataFrame,
    components: pd.DataFrame,
    freeze_freq: str,
    lam: float,
    tau: float,
    trap_mode: bool,
) -> tuple[pd.DataFrame, dict]:
    gate = _freeze_component(components["gate_score"], freeze_freq=freeze_freq)
    trap = _freeze_component(components["trap"], freeze_freq=freeze_freq)

    common = base_pred.index.intersection(gate.index).intersection(trap.index)
    pred = base_pred.loc[common].copy()
    gate = gate.loc[common].astype(float)
    trap = trap.loc[common].astype(float)

    penalty = lam * np.maximum(0.0, tau - gate.values)
    if trap_mode:
        penalty = penalty + TRAP_EXTRA_PENALTY * trap.values

    pred["score"] = pred["score"].values - penalty

    stats = {
        "gate_mean": float(np.nanmean(gate.values)),
        "gate_p10": float(np.nanquantile(gate.values, 0.10)),
        "trap_ratio": float(np.nanmean(trap.values)),
        "avg_penalty": float(np.nanmean(penalty)),
    }
    return pred.sort_index(), stats


def _prepare_fundamental_components(index_ref: pd.Index) -> pd.DataFrame:
    fund_df = pd.read_parquet(FUNDAMENTAL_RAW_PATH)
    fund_df = _flatten_columns(fund_df)
    fund_df = fund_df.loc[fund_df.index.intersection(index_ref)].sort_index()
    return _compute_gate_components(fund_df)


def _write_validation_best(best_row: dict):
    base_valid = _parse_summary_file(BASELINE_VALIDATION_BEST_PATH)
    lines = [
        f"window_key={base_valid['window_key']}",
        f"train={base_valid['train']}",
        f"valid={base_valid['valid']}",
        f"test={base_valid['test']}",
        "selection_scope=valuation_gate_weight_search",
        f"base_model_trial={base_valid['base_model_trial']}",
        f"base_strategy_trial={base_valid['base_strategy_trial']}",
        f"base_tabular_trial={base_valid['tabular_trial']}",
        f"freeze_freq={best_row['freeze_freq']}",
        f"lambda={best_row['lambda']}",
        f"tau={best_row['tau']}",
        f"trap_mode={best_row['trap_mode']}",
        f"validation_IC={best_row['IC']}",
        f"validation_Rank_IC={best_row['Rank IC']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"validation_score={best_row['score']}",
        f"validation_cache_path={best_row['validation_cache_path']}",
        f"validation_recorder_id={best_row['validation_recorder_id']}",
        f"validation_gate_mean={best_row.get('gate_mean')}",
        f"validation_trap_ratio={best_row.get('trap_ratio')}",
        f"validation_avg_penalty={best_row.get('avg_penalty')}",
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
        f"lambda={best_row['lambda']}",
        f"tau={best_row['tau']}",
        f"trap_mode={best_row['trap_mode']}",
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
        f"test_gate_mean={test_result.get('gate_mean')}",
        f"test_trap_ratio={test_result.get('trap_ratio')}",
        f"test_avg_penalty={test_result.get('avg_penalty')}",
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

    all_index = valid_pred.index.union(test_pred.index)
    components = _prepare_fundamental_components(all_index)

    rows: list[dict] = [
        {
            "trial_name": "baseline_no_gate",
            "freeze_freq": "none",
            "lambda": 0.0,
            "tau": 0.0,
            "trap_mode": False,
            "status": "success",
            "IC": float(baseline_valid["validation_IC"]),
            "Rank IC": float(baseline_valid["validation_Rank_IC"]),
            "with_cost_ann_return": float(baseline_valid["validation_with_cost_ann_return"]),
            "with_cost_ir": float(baseline_valid["validation_with_cost_ir"]),
            "with_cost_mdd": float(baseline_valid["validation_with_cost_mdd"]),
            "score": float(baseline_valid["validation_score"]),
            "validation_cache_path": str(baseline_valid["validation_cache_path"]),
            "validation_recorder_id": pd.NA,
            "gate_mean": pd.NA,
            "trap_ratio": pd.NA,
            "avg_penalty": 0.0,
        }
    ]

    for freeze_freq in FREEZE_FREQS:
        for lam in LAMBDAS:
            for tau in TAUS:
                for trap_mode in TRAP_MODES:
                    trial_name = f"valuation_gate_{freeze_freq}_l{str(lam).replace('.', '')}_t{str(tau).replace('.', '')}_trap{int(trap_mode)}"
                    row = {
                        "trial_name": trial_name,
                        "freeze_freq": freeze_freq,
                        "lambda": lam,
                        "tau": tau,
                        "trap_mode": trap_mode,
                        "status": "running",
                    }
                    try:
                        gated_pred, gate_stats = _apply_soft_gate(
                            base_pred=valid_pred,
                            components=components,
                            freeze_freq=freeze_freq,
                            lam=lam,
                            tau=tau,
                            trap_mode=trap_mode,
                        )
                        common = gated_pred.index.intersection(valid_label.index)
                        gated_pred = gated_pred.loc[common]
                        label = valid_label.loc[common]

                        cache_path = _write_fused_cache(trial_name, gated_pred, label)
                        metrics = _evaluate_signal(
                            recorder_name=trial_name,
                            pred=gated_pred,
                            label=label,
                            strategy_cfg=strategy_cfg_valid,
                            experiment_name="alpha360_tra_alpha158_valuation_gate_validation_eval",
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
                                "gate_mean": gate_stats["gate_mean"],
                                "trap_ratio": gate_stats["trap_ratio"],
                                "avg_penalty": gate_stats["avg_penalty"],
                            }
                        )
                    except Exception as exc:
                        row["status"] = "failed"
                        row["error"] = repr(exc)[:500]

                    rows = [existing for existing in rows if existing.get("trial_name") != trial_name] + [row]
                    _persist(rows, VALIDATION_RESULTS_PATH)

    success_rows = [row for row in rows if row["status"] == "success"]
    if not success_rows:
        raise RuntimeError("no successful valuation-gate rows")

    best_row = max(success_rows, key=lambda item: item["score"])
    _write_validation_best(best_row)

    if best_row["trial_name"] == "baseline_no_gate":
        test_result = {
            "cache_path": baseline_test["test_cache_path"],
            "recorder_id": baseline_test["test_recorder_id"],
            "IC": baseline_test["test_IC"],
            "Rank IC": baseline_test["test_Rank_IC"],
            "with_cost_ann_return": baseline_test["test_with_cost_ann_return"],
            "with_cost_ir": baseline_test["test_with_cost_ir"],
            "with_cost_mdd": baseline_test["test_with_cost_mdd"],
            "gate_mean": pd.NA,
            "trap_ratio": pd.NA,
            "avg_penalty": 0.0,
        }
        _write_final_test(best_row, test_result, baseline_test)
        return

    gated_test_pred, gate_stats_test = _apply_soft_gate(
        base_pred=test_pred,
        components=components,
        freeze_freq=best_row["freeze_freq"],
        lam=float(best_row["lambda"]),
        tau=float(best_row["tau"]),
        trap_mode=bool(best_row["trap_mode"]),
    )
    common_test = gated_test_pred.index.intersection(test_label.index)
    gated_test_pred = gated_test_pred.loc[common_test]
    test_label = test_label.loc[common_test]

    test_cache_path = _write_fused_cache(f"{best_row['trial_name']}_final_test", gated_test_pred, test_label)
    test_metrics = _evaluate_signal(
        recorder_name=f"{best_row['trial_name']}_final_test",
        pred=gated_test_pred,
        label=test_label,
        strategy_cfg=strategy_cfg_test,
        experiment_name="alpha360_tra_alpha158_valuation_gate_final_test_eval",
    )
    test_metrics["cache_path"] = str(test_cache_path)
    test_metrics.update(gate_stats_test)
    _write_final_test(best_row, test_metrics, baseline_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valuation residual + quality-trap soft gate over TRA+Tech baseline.")
    parser.parse_args()
    main()
