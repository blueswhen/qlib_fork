"""Route 2 (Plan B): regime-conditional fusion weight.

Daily market valuation regime (rolling 252-day rank of cross-sectional
median log(PE_TTM)) decides the seq/tab fusion weight:

    regime_pct > high_q  -> tw = tw_high  (expensive market: lean tech-DNN reversion)
    regime_pct <= high_q -> tw = tw_low   (normal/cheap: lean TRA momentum)

This is a post-processing on the existing TRA + DNN02 caches (no retraining).
We reuse:
  - TRA seq cache from `tra_strict_validation_best.txt` / `tra_strict_final_test.txt`
  - DNN02 signal from recorder ids in baseline weight_refine summary files
  - Valuation components from `tushare/processed/training/csi300_daily_fundamental_features_qlib.parquet`
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

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
BASELINE_VALIDATION_BEST_PATH = BASE_DIR / "alpha360_tra_alpha158_dnn_weight_refine_validation_best.txt"
BASELINE_FINAL_TEST_PATH = BASE_DIR / "alpha360_tra_alpha158_dnn_weight_refine_final_test.txt"

VALIDATION_RESULTS_PATH = BASE_DIR / "alpha360_tra_alpha158_valuation_regime_validation_results.csv"
VALIDATION_BEST_PATH = BASE_DIR / "alpha360_tra_alpha158_valuation_regime_validation_best.txt"
FINAL_TEST_PATH = BASE_DIR / "alpha360_tra_alpha158_valuation_regime_final_test.txt"

FUNDAMENTAL_RAW_PATH = (
    "/home/blueswhen/DL/qlib/tushare/processed/training/csi300_daily_fundamental_features_qlib.parquet"
)

FUSION_MODE = "zscore"

# Regime config
REGIME_LOOKBACK = 252
FREEZE_FREQ = "month"   # freeze regime to the first trading day of each month

# Grid (validation only)
HIGH_QS = [0.70, 0.80]
TW_LOWS = [0.05, 0.10]
TW_HIGHS = [0.20, 0.30, 0.40]


# ----------------------------------------------------------------------
# Regime computation
# ----------------------------------------------------------------------
def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        out = df.copy()
        out.columns = [str(col[-1]) for col in out.columns]
        return out
    return df


def _compute_market_regime(fund_df: pd.DataFrame, lookback: int = REGIME_LOOKBACK) -> pd.Series:
    """Daily rolling-rank of cross-sectional median log(PE_TTM).

    Returns a Series indexed by trading date with values in [0, 1]:
      higher = market overall more expensive than the past `lookback` days.
    """
    pe = fund_df["pe_ttm"].copy()
    pe = pe.where(pe > 0.1)
    log_pe = np.log(pe)
    daily_median = log_pe.groupby(level="datetime").median().sort_index()

    # Rolling rank percentile; fall back to expanding rank for warmup period.
    rolling_rank = daily_median.rolling(lookback, min_periods=60).rank(pct=True)
    expanding_rank = daily_median.expanding(min_periods=20).rank(pct=True)
    regime = rolling_rank.fillna(expanding_rank).fillna(0.5)
    return regime


def _freeze_regime(regime: pd.Series, freeze_freq: str) -> pd.Series:
    """Freeze daily regime to the first trading day of each month/quarter."""
    if freeze_freq == "none":
        return regime
    s = regime.sort_index()
    period = s.index.to_period("M" if freeze_freq == "month" else "Q")
    out = s.copy()
    for p in pd.unique(period):
        mask = period == p
        anchor_idx = s.index[mask].min()
        out.loc[s.index[mask]] = s.loc[anchor_idx]
    return out


# ----------------------------------------------------------------------
# Conditional fusion
# ----------------------------------------------------------------------
def _per_day_tw(regime: pd.Series, high_q: float, tw_low: float, tw_high: float) -> pd.Series:
    """Hard-step: regime > high_q -> tw_high, else tw_low."""
    tw = pd.Series(tw_low, index=regime.index, dtype=float)
    tw.loc[regime > high_q] = tw_high
    return tw


def _regime_fused(
    seq_norm: pd.DataFrame,
    tab_norm: pd.DataFrame,
    tw_per_day: pd.Series,
) -> pd.DataFrame:
    """Per-day fusion weight from `tw_per_day` indexed by date."""
    common = seq_norm.index.intersection(tab_norm.index)
    seq = seq_norm.loc[common]["score"].astype(float)
    tab = tab_norm.loc[common]["score"].astype(float)

    seq_wide = seq.unstack("instrument").sort_index()
    tab_wide = tab.unstack("instrument").reindex_like(seq_wide)
    tw = tw_per_day.reindex(seq_wide.index, method="ffill").fillna(method="bfill").fillna(tw_per_day.iloc[-1])

    seq_w = (1.0 - tw).values[:, None]
    tab_w = tw.values[:, None]
    fused_wide = seq_wide.fillna(0.0).values * seq_w + tab_wide.fillna(0.0).values * tab_w

    fused_df = pd.DataFrame(fused_wide, index=seq_wide.index, columns=seq_wide.columns)
    # Restore NaN where both inputs were NaN
    mask = seq_wide.isna() & tab_wide.isna()
    fused_df = fused_df.mask(mask)
    fused = fused_df.stack(dropna=True).to_frame("score")
    fused.index.names = ["datetime", "instrument"]
    return fused.sort_index(), tw


# ----------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------
def _write_validation_best(best_row: dict):
    base_valid = _parse_summary_file(BASELINE_VALIDATION_BEST_PATH)
    lines = [
        f"window_key={base_valid['window_key']}",
        f"train={base_valid['train']}",
        f"valid={base_valid['valid']}",
        f"test={base_valid['test']}",
        "selection_scope=valuation_regime_search",
        f"base_model_trial={base_valid['base_model_trial']}",
        f"base_strategy_trial={base_valid['base_strategy_trial']}",
        f"base_tabular_trial={base_valid['tabular_trial']}",
        f"regime_lookback={REGIME_LOOKBACK}",
        f"freeze_freq={FREEZE_FREQ}",
        f"high_q={best_row['high_q']}",
        f"tw_low={best_row['tw_low']}",
        f"tw_high={best_row['tw_high']}",
        f"validation_IC={best_row['IC']}",
        f"validation_Rank_IC={best_row['Rank IC']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"validation_score={best_row['score']}",
        f"validation_cache_path={best_row['validation_cache_path']}",
        f"validation_recorder_id={best_row['validation_recorder_id']}",
        f"validation_high_regime_day_ratio={best_row.get('high_regime_day_ratio')}",
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
        f"regime_lookback={REGIME_LOOKBACK}",
        f"freeze_freq={FREEZE_FREQ}",
        f"high_q={best_row['high_q']}",
        f"tw_low={best_row['tw_low']}",
        f"tw_high={best_row['tw_high']}",
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
        f"test_high_regime_day_ratio={test_result.get('high_regime_day_ratio')}",
        f"baseline_test_with_cost_ann_return={baseline_test['test_with_cost_ann_return']}",
        f"baseline_test_with_cost_ir={baseline_test['test_with_cost_ir']}",
        f"baseline_test_with_cost_mdd={baseline_test['test_with_cost_mdd']}",
    ]
    FINAL_TEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    tra_validation = _parse_summary_file(TRA_VALIDATION_BEST_PATH)
    tra_final = _parse_summary_file(TRA_FINAL_TEST_PATH)
    baseline_valid = _parse_summary_file(BASELINE_VALIDATION_BEST_PATH)
    baseline_test = _parse_summary_file(BASELINE_FINAL_TEST_PATH)

    strategy_cfg_valid = _load_alstm_strategy_cfg(tra_validation, eval_segment="valid")
    strategy_cfg_test = _load_alstm_strategy_cfg(tra_validation, eval_segment="test")

    # ---- Load per-source caches ----
    seq_valid_pred, seq_valid_label = _load_signal_from_cache(tra_validation["validation_cache_path"])
    tab_valid_pred, tab_valid_label = _load_signal_from_recorder(
        baseline_valid["tabular_validation_recorder_id"]
    )
    seq_v, tab_v, label_v = _align_signals(seq_valid_pred, seq_valid_label, tab_valid_pred, tab_valid_label)
    seq_v_norm = _normalize_scores(seq_v, FUSION_MODE)
    tab_v_norm = _normalize_scores(tab_v, FUSION_MODE)

    seq_test_pred, seq_test_label = _load_signal_from_cache(tra_final["test_cache_path"])
    tab_test_pred, tab_test_label = _load_signal_from_recorder(baseline_test["test_recorder_id"])
    # baseline_test cache_path is fused; we need raw DNN signal -> use the
    # baseline weight_refine recorder for tabular. Look for tabular_test_recorder_id.
    if "tabular_test_recorder_id" in baseline_test and baseline_test["tabular_test_recorder_id"]:
        tab_test_pred, tab_test_label = _load_signal_from_recorder(
            baseline_test["tabular_test_recorder_id"]
        )
    seq_t, tab_t, label_t = _align_signals(seq_test_pred, seq_test_label, tab_test_pred, tab_test_label)
    seq_t_norm = _normalize_scores(seq_t, FUSION_MODE)
    tab_t_norm = _normalize_scores(tab_t, FUSION_MODE)

    # ---- Compute regime over the full date range ----
    fund_df = pd.read_parquet(FUNDAMENTAL_RAW_PATH)
    fund_df = _flatten_columns(fund_df)
    regime_full = _compute_market_regime(fund_df, lookback=REGIME_LOOKBACK)
    regime_frozen = _freeze_regime(regime_full, freeze_freq=FREEZE_FREQ)

    # ---- Baseline (constant tw=0.10) row for reference ----
    rows: list[dict] = [
        {
            "trial_name": "baseline_constant_tw010",
            "high_q": pd.NA,
            "tw_low": 0.10,
            "tw_high": 0.10,
            "status": "success",
            "IC": float(baseline_valid["validation_IC"]),
            "Rank IC": float(baseline_valid["validation_Rank_IC"]),
            "with_cost_ann_return": float(baseline_valid["validation_with_cost_ann_return"]),
            "with_cost_ir": float(baseline_valid["validation_with_cost_ir"]),
            "with_cost_mdd": float(baseline_valid["validation_with_cost_mdd"]),
            "score": float(baseline_valid["validation_score"]),
            "validation_cache_path": str(baseline_valid["validation_cache_path"]),
            "validation_recorder_id": pd.NA,
            "high_regime_day_ratio": pd.NA,
        }
    ]

    for high_q in HIGH_QS:
        for tw_low in TW_LOWS:
            for tw_high in TW_HIGHS:
                if tw_high <= tw_low:
                    continue
                trial_name = (
                    f"valuation_regime_hq{int(high_q*100)}"
                    f"_twl{str(tw_low).replace('.', '')}"
                    f"_twh{str(tw_high).replace('.', '')}"
                )
                row = {
                    "trial_name": trial_name,
                    "high_q": high_q,
                    "tw_low": tw_low,
                    "tw_high": tw_high,
                    "status": "running",
                }
                try:
                    tw_v = _per_day_tw(regime_frozen, high_q=high_q, tw_low=tw_low, tw_high=tw_high)
                    fused_v, tw_used_v = _regime_fused(seq_v_norm, tab_v_norm, tw_v)
                    common = fused_v.index.intersection(label_v.index)
                    fused_v = fused_v.loc[common]
                    label_v_aligned = label_v.loc[common]
                    cache_path = _write_fused_cache(trial_name, fused_v, label_v_aligned)
                    metrics = _evaluate_signal(
                        recorder_name=trial_name,
                        pred=fused_v,
                        label=label_v_aligned,
                        strategy_cfg=strategy_cfg_valid,
                        experiment_name="alpha360_tra_alpha158_valuation_regime_validation_eval",
                    )
                    high_ratio_v = float((tw_used_v.reindex(fused_v.index.get_level_values("datetime").unique()) > tw_low + 1e-9).mean())
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
                            "high_regime_day_ratio": high_ratio_v,
                        }
                    )
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = repr(exc)[:500]
                rows = [r for r in rows if r.get("trial_name") != trial_name] + [row]
                _persist(rows, VALIDATION_RESULTS_PATH)

    success_rows = [r for r in rows if r["status"] == "success"]
    if not success_rows:
        raise RuntimeError("no successful valuation-regime rows")
    best_row = max(success_rows, key=lambda item: item["score"])
    _write_validation_best(best_row)

    if best_row["trial_name"] == "baseline_constant_tw010":
        test_result = {
            "cache_path": baseline_test["test_cache_path"],
            "recorder_id": baseline_test["test_recorder_id"],
            "IC": baseline_test["test_IC"],
            "Rank IC": baseline_test["test_Rank_IC"],
            "with_cost_ann_return": baseline_test["test_with_cost_ann_return"],
            "with_cost_ir": baseline_test["test_with_cost_ir"],
            "with_cost_mdd": baseline_test["test_with_cost_mdd"],
            "high_regime_day_ratio": pd.NA,
        }
        _write_final_test(best_row, test_result, baseline_test)
        return

    tw_t = _per_day_tw(
        regime_frozen,
        high_q=float(best_row["high_q"]),
        tw_low=float(best_row["tw_low"]),
        tw_high=float(best_row["tw_high"]),
    )
    fused_t, tw_used_t = _regime_fused(seq_t_norm, tab_t_norm, tw_t)
    common_t = fused_t.index.intersection(label_t.index)
    fused_t = fused_t.loc[common_t]
    label_t_aligned = label_t.loc[common_t]

    test_cache_path = _write_fused_cache(f"{best_row['trial_name']}_final_test", fused_t, label_t_aligned)
    test_metrics = _evaluate_signal(
        recorder_name=f"{best_row['trial_name']}_final_test",
        pred=fused_t,
        label=label_t_aligned,
        strategy_cfg=strategy_cfg_test,
        experiment_name="alpha360_tra_alpha158_valuation_regime_final_test_eval",
    )
    test_metrics["cache_path"] = str(test_cache_path)
    test_metrics["high_regime_day_ratio"] = float(
        (tw_used_t.reindex(fused_t.index.get_level_values("datetime").unique()) > float(best_row["tw_low"]) + 1e-9).mean()
    )
    _write_final_test(best_row, test_metrics, baseline_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valuation regime-conditional fusion weight.")
    parser.parse_args()
    main()
