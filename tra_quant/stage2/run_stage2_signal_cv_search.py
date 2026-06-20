from __future__ import annotations

import argparse
import copy
import json
import logging
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd
import qlib
from qlib.backtest import backtest as normal_backtest
from qlib.constant import REG_CN
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.eva.alpha import calc_ic

import run_stage2_signal_search_fast as fast
import tune_alpha360_tra_fundamental_xgb_weight_refine as stage2
from rolling_alstm_alpha360_latest import (
    PROVIDER_URI,
    build_base_task,
)


logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp" / "stage2_signal_cv"
ANNOUNCEMENT_DECAY_RANKPCT_FEATURE_PATH = (
    BASE_DIR.parent.parent
    / "training_data"
    / "processed"
    / "training"
    / "csi300_daily_announcement_features_decay_rankpct_qlib.parquet"
)
SORT_COLUMNS = ["stable_score", "year_ann_min", "year_ann_mean", "full_ann"]
SEGMENTS = {
    "valid_all": ("2020-01-01", "2022-12-31"),
    "2020": ("2020-01-01", "2020-12-31"),
    "2021": ("2021-01-01", "2021-12-31"),
    "2022": ("2022-01-01", "2022-12-31"),
}
STABILITY_WEIGHTS = {"ann_std": 400.0, "ann_spread": 300.0, "negative_worst_ann": 2000.0}

_QLIB_INITIALIZED = False


def _ensure_qlib_initialized() -> None:
    global _QLIB_INITIALIZED
    if not _QLIB_INITIALIZED:
        qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)
        _QLIB_INITIALIZED = True


def _effective_prefix(output_prefix: Path, *, num_shards: int, shard_index: int) -> Path:
    if num_shards <= 1:
        return output_prefix
    return output_prefix.with_name(f"{output_prefix.name}_shard{shard_index:02d}of{num_shards:02d}")


def _grid_path(output_prefix: Path) -> Path:
    return output_prefix.with_name(f"{output_prefix.name}_cv_grid.csv")


def _summary_path(output_prefix: Path) -> Path:
    return output_prefix.with_name(f"{output_prefix.name}_summary.json")


def _load_existing_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "trial_name" not in df.columns:
        return {}
    return {str(row["trial_name"]): row for row in df.to_dict(orient="records")}


def _persist(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    pd.DataFrame(rows).sort_values(
        SORT_COLUMNS,
        ascending=[False, False, False, False],
        na_position="last",
    ).to_csv(path, index=False)


def _slice_signal(df: pd.DataFrame, date_range: tuple[str, str]) -> pd.DataFrame:
    dates = df.index.get_level_values("datetime" if "datetime" in df.index.names else 0)
    mask = (dates >= pd.Timestamp(date_range[0])) & (dates <= pd.Timestamp(date_range[1]))
    return df.loc[mask].sort_index()


def _objective(metrics: dict) -> float:
    ann = float(metrics["with_cost_ann_return"])
    ir = float(metrics["with_cost_ir"])
    mdd = float(metrics["with_cost_mdd"])
    return ann * 1000 + ir * 50 + mdd * 20


def _signal_ic(pred: pd.DataFrame, label: pd.DataFrame) -> tuple[float, float]:
    ic, rank_ic = calc_ic(pred.iloc[:, 0], label.iloc[:, 0])
    return float(ic.mean()), float(rank_ic.mean())


def _evaluate_segment(
    tra_validation: dict,
    strategy_trial: dict,
    pred: pd.DataFrame,
    label: pd.DataFrame,
    *,
    segment: tuple[str, str],
) -> dict:
    _ensure_qlib_initialized()
    eval_pred = _slice_signal(pred, segment)
    eval_label = _slice_signal(label, segment)
    common = eval_pred.index.intersection(eval_label.index).sort_values()
    eval_pred = eval_pred.loc[common]
    eval_label = eval_label.loc[common]
    ic, rank_ic = _signal_ic(eval_pred, eval_label)

    task = build_base_task(
        topk=stage2.TOPK,
        n_drop=int(strategy_trial["n_drop"]),
        window_key=tra_validation["window_key"],
        model_key="gru_base",
        handler_class="Alpha360",
        handler_module_path="qlib.contrib.data.handler",
        account=stage2.ACCOUNT,
        risk_degree=float(strategy_trial["risk_degree"]),
        benchmark=stage2.BENCHMARK,
        instruments=stage2.INSTRUMENTS,
        handler_kwargs_extra=tra_validation["handler_kwargs_extra"],
        strategy_class=strategy_trial["strategy_class"],
        strategy_module_path=strategy_trial["strategy_module_path"],
        strategy_kwargs_extra=strategy_trial["kwargs_extra"],
        eval_segment="valid",
        backtest_segment="valid",
    )
    port_record = task["record"][2]
    strategy_config = copy.deepcopy(port_record["kwargs"]["config"]["strategy"])
    strategy_config["kwargs"]["signal"] = eval_pred
    backtest_config = copy.deepcopy(port_record["kwargs"]["config"]["backtest"])
    backtest_config["start_time"] = segment[0]
    backtest_config["end_time"] = segment[1]
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }
    portfolio_metric_dict, _ = normal_backtest(
        executor=executor_config,
        strategy=strategy_config,
        **backtest_config,
    )
    report_normal, _ = portfolio_metric_dict["1day"]
    with_cost = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq="1day")
    result = {
        "rows": int(len(common)),
        "IC": ic,
        "Rank IC": rank_ic,
        "with_cost_ann_return": float(with_cost.loc["annualized_return", "risk"]),
        "with_cost_ir": float(with_cost.loc["information_ratio", "risk"]),
        "with_cost_mdd": float(with_cost.loc["max_drawdown", "risk"]),
    }
    result["score"] = _objective(result)
    return result


def _cols(fund: pd.DataFrame, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if name in fund.columns]


def _composite(fund: pd.DataFrame, *, plus: tuple[str, ...] = (), minus: tuple[str, ...] = ()) -> pd.DataFrame:
    parts = []
    for col in _cols(fund, plus):
        parts.append(fund[col].astype(float))
    for col in _cols(fund, minus):
        parts.append(-fund[col].astype(float))
    if not parts:
        score = pd.Series(0.0, index=fund.index)
    else:
        score = pd.concat(parts, axis=1).mean(axis=1).fillna(0.0)
    return pd.DataFrame({"score": score}, index=fund.index)


def _extended_composites(index) -> dict[str, pd.DataFrame]:
    fund = pd.read_parquet(stage2.FUNDAMENTAL_FEATURE_PATH)
    if isinstance(fund.columns, pd.MultiIndex):
        fund.columns = fund.columns.get_level_values(-1)
    fund = fund.loc[index.intersection(fund.index).sort_values()]
    base = fast._fundamental_composites(index)
    base.update(
        {
            "quality_balance": _composite(
                fund,
                plus=("fi_roe", "fi_roe_dt", "fi_netprofit_margin", "fi_assets_turn", "fi_bps"),
                minus=("fi_debt_to_assets",),
            ),
            "large_liquid": _composite(
                fund,
                plus=("total_mv", "circ_mv", "turnover_rate_f", "volume_ratio"),
            ),
            "low_valuation": _composite(
                fund,
                plus=("dv_ttm", "dv_ratio", "fi_bps"),
                minus=("pe_ttm", "pe", "pb", "ps_ttm", "ps"),
            ),
        }
    )
    return base


def _announcement_composites(index) -> dict[str, pd.DataFrame]:
    ann = pd.read_parquet(ANNOUNCEMENT_DECAY_RANKPCT_FEATURE_PATH)
    if isinstance(ann.columns, pd.MultiIndex):
        ann.columns = ann.columns.get_level_values(-1)
    ann = ann.loc[index.intersection(ann.index).sort_values()]
    return {
        "ann_good_decay": _composite(
            ann,
            plus=(
                "ann_positive_cnt_decay_h3",
                "ann_positive_cnt_decay_h10",
                "ann_contract_order_cnt_decay_h3",
                "ann_contract_order_cnt_decay_h10",
                "ann_buyback_cnt_decay_h3",
                "ann_buyback_cnt_decay_h10",
                "ann_sentiment_score_decay_h3",
                "ann_sentiment_score_decay_h10",
            ),
            minus=(
                "ann_negative_cnt_decay_h3",
                "ann_negative_cnt_decay_h10",
                "ann_regulatory_cnt_decay_h3",
                "ann_risk_cnt_decay_h3",
                "ann_risk_cnt_decay_h10",
            ),
        ),
        "ann_risk_avoid": _composite(
            ann,
            plus=("ann_sentiment_score_decay_h10", "ann_sentiment_score_decay_h30"),
            minus=(
                "ann_negative_cnt_decay_h3",
                "ann_negative_cnt_decay_h10",
                "ann_negative_cnt_decay_h30",
                "ann_regulatory_cnt_decay_h3",
                "ann_regulatory_cnt_decay_h10",
                "ann_risk_cnt_decay_h3",
                "ann_risk_cnt_decay_h10",
                "ann_risk_cnt_decay_h30",
                "ann_insider_change_cnt_decay_h3",
                "ann_insider_change_cnt_decay_h10",
            ),
        ),
        "ann_positive_event": _composite(
            ann,
            plus=(
                "ann_positive_cnt_decay_h3",
                "ann_contract_order_cnt_decay_h3",
                "ann_buyback_cnt_decay_h3",
                "ann_sentiment_score_decay_h3",
            ),
            minus=("ann_negative_cnt_decay_h3", "ann_risk_cnt_decay_h3"),
        ),
        "ann_clean_recent": _composite(
            ann,
            plus=("days_since_last_announcement", "days_since_last_earnings_announcement"),
            minus=(
                "ann_cnt_decay_h3",
                "ann_negative_cnt_decay_h3",
                "ann_regulatory_cnt_decay_h3",
                "ann_risk_cnt_decay_h3",
            ),
        ),
    }


def _add_overlay_candidates(
    candidates: list[dict],
    seen: set[str],
    *,
    base_name: str,
    base_pred: pd.DataFrame,
    seq_label: pd.DataFrame,
    overlays: dict[str, pd.DataFrame],
    linear_weights: tuple[float, ...],
    low_specs: tuple[tuple[float, float], ...],
    high_specs: tuple[tuple[float, float], ...],
) -> None:
    for name, raw in overlays.items():
        common = base_pred.index.intersection(raw.index).intersection(seq_label.index).sort_values()
        base = base_pred.loc[common]
        label = seq_label.loc[common]
        raw_z = stage2._normalize_scores(raw.loc[common], "zscore")
        raw_rank = stage2._normalize_scores(raw.loc[common], "rank_pct")

        for weight in linear_weights:
            signal_name = f"{base_name}_{name}_z_aw{str(weight).replace('.', '')}"
            if signal_name not in seen:
                pred = base.mul(1.0 - weight).add(raw_z.mul(weight), fill_value=0.0)
                candidates.append(
                    {
                        "signal_name": signal_name,
                        "kind": f"{base_name}_{name}_linear",
                        "ann_weight": weight,
                        "pred": pred,
                        "label": label,
                    }
                )
                seen.add(signal_name)

        for low_q, penalty in low_specs:
            signal_name = f"{base_name}_{name}_lowpen_lq{str(low_q).replace('.', '')}_p{str(penalty).replace('.', '')}"
            if signal_name not in seen:
                pred = base.copy()
                pred["score"] = pred["score"] - (raw_rank["score"] <= low_q).astype(float) * penalty
                candidates.append(
                    {
                        "signal_name": signal_name,
                        "kind": f"{base_name}_{name}_low_penalty",
                        "low_q": low_q,
                        "penalty": penalty,
                        "pred": pred,
                        "label": label,
                    }
                )
                seen.add(signal_name)

        for high_q, bonus in high_specs:
            signal_name = f"{base_name}_{name}_hibonus_hq{str(high_q).replace('.', '')}_b{str(bonus).replace('.', '')}"
            if signal_name not in seen:
                pred = base.copy()
                pred["score"] = pred["score"] + (raw_rank["score"] >= high_q).astype(float) * bonus
                candidates.append(
                    {
                        "signal_name": signal_name,
                        "kind": f"{base_name}_{name}_high_bonus",
                        "high_q": high_q,
                        "bonus": bonus,
                        "pred": pred,
                        "label": label,
                    }
                )
                seen.add(signal_name)


def _build_cv_signal_candidates(seq_pred: pd.DataFrame, seq_label: pd.DataFrame, *, profile: str) -> list[dict]:
    candidates = [{"signal_name": "baseline_seq", "kind": "baseline", "pred": seq_pred, "label": seq_label}]
    composites = _extended_composites(seq_pred.index)
    if profile == "robust-small":
        names = ("value_quality", "growth_quality", "cash_quality", "quality_balance", "large_liquid", "low_valuation")
        linear_weights = (0.005, 0.01, 0.02)
        low_specs = ((0.10, 0.05), (0.15, 0.05), (0.15, 0.10), (0.20, 0.10), (0.20, 0.15))
        high_specs = ((0.80, 0.02), (0.85, 0.02), (0.85, 0.05), (0.90, 0.05))
        two_sided_specs = ((0.15, 0.05, 0.85, 0.02), (0.20, 0.10, 0.85, 0.02))
    elif profile == "robust-tiny":
        names = ("value_quality", "growth_quality", "cash_quality")
        linear_weights = (0.005, 0.01)
        low_specs = ((0.15, 0.05), (0.20, 0.10))
        high_specs = ((0.85, 0.02),)
        two_sided_specs = ((0.20, 0.10, 0.85, 0.02),)
    elif profile in {"announcement-small", "cash-announcement-small"}:
        ann_composites = _announcement_composites(seq_pred.index)
        common = seq_pred.index.intersection(seq_label.index).sort_values()
        label = seq_label.loc[common]
        seq_z = stage2._normalize_scores(seq_pred.loc[common], "zscore")
        seen = {"baseline_seq"}

        if profile == "announcement-small":
            _add_overlay_candidates(
                candidates,
                seen,
                base_name="seq",
                base_pred=seq_z,
                seq_label=label,
                overlays=ann_composites,
                linear_weights=(0.0025, 0.005, 0.01, 0.015),
                low_specs=((0.10, 0.02), (0.15, 0.02), (0.15, 0.05), (0.20, 0.05)),
                high_specs=((0.80, 0.005), (0.85, 0.005), (0.85, 0.01)),
            )
            return candidates

        cash_quality = composites["cash_quality"]
        cash_common = common.intersection(cash_quality.index).sort_values()
        cash_z = stage2._normalize_scores(cash_quality.loc[cash_common], "zscore")
        cash_base = seq_z.loc[cash_common].mul(0.99).add(cash_z.mul(0.01), fill_value=0.0)
        cash_base_name = "cash_quality_z_tw001"
        candidates.append(
            {
                "signal_name": cash_base_name,
                "kind": "cash_quality_linear_fixed",
                "tab_weight": 0.01,
                "pred": cash_base,
                "label": label.loc[cash_common],
            }
        )
        seen.add(cash_base_name)
        _add_overlay_candidates(
            candidates,
            seen,
            base_name=cash_base_name,
            base_pred=cash_base,
            seq_label=label,
            overlays=ann_composites,
            linear_weights=(0.0025, 0.005, 0.01),
            low_specs=((0.10, 0.01), (0.15, 0.01), (0.15, 0.02), (0.20, 0.02)),
            high_specs=((0.80, 0.0025), (0.85, 0.0025), (0.85, 0.005)),
        )
        return candidates
    else:
        raise ValueError(f"unknown signal profile: {profile}")

    seen = {"baseline_seq"}
    for name in names:
        raw = composites[name]
        common = seq_pred.index.intersection(raw.index).intersection(seq_label.index).sort_values()
        seq = seq_pred.loc[common]
        label = seq_label.loc[common]
        seq_z = stage2._normalize_scores(seq, "zscore")
        raw_z = stage2._normalize_scores(raw.loc[common], "zscore")
        raw_rank = stage2._normalize_scores(raw.loc[common], "rank_pct")

        for weight in linear_weights:
            signal_name = f"{name}_z_tw{str(weight).replace('.', '')}"
            if signal_name not in seen:
                pred = seq_z.mul(1.0 - weight).add(raw_z.mul(weight), fill_value=0.0)
                candidates.append(
                    {"signal_name": signal_name, "kind": f"{name}_linear", "tab_weight": weight, "pred": pred, "label": label}
                )
                seen.add(signal_name)
        for low_q, penalty in low_specs:
            signal_name = f"{name}_lowpen_lq{str(low_q).replace('.', '')}_p{str(penalty).replace('.', '')}"
            if signal_name not in seen:
                pred = seq_z.copy()
                pred["score"] = pred["score"] - (raw_rank["score"] <= low_q).astype(float) * penalty
                candidates.append(
                    {
                        "signal_name": signal_name,
                        "kind": f"{name}_low_penalty",
                        "low_q": low_q,
                        "penalty": penalty,
                        "pred": pred,
                        "label": label,
                    }
                )
                seen.add(signal_name)
        for high_q, bonus in high_specs:
            signal_name = f"{name}_hibonus_hq{str(high_q).replace('.', '')}_b{str(bonus).replace('.', '')}"
            if signal_name not in seen:
                pred = seq_z.copy()
                pred["score"] = pred["score"] + (raw_rank["score"] >= high_q).astype(float) * bonus
                candidates.append(
                    {
                        "signal_name": signal_name,
                        "kind": f"{name}_high_bonus",
                        "high_q": high_q,
                        "bonus": bonus,
                        "pred": pred,
                        "label": label,
                    }
                )
                seen.add(signal_name)
        for low_q, penalty, high_q, bonus in two_sided_specs:
            signal_name = (
                f"{name}_twoside_lq{str(low_q).replace('.', '')}_p{str(penalty).replace('.', '')}"
                f"_hq{str(high_q).replace('.', '')}_b{str(bonus).replace('.', '')}"
            )
            if signal_name not in seen:
                pred = seq_z.copy()
                pred["score"] = (
                    pred["score"]
                    - (raw_rank["score"] <= low_q).astype(float) * penalty
                    + (raw_rank["score"] >= high_q).astype(float) * bonus
                )
                candidates.append(
                    {
                        "signal_name": signal_name,
                        "kind": f"{name}_two_sided",
                        "low_q": low_q,
                        "penalty": penalty,
                        "high_q": high_q,
                        "bonus": bonus,
                        "pred": pred,
                        "label": label,
                    }
                )
                seen.add(signal_name)
    return candidates


def _strategy_trials(profile: str) -> list[dict]:
    if profile == "stable-core":
        trials = []
        for hold_thresh in (4, 5, 7):
            for risk_degree in (0.80, 0.85):
                trials.append(
                    stage2._prac(
                        f"prac_m000_hold{hold_thresh}_{stage2._risk_tag(risk_degree)}",
                        risk_degree,
                        0.0,
                        hold_thresh,
                        n_drop=1,
                    )
                )
        return trials
    if profile == "stable-tiny":
        return [
            stage2._prac("prac_m000_hold4_r08", 0.80, 0.0, 4, n_drop=1),
            stage2._prac("prac_m000_hold5_r08", 0.80, 0.0, 5, n_drop=1),
        ]
    raise ValueError(f"unknown strategy profile: {profile}")


def _evaluate_cv_candidate(tra_validation: dict, signal_candidate: dict, strategy_trial: dict) -> dict:
    segment_results = {}
    for segment_name, date_range in SEGMENTS.items():
        segment_results[segment_name] = _evaluate_segment(
            tra_validation,
            strategy_trial,
            signal_candidate["pred"],
            signal_candidate["label"],
            segment=date_range,
        )

    year_results = [segment_results[str(year)] for year in (2020, 2021, 2022)]
    ann = pd.Series([row["with_cost_ann_return"] for row in year_results], dtype="float64")
    ir = pd.Series([row["with_cost_ir"] for row in year_results], dtype="float64")
    mdd = pd.Series([row["with_cost_mdd"] for row in year_results], dtype="float64")
    mean_metrics = {
        "with_cost_ann_return": float(ann.mean()),
        "with_cost_ir": float(ir.mean()),
        "with_cost_mdd": float(mdd.mean()),
    }
    mean_score = _objective(mean_metrics)
    ann_std = float(ann.std(ddof=0))
    ann_spread = float(ann.max() - ann.min())
    year_ann_min = float(ann.min())
    stability_penalty = (
        ann_std * STABILITY_WEIGHTS["ann_std"]
        + ann_spread * STABILITY_WEIGHTS["ann_spread"]
        + max(0.0, -year_ann_min) * STABILITY_WEIGHTS["negative_worst_ann"]
    )
    stable_score = mean_score - stability_penalty
    full = segment_results["valid_all"]

    row = {
        "trial_name": f"{signal_candidate['signal_name']}__{strategy_trial['trial_name']}",
        "signal_name": signal_candidate["signal_name"],
        "kind": signal_candidate["kind"],
        "strategy_trial": strategy_trial["trial_name"],
        "n_drop": strategy_trial["n_drop"],
        "risk_degree": strategy_trial["risk_degree"],
        "kwargs_extra": str(strategy_trial["kwargs_extra"]),
        "tab_weight": signal_candidate.get("tab_weight", pd.NA),
        "low_q": signal_candidate.get("low_q", pd.NA),
        "penalty": signal_candidate.get("penalty", pd.NA),
        "high_q": signal_candidate.get("high_q", pd.NA),
        "bonus": signal_candidate.get("bonus", pd.NA),
        "ann_weight": signal_candidate.get("ann_weight", pd.NA),
        "status": "success",
        "score_mode": "valid_yearly_cv_stage2_v1",
        "stable_score": stable_score,
        "segment_mean_score": mean_score,
        "stability_penalty": stability_penalty,
        "year_ann_mean": mean_metrics["with_cost_ann_return"],
        "year_ann_min": year_ann_min,
        "year_ann_std": ann_std,
        "year_ann_spread": ann_spread,
        "year_ir_mean": mean_metrics["with_cost_ir"],
        "year_mdd_mean": mean_metrics["with_cost_mdd"],
        "full_score": full["score"],
        "full_ann": full["with_cost_ann_return"],
        "full_ir": full["with_cost_ir"],
        "full_mdd": full["with_cost_mdd"],
    }
    for year in (2020, 2021, 2022):
        item = segment_results[str(year)]
        row[f"ann_{year}"] = item["with_cost_ann_return"]
        row[f"ir_{year}"] = item["with_cost_ir"]
        row[f"mdd_{year}"] = item["with_cost_mdd"]
    return row


def run_shard(args: argparse.Namespace) -> None:
    output_prefix = _effective_prefix(args.output_prefix.resolve(), num_shards=args.num_shards, shard_index=args.shard_index)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path = _grid_path(output_prefix)
    output_summary_path = _summary_path(output_prefix)

    tra_validation, seq_pred, seq_label = fast._load_rank_ensemble_validation(
        args.result_suffix,
        validation_summary_path=args.stage1_validation_summary_path,
        repro_json_path=args.stage1_repro_json_path,
    )
    signal_candidates = _build_cv_signal_candidates(seq_pred, seq_label, profile=args.signal_profile)
    strategy_trials = _strategy_trials(args.strategy_profile)
    all_candidates = [
        (signal_candidate, strategy_trial)
        for signal_candidate in signal_candidates
        for strategy_trial in strategy_trials
    ]
    shard_candidates = [
        item for index, item in enumerate(all_candidates) if index % args.num_shards == args.shard_index
    ]
    row_map = {} if args.force_recompute else _load_existing_rows(output_csv_path)
    rows = list(row_map.values())
    evaluated_count = 0
    reused_count = 0

    for signal_candidate, strategy_trial in shard_candidates:
        trial_name = f"{signal_candidate['signal_name']}__{strategy_trial['trial_name']}"
        if trial_name in row_map:
            reused_count += 1
            continue
        try:
            row = _evaluate_cv_candidate(tra_validation, signal_candidate, strategy_trial)
        except Exception as exc:
            row = {
                "trial_name": trial_name,
                "signal_name": signal_candidate["signal_name"],
                "kind": signal_candidate["kind"],
                "strategy_trial": strategy_trial["trial_name"],
                "status": "failed",
                "error": repr(exc)[:500],
            }
        row_map[trial_name] = row
        rows = list(row_map.values())
        evaluated_count += 1
        _persist(rows, output_csv_path)
        print(
            f"evaluated={evaluated_count} reused={reused_count} "
            f"trial={trial_name} status={row['status']} stable_score={row.get('stable_score')}"
        )

    _persist(rows, output_csv_path)
    success = [row for row in rows if row.get("status") == "success"]
    success = sorted(
        success,
        key=lambda row: (
            float(row.get("stable_score", float("-inf"))),
            float(row.get("year_ann_min", float("-inf"))),
            float(row.get("year_ann_mean", float("-inf"))),
            float(row.get("full_ann", float("-inf"))),
        ),
        reverse=True,
    )
    summary = {
        "run_mode": "stage2_signal_cv_validation_shard",
        "selection_scope": "validation_only",
        "signal_profile": args.signal_profile,
        "strategy_profile": args.strategy_profile,
        "segments": SEGMENTS,
        "stability_weights": STABILITY_WEIGHTS,
        "candidate_count_total": len(all_candidates),
        "candidate_count_this_shard": len(shard_candidates),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "reused_count": reused_count,
        "evaluated_count": evaluated_count,
        "output_csv_path": str(output_csv_path),
        "best_validation_selected": success[0] if success else None,
        "top10_validation": success[:10],
    }
    output_summary_path.write_text(json.dumps(fast._jsonable(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    print(output_summary_path)


def merge_shards(args: argparse.Namespace) -> None:
    output_prefix = args.output_prefix.resolve()
    frames = []
    shard_paths = []
    for shard_index in range(args.num_shards):
        shard_prefix = _effective_prefix(output_prefix, num_shards=args.num_shards, shard_index=shard_index)
        shard_path = _grid_path(shard_prefix)
        if not shard_path.exists():
            raise FileNotFoundError(f"missing shard grid: {shard_path}")
        shard_paths.append(str(shard_path))
        frames.append(pd.read_csv(shard_path))
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["trial_name"], keep="last")
    merged = merged.sort_values(SORT_COLUMNS, ascending=[False, False, False, False], na_position="last")
    output_csv_path = _grid_path(output_prefix)
    output_summary_path = _summary_path(output_prefix)
    output_csv_path.write_text(merged.to_csv(index=False), encoding="utf-8")
    success = merged[merged["status"] == "success"]
    summary = {
        "run_mode": "stage2_signal_cv_validation_merged",
        "selection_scope": "validation_only",
        "segments": SEGMENTS,
        "stability_weights": STABILITY_WEIGHTS,
        "num_shards": args.num_shards,
        "shard_grid_paths": shard_paths,
        "grid_rows": int(len(merged)),
        "success_rows": int(len(success)),
        "output_csv_path": str(output_csv_path),
        "best_validation_selected": success.iloc[0].to_dict() if not success.empty else None,
        "top10_validation": success.head(10).to_dict(orient="records"),
    }
    output_summary_path.write_text(json.dumps(fast._jsonable(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    print(output_summary_path)


def launch_shards(args: argparse.Namespace) -> None:
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_prefix.resolve()
    procs = []
    for shard_index in range(args.num_shards):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output-prefix",
            str(output_prefix),
            "--result-suffix",
            args.result_suffix,
            "--signal-profile",
            args.signal_profile,
            "--strategy-profile",
            args.strategy_profile,
            "--stage1-validation-summary-path",
            str(args.stage1_validation_summary_path),
            "--stage1-repro-json-path",
            str(args.stage1_repro_json_path),
            "--num-shards",
            str(args.num_shards),
            "--shard-index",
            str(shard_index),
        ]
        if args.force_recompute:
            cmd.append("--force-recompute")
        log_path = log_dir / f"{output_prefix.name}_shard{shard_index:02d}of{args.num_shards:02d}.log"
        log_file = log_path.open("w", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, cwd=BASE_DIR, stdout=log_file, stderr=subprocess.STDOUT), log_file, log_path))
        print(f"started shard {shard_index}/{args.num_shards}: {log_path}")

    failed = []
    for proc, log_file, log_path in procs:
        return_code = proc.wait()
        log_file.close()
        if return_code != 0:
            failed.append((return_code, log_path))
    if failed:
        raise RuntimeError(f"failed shards: {failed}")
    merge_shards(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only yearly CV search for stage2 signals.")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--result-suffix", default="stage2_signal_cv")
    parser.add_argument(
        "--signal-profile",
        choices=["robust-tiny", "robust-small", "announcement-small", "cash-announcement-small"],
        default="robust-tiny",
    )
    parser.add_argument("--strategy-profile", choices=["stable-tiny", "stable-core"], default="stable-tiny")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--launch-shards", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=BASE_DIR / "logs" / "stage2_signal_cv")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--stage1-validation-summary-path",
        type=Path,
        default=stage2.TRA_BEST_VALIDATION_SUMMARY_PATH,
        help="stage1 validation wrapper summary produced by this run",
    )
    parser.add_argument(
        "--stage1-repro-json-path",
        type=Path,
        default=stage2.TRA_BEST_REPRO_JSON_PATH,
        help="stage1 rank-ensemble repro JSON produced by this run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    if args.launch_shards:
        launch_shards(args)
    elif args.merge_shards:
        merge_shards(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
