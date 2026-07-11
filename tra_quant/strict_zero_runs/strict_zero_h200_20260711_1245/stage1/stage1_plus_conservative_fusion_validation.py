from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

WORKSPACE_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_FOR_IMPORT))

import pandas as pd
import qlib
from qlib.constant import REG_CN

import run_rank_ensemble_tra_alpha360_once as rank_runner


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = BASE_DIR.parent.parent
RUN_DIR = WORKSPACE / "tra_quant/runs/stage1_h200_20260621_11_4seed_b12000_fast/stage1"
DEFAULT_VALIDATION_SUMMARY = (
    RUN_DIR / "tmp/rank_ensemble_h200_validation_summary_stage1_h200_20260621_11_4seed_b12000_fast.txt"
)
DEFAULT_BASELINE_GRID = (
    RUN_DIR / "tmp/rank_ensemble_h200_search_stage1_h200_20260621_11_4seed_b12000_fast_validation_grid.csv"
)
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp/stage1_plus_conservative_fusion_validation_20260707_1"
FUNDAMENTAL_FEATURE_PATH = (
    WORKSPACE / "training_data/processed/training/csi300_daily_fundamental_features_rankpct_qlib.parquet"
)
INCUMBENT_STRATEGY = "prac_m000_drop3_hold4_r095"

ANNUAL_SEGMENTS = {
    "valid_all": ("2020-01-01", "2022-12-31"),
    "Y2020": ("2020-01-01", "2020-12-31"),
    "Y2021": ("2021-01-01", "2021-12-31"),
    "Y2022": ("2022-01-01", "2022-12-31"),
}
QUARTER_SEGMENTS = {
    f"Q{quarter}_{year}": (f"{year}-{start}", f"{year}-{end}")
    for year in (2020, 2021, 2022)
    for quarter, start, end in (
        (1, "01-01", "03-31"),
        (2, "04-01", "06-30"),
        (3, "07-01", "09-30"),
        (4, "10-01", "12-31"),
    )
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _localize_path(raw: str | Path) -> Path:
    text = str(raw)
    replacements = {
        "/home/niushengxiao/qlib_fork": str(WORKSPACE),
        "/home/blueswhen/DL/qlib_fork": str(WORKSPACE),
        "/home/blueswhen/DL/qlib/examples/my_strategy": str(BASE_DIR),
    }
    for old, new in replacements.items():
        if text == old or text.startswith(old + "/"):
            text = new + text[len(old) :]
            break
    return Path(text).expanduser().resolve()


def _parse_summary(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            out[key] = ast.literal_eval(value)
        except Exception:
            out[key] = value
    return out


def _load_seed_cache_paths(summary_path: Path, key: str) -> list[Path]:
    data = _parse_summary(summary_path)
    runs = data.get(key)
    if not runs:
        raise RuntimeError(f"{key} is missing or empty in {summary_path}")
    paths = []
    for row in runs:
        path = _localize_path(row["cache_path"])
        if not path.exists():
            raise FileNotFoundError(f"missing localized cache path: {path}")
        paths.append(path)
    return paths


def _date_level(index: pd.Index) -> str | int:
    return "datetime" if "datetime" in index.names else 0


def _slice_frame(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = df.index.get_level_values(_date_level(df.index))
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return df.loc[mask].sort_index()


def _ensure_score(df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        return df.to_frame("score")
    if "score" in df.columns:
        return df[["score"]].copy()
    return df.iloc[:, [0]].rename(columns={df.columns[0]: "score"})


def _daily_rank_pct(df: pd.DataFrame) -> pd.DataFrame:
    score = _ensure_score(df).sort_index()
    level = _date_level(score.index)
    return score.groupby(level=level, group_keys=False).rank(method="average", pct=True)


def _load_fundamental(index: pd.Index) -> pd.DataFrame:
    fund = pd.read_parquet(FUNDAMENTAL_FEATURE_PATH)
    if isinstance(fund.columns, pd.MultiIndex):
        fund.columns = fund.columns.get_level_values(-1)
    common = index.intersection(fund.index).sort_values()
    if common.empty:
        raise RuntimeError("fundamental feature index has no overlap with stage1 validation signal")
    return fund.loc[common].sort_index()


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


def _fundamental_composites(index: pd.Index) -> dict[str, pd.DataFrame]:
    fund = _load_fundamental(index)
    return {
        "cash_quality": _composite(
            fund,
            plus=("fi_ocf_yoy", "cf_n_cashflow_act", "fi_netprofit_margin", "fi_roe", "fi_assets_turn"),
            minus=("fi_debt_to_assets",),
        ),
        "value_quality": _composite(
            fund,
            plus=("dv_ttm", "dv_ratio", "fi_bps", "fi_roe", "fi_netprofit_margin"),
            minus=("pe_ttm", "pe", "pb", "ps_ttm", "ps", "fi_debt_to_assets"),
        ),
        "growth_quality": _composite(
            fund,
            plus=("fi_q_sales_yoy", "fi_tr_yoy", "fi_or_yoy", "fi_assets_yoy", "fi_eqt_yoy", "fi_roe"),
            minus=("fi_debt_to_assets",),
        ),
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


def _build_fusion_candidates(
    seq_pred: pd.DataFrame,
    seq_label: pd.DataFrame,
    weights: list[float],
    selected_overlays: set[str] | None = None,
) -> list[dict[str, Any]]:
    common = seq_pred.index.intersection(seq_label.index).sort_values()
    base_rank = _daily_rank_pct(seq_pred.loc[common])
    label = seq_label.loc[common].sort_index()
    candidates: list[dict[str, Any]] = [
        {
            "candidate": "baseline_seq",
            "kind": "baseline",
            "overlay": None,
            "weight": 0.0,
            "pred": base_rank,
            "label": label,
        }
    ]
    composites = _fundamental_composites(common)
    if selected_overlays is not None:
        missing = sorted(selected_overlays.difference(composites))
        if missing:
            raise RuntimeError(f"unknown fundamental overlay(s): {missing}")
    for overlay_name, raw in composites.items():
        if selected_overlays is not None and overlay_name not in selected_overlays:
            continue
        overlay_common = common.intersection(raw.index).sort_values()
        overlay_rank = _daily_rank_pct(raw.loc[overlay_common])
        base = base_rank.loc[overlay_common]
        for weight in weights:
            pred = base.mul(1.0 - weight).add(overlay_rank.mul(weight), fill_value=0.0)
            candidates.append(
                {
                    "candidate": f"{overlay_name}_linear_rank_w{str(weight).replace('.', '')}",
                    "kind": "linear_rank_residual",
                    "overlay": overlay_name,
                    "weight": float(weight),
                    "pred": pred.sort_index(),
                    "label": label.loc[overlay_common].sort_index(),
                }
            )
    return candidates


def _strategy_by_name(mod: Any, names: list[str]) -> dict[str, dict[str, Any]]:
    trials = rank_runner.build_candidates(mod, strategy_profile=rank_runner.STRATEGY_PROFILE_TRA_BEST_72)
    by_name = {trial["trial_name"]: trial for trial in trials}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise RuntimeError(f"unknown strategy trial(s): {missing}")
    return {name: by_name[name] for name in names}


def _evaluate_segment(
    mod: Any,
    strategy: dict[str, Any],
    pred: pd.DataFrame,
    label: pd.DataFrame,
    segment_name: str,
    date_range: tuple[str, str],
) -> dict[str, Any]:
    metrics = rank_runner._evaluate_direct_signal(
        mod,
        strategy,
        pred,
        label,
        window_key="w1",
        eval_segment="valid",
        backtest_segment="valid",
        eval_range=date_range,
        backtest_range=date_range,
    )
    return {
        "segment_name": segment_name,
        "start": date_range[0],
        "end": date_range[1],
        "rows": int(len(_slice_frame(pred, *date_range).index.intersection(_slice_frame(label, *date_range).index))),
        "ann": float(metrics["with_cost_ann_return"]),
        "ir": float(metrics["with_cost_ir"]),
        "mdd": float(metrics["with_cost_mdd"]),
        "IC": float(metrics["IC"]),
        "Rank_IC": float(metrics["Rank IC"]),
    }


def _summarize_candidate(
    candidate: dict[str, Any],
    strategy_name: str,
    segment_rows: list[dict[str, Any]],
    baseline_row: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    full = next(row for row in segment_rows if row["segment_name"] == "valid_all")
    years = [row for row in segment_rows if row["segment_name"].startswith("Y")]
    quarters = [row for row in segment_rows if row["segment_name"].startswith("Q")]
    year_ann = pd.Series([row["ann"] for row in years], dtype="float64")
    quarter_ann = pd.Series([row["ann"] for row in quarters], dtype="float64") if quarters else pd.Series(dtype="float64")
    quarter_rankic = (
        pd.Series([row["Rank_IC"] for row in quarters], dtype="float64") if quarters else pd.Series(dtype="float64")
    )
    row = {
        "candidate": candidate["candidate"],
        "kind": candidate["kind"],
        "overlay": candidate["overlay"],
        "weight": candidate["weight"],
        "strategy_trial": strategy_name,
        "full_ann": float(full["ann"]),
        "full_ir": float(full["ir"]),
        "full_mdd": float(full["mdd"]),
        "full_Rank_IC": float(full["Rank_IC"]),
        "year_ann_min": float(year_ann.min()),
        "year_ann_mean": float(year_ann.mean()),
        "year_ann_std": float(year_ann.std(ddof=0)),
        "year_ann_lcb": float(year_ann.mean() - year_ann.std(ddof=0)),
    }
    if not quarters:
        row.update(
            {
                "quarter_ann_min": None,
                "quarter_ann_mean": None,
                "quarter_ann_std": None,
                "quarter_ann_lcb": None,
                "quarter_positive_ratio": None,
                "quarter_Rank_IC_min": None,
                "quarter_Rank_IC_positive_ratio": None,
            }
        )
    else:
        row.update(
            {
                "quarter_ann_min": float(quarter_ann.min()),
                "quarter_ann_mean": float(quarter_ann.mean()),
                "quarter_ann_std": float(quarter_ann.std(ddof=0)),
                "quarter_ann_lcb": float(quarter_ann.mean() - quarter_ann.std(ddof=0)),
                "quarter_positive_ratio": float((quarter_ann > 0).mean()),
                "quarter_Rank_IC_min": float(quarter_rankic.min()),
                "quarter_Rank_IC_positive_ratio": float((quarter_rankic > 0).mean()),
            }
        )
    if baseline_row is None:
        row.update(
            {
                "delta_full_ann": 0.0,
                "delta_year_ann_min": 0.0,
                "delta_year_ann_lcb": 0.0,
                "delta_quarter_ann_min": 0.0 if quarters else None,
                "delta_quarter_ann_lcb": 0.0 if quarters else None,
                "delta_full_mdd": 0.0,
                "annual_prepass": True,
                "stable_over_stage1_pass": False,
                "stable_gate_reason": "baseline_reference",
            }
        )
    else:
        row["delta_full_ann"] = row["full_ann"] - baseline_row["full_ann"]
        row["delta_year_ann_min"] = row["year_ann_min"] - baseline_row["year_ann_min"]
        row["delta_year_ann_lcb"] = row["year_ann_lcb"] - baseline_row["year_ann_lcb"]
        row["delta_full_mdd"] = row["full_mdd"] - baseline_row["full_mdd"]
        if quarters and baseline_row.get("quarter_ann_min") is not None:
            row["delta_quarter_ann_min"] = row["quarter_ann_min"] - baseline_row["quarter_ann_min"]
            row["delta_quarter_ann_lcb"] = row["quarter_ann_lcb"] - baseline_row["quarter_ann_lcb"]
        else:
            row["delta_quarter_ann_min"] = None
            row["delta_quarter_ann_lcb"] = None
        row["annual_prepass"] = (
            row["delta_full_ann"] >= args.min_full_ann_improvement
            and row["delta_year_ann_min"] >= args.min_year_min_improvement
            and row["delta_year_ann_lcb"] >= args.min_year_lcb_improvement
            and row["delta_full_mdd"] >= -args.max_mdd_deterioration
        )
        if not quarters:
            row["stable_over_stage1_pass"] = False
            row["stable_gate_reason"] = "quarter_stress_not_run"
        else:
            checks = [
                (row["annual_prepass"], "annual_prepass"),
                (row["delta_quarter_ann_min"] >= -args.max_quarter_min_deterioration, "quarter_min_not_worse"),
                (row["delta_quarter_ann_lcb"] >= -args.max_quarter_lcb_deterioration, "quarter_lcb_not_worse"),
                (row["quarter_positive_ratio"] >= args.min_quarter_positive_ratio, "quarter_positive_ratio"),
                (row["quarter_Rank_IC_positive_ratio"] >= args.min_quarter_rankic_positive_ratio, "quarter_rankic_positive_ratio"),
            ]
            failed = [name for ok, name in checks if not ok]
            row["stable_over_stage1_pass"] = not failed
            row["stable_gate_reason"] = "pass" if not failed else "failed:" + ",".join(failed)
    row["selection_score"] = (
        row["delta_year_ann_lcb"] * 500.0
        + row["delta_full_ann"] * 300.0
        + row["delta_year_ann_min"] * 200.0
        + (row.get("delta_quarter_ann_lcb") or 0.0) * 200.0
        + row["delta_full_mdd"] * 50.0
    )
    return row


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    pd.DataFrame(rows).sort_values(
        ["stable_over_stage1_pass", "annual_prepass", "selection_score", "delta_full_ann"],
        ascending=[False, False, False, False],
        na_position="last",
    ).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only conservative Stage1+ fusion selector.")
    parser.add_argument("--validation-summary-path", type=Path, default=DEFAULT_VALIDATION_SUMMARY)
    parser.add_argument("--baseline-grid-path", type=Path, default=DEFAULT_BASELINE_GRID)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--strategy-trial", action="append", default=[INCUMBENT_STRATEGY])
    parser.add_argument(
        "--overlay",
        action="append",
        help="Restrict fundamental overlays. Can be repeated; default runs all overlays.",
    )
    parser.add_argument("--weights", default="0.0025,0.005,0.01,0.015,0.02")
    parser.add_argument("--max-quarter-candidates", type=int, default=12)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--min-full-ann-improvement", type=float, default=0.005)
    parser.add_argument("--min-year-min-improvement", type=float, default=0.0)
    parser.add_argument("--min-year-lcb-improvement", type=float, default=0.0)
    parser.add_argument("--max-mdd-deterioration", type=float, default=0.015)
    parser.add_argument("--max-quarter-min-deterioration", type=float, default=0.0)
    parser.add_argument("--max-quarter-lcb-deterioration", type=float, default=0.0)
    parser.add_argument("--min-quarter-positive-ratio", type=float, default=0.75)
    parser.add_argument("--min-quarter-rankic-positive-ratio", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    annual_path = args.output_prefix.with_name(f"{args.output_prefix.name}_annual_grid.csv")
    stress_path = args.output_prefix.with_name(f"{args.output_prefix.name}_stress_grid.csv")
    detail_path = args.output_prefix.with_name(f"{args.output_prefix.name}_segment_detail.csv")
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.json")
    outputs = [annual_path, stress_path, detail_path, summary_path]
    if not args.force_recompute and any(path.exists() for path in outputs):
        raise FileExistsError("output already exists; use --force-recompute")

    mod = rank_runner.load_module()
    qlib.init(provider_uri=mod.PROVIDER_URI, region=REG_CN)
    weights = [float(item.strip()) for item in args.weights.split(",") if item.strip()]
    selected_overlays = set(args.overlay) if args.overlay else None
    strategies = _strategy_by_name(mod, args.strategy_trial)
    valid_cache_paths = _load_seed_cache_paths(args.validation_summary_path.expanduser().resolve(), "seed_runs_valid")
    valid_pred, valid_label, valid_rows = rank_runner.rank_ensemble_signal(mod, valid_cache_paths)
    candidates = _build_fusion_candidates(
        valid_pred.sort_index(),
        valid_label.sort_index(),
        weights,
        selected_overlays,
    )

    annual_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    baseline_by_strategy: dict[str, dict[str, Any]] = {}
    for strategy_name, strategy in strategies.items():
        baseline_candidate = next(item for item in candidates if item["candidate"] == "baseline_seq")
        baseline_segments = [
            _evaluate_segment(mod, strategy, baseline_candidate["pred"], baseline_candidate["label"], name, date_range)
            for name, date_range in ANNUAL_SEGMENTS.items()
        ]
        segment_rows.extend(
            {**row, "candidate": "baseline_seq", "strategy_trial": strategy_name, "segment_group": "annual"}
            for row in baseline_segments
        )
        baseline_summary = _summarize_candidate(baseline_candidate, strategy_name, baseline_segments, None, args)
        baseline_by_strategy[strategy_name] = baseline_summary
        annual_rows.append(baseline_summary)

        for candidate in candidates:
            if candidate["candidate"] == "baseline_seq":
                continue
            rows = [
                _evaluate_segment(mod, strategy, candidate["pred"], candidate["label"], name, date_range)
                for name, date_range in ANNUAL_SEGMENTS.items()
            ]
            segment_rows.extend(
                {**row, "candidate": candidate["candidate"], "strategy_trial": strategy_name, "segment_group": "annual"}
                for row in rows
            )
            annual_rows.append(_summarize_candidate(candidate, strategy_name, rows, baseline_summary, args))
            _write_csv(annual_rows, annual_path)
            print(
                f"annual candidate={candidate['candidate']} strategy={strategy_name} "
                f"delta_full={annual_rows[-1]['delta_full_ann']:.6f} "
                f"delta_year_lcb={annual_rows[-1]['delta_year_ann_lcb']:.6f} "
                f"prepass={annual_rows[-1]['annual_prepass']}"
            )

    annual_df = pd.DataFrame(annual_rows).sort_values(
        ["annual_prepass", "selection_score", "delta_full_ann"],
        ascending=[False, False, False],
        na_position="last",
    )
    stress_keys = set()
    for row in annual_df[annual_df["candidate"] != "baseline_seq"].head(args.max_quarter_candidates).to_dict(orient="records"):
        stress_keys.add((row["candidate"], row["strategy_trial"]))
    for row in annual_df[annual_df["annual_prepass"] & (annual_df["candidate"] != "baseline_seq")].to_dict(orient="records"):
        stress_keys.add((row["candidate"], row["strategy_trial"]))
    for strategy_name in strategies:
        stress_keys.add(("baseline_seq", strategy_name))

    candidate_by_name = {row["candidate"]: row for row in candidates}
    stress_rows: list[dict[str, Any]] = []
    stress_baseline_by_strategy: dict[str, dict[str, Any]] = {}
    for candidate_name, strategy_name in sorted(stress_keys):
        candidate = candidate_by_name[candidate_name]
        strategy = strategies[strategy_name]
        annual_detail = [
            row
            for row in segment_rows
            if row["candidate"] == candidate_name
            and row["strategy_trial"] == strategy_name
            and row["segment_group"] == "annual"
        ]
        quarter_detail = [
            _evaluate_segment(mod, strategy, candidate["pred"], candidate["label"], name, date_range)
            for name, date_range in QUARTER_SEGMENTS.items()
        ]
        segment_rows.extend(
            {**row, "candidate": candidate_name, "strategy_trial": strategy_name, "segment_group": "quarter"}
            for row in quarter_detail
        )
        baseline_row = None if candidate_name == "baseline_seq" else stress_baseline_by_strategy[strategy_name]
        stress_summary = _summarize_candidate(candidate, strategy_name, annual_detail + quarter_detail, baseline_row, args)
        stress_rows.append(stress_summary)
        if candidate_name == "baseline_seq":
            stress_baseline_by_strategy[strategy_name] = stress_summary
        _write_csv(stress_rows, stress_path)
        print(
            f"stress candidate={candidate_name} strategy={strategy_name} "
            f"pass={stress_rows[-1]['stable_over_stage1_pass']} reason={stress_rows[-1]['stable_gate_reason']}"
        )

    detail_path.write_text(pd.DataFrame(segment_rows).to_csv(index=False), encoding="utf-8")
    _write_csv(annual_rows, annual_path)
    _write_csv(stress_rows, stress_path)
    stress_df = pd.DataFrame(stress_rows).sort_values(
        ["stable_over_stage1_pass", "annual_prepass", "selection_score", "delta_full_ann"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    pass_df = stress_df[stress_df["stable_over_stage1_pass"] == True] if not stress_df.empty else stress_df
    locked_candidate = pass_df.iloc[0].to_dict() if not pass_df.empty else None
    summary = {
        "run_mode": "stage1_plus_conservative_fusion_validation_only",
        "selection_scope": "validation_only_no_test_loaded",
        "test_usage_policy": "test_not_loaded_not_used_for_selection",
        "hard_constraints": {"account": 150000, "topk": 5, "universe": "csi300"},
        "incumbent_strategy": INCUMBENT_STRATEGY,
        "strategy_trials": list(strategies),
        "validation_summary_path": str(args.validation_summary_path),
        "validation_cache_paths": [str(path) for path in valid_cache_paths],
        "valid_common_rows": int(valid_rows),
        "fundamental_feature_path": str(FUNDAMENTAL_FEATURE_PATH),
        "weights": weights,
        "candidate_count": len(candidates),
        "annual_segments": ANNUAL_SEGMENTS,
        "quarter_segments": QUARTER_SEGMENTS,
        "stable_over_stage1_gate": {
            "min_full_ann_improvement": args.min_full_ann_improvement,
            "min_year_min_improvement": args.min_year_min_improvement,
            "min_year_lcb_improvement": args.min_year_lcb_improvement,
            "max_mdd_deterioration": args.max_mdd_deterioration,
            "max_quarter_min_deterioration": args.max_quarter_min_deterioration,
            "max_quarter_lcb_deterioration": args.max_quarter_lcb_deterioration,
            "min_quarter_positive_ratio": args.min_quarter_positive_ratio,
            "min_quarter_rankic_positive_ratio": args.min_quarter_rankic_positive_ratio,
        },
        "annual_grid_path": str(annual_path),
        "stress_grid_path": str(stress_path),
        "segment_detail_path": str(detail_path),
        "baseline_rows": [row for row in stress_rows if row["candidate"] == "baseline_seq"],
        "stable_pass_count": int(len(pass_df)) if not stress_df.empty else 0,
        "locked_candidate_validation_only": locked_candidate,
        "top10_stress": stress_df.head(10).to_dict(orient="records") if not stress_df.empty else [],
        "final_test_eligible": locked_candidate is not None,
        "final_test_pending_reason": (
            "candidate_passed_validation_gate_can_be_locked_before_one_final"
            if locked_candidate is not None
            else "no_candidate_stably_exceeds_stage1_on_validation"
        ),
    }
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    print(summary_path)
    if locked_candidate is None:
        print("no validation candidate passed stable-over-stage1 gate")
    else:
        print(f"locked_candidate_validation_only={locked_candidate['candidate']} strategy={locked_candidate['strategy_trial']}")


if __name__ == "__main__":
    main()
