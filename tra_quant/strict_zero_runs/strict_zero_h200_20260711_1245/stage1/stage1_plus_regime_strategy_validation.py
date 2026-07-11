from __future__ import annotations

import argparse
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
from stage1_plus_conservative_fusion_validation import (
    ANNUAL_SEGMENTS,
    DEFAULT_OUTPUT_PREFIX,
    DEFAULT_VALIDATION_SUMMARY,
    INCUMBENT_STRATEGY,
    QUARTER_SEGMENTS,
    _build_fusion_candidates,
    _evaluate_segment,
    _jsonable,
    _load_seed_cache_paths,
    _strategy_by_name,
    _summarize_candidate,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REGIME_OUTPUT_PREFIX = BASE_DIR / "tmp/stage1_plus_regime_strategy_validation_20260708_1"
REGIME_CLASS = "RegimeAwarePracticalTopkDropoutStrategy"
REGIME_MODULE = "qlib.contrib.strategy.custom_signal_strategy"


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_trend_pairs(raw: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        short, long = item.split(":", 1)
        pairs.append((int(short), int(long)))
    return pairs


def _float_tag(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "")


def _risk_tag(value: float) -> str:
    if value == 1.0:
        return "r10"
    return f"r{str(value).replace('.', '')}"


def _n_drop_tag(n_drop: int) -> str:
    return "" if n_drop == 1 else f"_drop{n_drop}"


def _build_regime_strategy_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for risk_degree in _parse_float_list(args.risk_degrees):
        for n_drop in _parse_int_list(args.n_drops):
            for hold_thresh in _parse_int_list(args.hold_threshes):
                for score_margin in _parse_float_list(args.score_margins):
                    for trend_short, trend_long in _parse_trend_pairs(args.trend_pairs):
                        for vol_thresh in _parse_float_list(args.vol_thresholds):
                            for caution_risk in _parse_float_list(args.caution_risks):
                                for defensive_min_hold in _parse_int_list(args.defensive_min_holds):
                                    for defensive_action in args.defensive_actions.split(","):
                                        defensive_action = defensive_action.strip()
                                        if not defensive_action:
                                            continue
                                        margin_tag = f"m{_float_tag(score_margin)}"
                                        trial_name = (
                                            f"regprac_{margin_tag}{_n_drop_tag(n_drop)}"
                                            f"_hold{hold_thresh}_{_risk_tag(risk_degree)}"
                                            f"_ts{trend_short}tl{trend_long}"
                                            f"_v{_float_tag(vol_thresh)}"
                                            f"_cr{_float_tag(caution_risk)}"
                                            f"_dh{defensive_min_hold}_{defensive_action}"
                                        )
                                        trials.append(
                                            {
                                                "trial_name": trial_name,
                                                "strategy_class": REGIME_CLASS,
                                                "strategy_module_path": REGIME_MODULE,
                                                "n_drop": n_drop,
                                                "risk_degree": risk_degree,
                                                "kwargs_extra": {
                                                    "score_margin": score_margin,
                                                    "hold_thresh": hold_thresh,
                                                    "slot_budget_ratio": 1.0,
                                                    "benchmark": "SH000300",
                                                    "trend_short_window": trend_short,
                                                    "trend_long_window": trend_long,
                                                    "vol_window": args.vol_window,
                                                    "vol_thresh": vol_thresh,
                                                    "caution_risk_degree": caution_risk,
                                                    "defensive_risk_degree": args.defensive_risk_degree,
                                                    "defensive_action": defensive_action,
                                                    "defensive_min_hold": defensive_min_hold,
                                                },
                                            }
                                        )
    if args.max_strategies > 0:
        trials = trials[: args.max_strategies]
    return trials


def _write_grid(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    sort_cols = [
        col
        for col in ["stable_over_stage1_pass", "annual_prepass", "selection_score", "delta_full_ann"]
        if col in df.columns
    ]
    ascending = [False for _ in sort_cols]
    df.sort_values(sort_cols, ascending=ascending, na_position="last").to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only Stage1 regime strategy selector.")
    parser.add_argument("--validation-summary-path", type=Path, default=DEFAULT_VALIDATION_SUMMARY)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_REGIME_OUTPUT_PREFIX)
    parser.add_argument("--incumbent-strategy", default=INCUMBENT_STRATEGY)
    parser.add_argument("--signals", default="baseline_seq,growth_quality_linear_rank_w0001")
    parser.add_argument("--growth-weight", type=float, default=0.001)
    parser.add_argument("--risk-degrees", default="0.95")
    parser.add_argument("--n-drops", default="3")
    parser.add_argument("--hold-threshes", default="4")
    parser.add_argument("--score-margins", default="0.0")
    parser.add_argument("--trend-pairs", default="10:30,20:60")
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--vol-thresholds", default="0.018,0.022,0.026")
    parser.add_argument("--caution-risks", default="0.35,0.50,0.70")
    parser.add_argument("--defensive-risk-degree", type=float, default=0.0)
    parser.add_argument("--defensive-actions", default="liquidate")
    parser.add_argument("--defensive-min-holds", default="1,3")
    parser.add_argument("--max-strategies", type=int, default=0)
    parser.add_argument("--max-quarter-candidates", type=int, default=16)
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
    valid_cache_paths = _load_seed_cache_paths(args.validation_summary_path.expanduser().resolve(), "seed_runs_valid")
    valid_pred, valid_label, valid_rows = rank_runner.rank_ensemble_signal(mod, valid_cache_paths)
    all_signal_candidates = _build_fusion_candidates(
        valid_pred.sort_index(),
        valid_label.sort_index(),
        [args.growth_weight],
        {"growth_quality"},
    )
    selected_signal_names = {item.strip() for item in args.signals.split(",") if item.strip()}
    signal_candidates = [item for item in all_signal_candidates if item["candidate"] in selected_signal_names]
    missing_signals = selected_signal_names.difference({item["candidate"] for item in signal_candidates})
    if missing_signals:
        raise RuntimeError(f"unknown signal candidate(s): {sorted(missing_signals)}")

    incumbent = _strategy_by_name(mod, [args.incumbent_strategy])[args.incumbent_strategy]
    baseline_signal = next(item for item in all_signal_candidates if item["candidate"] == "baseline_seq")
    baseline_annual_segments = [
        _evaluate_segment(mod, incumbent, baseline_signal["pred"], baseline_signal["label"], name, date_range)
        for name, date_range in ANNUAL_SEGMENTS.items()
    ]
    baseline_quarter_segments = [
        _evaluate_segment(mod, incumbent, baseline_signal["pred"], baseline_signal["label"], name, date_range)
        for name, date_range in QUARTER_SEGMENTS.items()
    ]
    baseline_annual_summary = _summarize_candidate(
        baseline_signal,
        args.incumbent_strategy,
        baseline_annual_segments,
        None,
        args,
    )
    baseline_stress_summary = _summarize_candidate(
        baseline_signal,
        args.incumbent_strategy,
        baseline_annual_segments + baseline_quarter_segments,
        None,
        args,
    )

    segment_rows: list[dict[str, Any]] = []
    segment_rows.extend(
        {**row, "candidate": "baseline_seq", "strategy_trial": args.incumbent_strategy, "segment_group": "annual"}
        for row in baseline_annual_segments
    )
    segment_rows.extend(
        {**row, "candidate": "baseline_seq", "strategy_trial": args.incumbent_strategy, "segment_group": "quarter"}
        for row in baseline_quarter_segments
    )

    strategy_trials = _build_regime_strategy_trials(args)
    annual_rows: list[dict[str, Any]] = [baseline_annual_summary]
    for strategy in strategy_trials:
        strategy_name = strategy["trial_name"]
        for signal_candidate in signal_candidates:
            rows = [
                _evaluate_segment(mod, strategy, signal_candidate["pred"], signal_candidate["label"], name, date_range)
                for name, date_range in ANNUAL_SEGMENTS.items()
            ]
            segment_rows.extend(
                {
                    **row,
                    "candidate": signal_candidate["candidate"],
                    "strategy_trial": strategy_name,
                    "segment_group": "annual",
                }
                for row in rows
            )
            summary = _summarize_candidate(signal_candidate, strategy_name, rows, baseline_annual_summary, args)
            annual_rows.append(summary)
            _write_grid(annual_rows, annual_path)
            print(
                f"annual signal={signal_candidate['candidate']} strategy={strategy_name} "
                f"delta_full={summary['delta_full_ann']:.6f} "
                f"delta_year_lcb={summary['delta_year_ann_lcb']:.6f} "
                f"delta_mdd={summary['delta_full_mdd']:.6f} "
                f"prepass={summary['annual_prepass']}"
            )

    annual_df = pd.DataFrame(annual_rows).sort_values(
        ["annual_prepass", "selection_score", "delta_full_ann"],
        ascending=[False, False, False],
        na_position="last",
    )
    stress_keys: set[tuple[str, str]] = set()
    candidate_rows = annual_df[annual_df["strategy_trial"] != args.incumbent_strategy]
    for row in candidate_rows.head(args.max_quarter_candidates).to_dict(orient="records"):
        stress_keys.add((row["candidate"], row["strategy_trial"]))
    for row in candidate_rows[candidate_rows["annual_prepass"]].to_dict(orient="records"):
        stress_keys.add((row["candidate"], row["strategy_trial"]))

    signal_by_name = {item["candidate"]: item for item in signal_candidates}
    strategy_by_name = {item["trial_name"]: item for item in strategy_trials}
    stress_rows: list[dict[str, Any]] = [baseline_stress_summary]
    for candidate_name, strategy_name in sorted(stress_keys):
        signal_candidate = signal_by_name[candidate_name]
        strategy = strategy_by_name[strategy_name]
        annual_detail = [
            row
            for row in segment_rows
            if row["candidate"] == candidate_name
            and row["strategy_trial"] == strategy_name
            and row["segment_group"] == "annual"
        ]
        quarter_detail = [
            _evaluate_segment(mod, strategy, signal_candidate["pred"], signal_candidate["label"], name, date_range)
            for name, date_range in QUARTER_SEGMENTS.items()
        ]
        segment_rows.extend(
            {
                **row,
                "candidate": candidate_name,
                "strategy_trial": strategy_name,
                "segment_group": "quarter",
            }
            for row in quarter_detail
        )
        summary = _summarize_candidate(
            signal_candidate,
            strategy_name,
            annual_detail + quarter_detail,
            baseline_stress_summary,
            args,
        )
        stress_rows.append(summary)
        _write_grid(stress_rows, stress_path)
        print(
            f"stress signal={candidate_name} strategy={strategy_name} "
            f"pass={summary['stable_over_stage1_pass']} reason={summary['stable_gate_reason']} "
            f"qpos={summary['quarter_positive_ratio']:.3f}"
        )

    pd.DataFrame(segment_rows).to_csv(detail_path, index=False)
    _write_grid(annual_rows, annual_path)
    _write_grid(stress_rows, stress_path)
    stress_df = pd.DataFrame(stress_rows).sort_values(
        ["stable_over_stage1_pass", "annual_prepass", "selection_score", "delta_full_ann"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    pass_df = stress_df[stress_df["stable_over_stage1_pass"] == True] if not stress_df.empty else stress_df
    locked_candidate = pass_df.iloc[0].to_dict() if not pass_df.empty else None
    summary = {
        "run_mode": "stage1_plus_regime_strategy_validation_only",
        "selection_scope": "validation_only_no_test_loaded",
        "test_usage_policy": "test_not_loaded_not_used_for_selection",
        "hard_constraints": {"account": 150000, "topk": 5, "universe": "csi300"},
        "incumbent_strategy": args.incumbent_strategy,
        "strategy_class": REGIME_CLASS,
        "validation_summary_path": str(args.validation_summary_path),
        "validation_cache_paths": [str(path) for path in valid_cache_paths],
        "valid_common_rows": int(valid_rows),
        "signal_candidates": [item["candidate"] for item in signal_candidates],
        "strategy_count": len(strategy_trials),
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
        "baseline_stress": baseline_stress_summary,
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
        print(
            "locked_candidate_validation_only="
            f"{locked_candidate['candidate']} strategy={locked_candidate['strategy_trial']}"
        )


if __name__ == "__main__":
    main()
