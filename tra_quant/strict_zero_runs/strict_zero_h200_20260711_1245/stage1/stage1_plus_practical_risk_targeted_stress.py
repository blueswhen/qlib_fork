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
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp/stage1_plus_practical_risk_targeted_stress_20260708_1"


def _parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _risk_tag(value: float) -> str:
    if value == 1.0:
        return "r10"
    return f"r{str(value).replace('.', '')}"


def _write_grid(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    sort_cols = [
        col
        for col in ["stable_over_stage1_pass", "annual_prepass", "selection_score", "delta_full_ann"]
        if col in rows[0]
    ]
    pd.DataFrame(rows).sort_values(
        sort_cols,
        ascending=[False for _ in sort_cols],
        na_position="last",
    ).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only targeted stress for practical risk-degree variants."
    )
    parser.add_argument("--validation-summary-path", type=Path, default=DEFAULT_VALIDATION_SUMMARY)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--incumbent-strategy", default=INCUMBENT_STRATEGY)
    parser.add_argument("--signal", default="growth_quality_linear_rank_w0001")
    parser.add_argument("--growth-weight", type=float, default=0.001)
    parser.add_argument("--risk-degrees", default="0.875,0.925")
    parser.add_argument("--n-drop", type=int, default=3)
    parser.add_argument("--hold-thresh", type=int, default=4)
    parser.add_argument("--score-margin", type=float, default=0.0)
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
    stress_path = args.output_prefix.with_name(f"{args.output_prefix.name}_stress_grid.csv")
    detail_path = args.output_prefix.with_name(f"{args.output_prefix.name}_segment_detail.csv")
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.json")
    outputs = [stress_path, detail_path, summary_path]
    if not args.force_recompute and any(path.exists() for path in outputs):
        raise FileExistsError("output already exists; use --force-recompute")

    mod = rank_runner.load_module()
    qlib.init(provider_uri=mod.PROVIDER_URI, region=REG_CN)
    valid_cache_paths = _load_seed_cache_paths(args.validation_summary_path.expanduser().resolve(), "seed_runs_valid")
    valid_pred, valid_label, valid_rows = rank_runner.rank_ensemble_signal(mod, valid_cache_paths)
    signal_candidates = _build_fusion_candidates(
        valid_pred.sort_index(),
        valid_label.sort_index(),
        [args.growth_weight],
        {"growth_quality"},
    )
    candidate_by_name = {item["candidate"]: item for item in signal_candidates}
    if args.signal not in candidate_by_name:
        raise RuntimeError(f"unknown signal candidate: {args.signal}")

    baseline_signal = candidate_by_name["baseline_seq"]
    target_signal = candidate_by_name[args.signal]
    incumbent = _strategy_by_name(mod, [args.incumbent_strategy])[args.incumbent_strategy]

    baseline_segments = [
        _evaluate_segment(mod, incumbent, baseline_signal["pred"], baseline_signal["label"], name, date_range)
        for name, date_range in {**ANNUAL_SEGMENTS, **QUARTER_SEGMENTS}.items()
    ]
    segment_rows: list[dict[str, Any]] = [
        {
            **row,
            "candidate": "baseline_seq",
            "strategy_trial": args.incumbent_strategy,
            "segment_group": "annual" if row["segment_name"].startswith(("valid", "Y")) else "quarter",
        }
        for row in baseline_segments
    ]
    baseline_summary = _summarize_candidate(
        baseline_signal,
        args.incumbent_strategy,
        baseline_segments,
        None,
        args,
    )

    stress_rows: list[dict[str, Any]] = [baseline_summary]
    risk_degrees = _parse_float_list(args.risk_degrees)
    for risk_degree in risk_degrees:
        strategy_name = (
            f"prac_m000_drop{args.n_drop}_hold{args.hold_thresh}_{_risk_tag(risk_degree)}"
        )
        strategy = mod._prac(
            strategy_name,
            args.n_drop,
            risk_degree,
            args.score_margin,
            args.hold_thresh,
        )
        rows = [
            _evaluate_segment(mod, strategy, target_signal["pred"], target_signal["label"], name, date_range)
            for name, date_range in {**ANNUAL_SEGMENTS, **QUARTER_SEGMENTS}.items()
        ]
        segment_rows.extend(
            {
                **row,
                "candidate": target_signal["candidate"],
                "strategy_trial": strategy_name,
                "segment_group": "annual" if row["segment_name"].startswith(("valid", "Y")) else "quarter",
            }
            for row in rows
        )
        summary = _summarize_candidate(target_signal, strategy_name, rows, baseline_summary, args)
        summary["risk_degree"] = float(risk_degree)
        stress_rows.append(summary)
        _write_grid(stress_rows, stress_path)
        print(
            f"stress strategy={strategy_name} "
            f"pass={summary['stable_over_stage1_pass']} reason={summary['stable_gate_reason']} "
            f"full={summary['full_ann']:.6f} d_full={summary['delta_full_ann']:.6f} "
            f"qpos={summary['quarter_positive_ratio']:.3f} "
            f"d_qmin={summary['delta_quarter_ann_min']:.6f} "
            f"d_qlcb={summary['delta_quarter_ann_lcb']:.6f}"
        )

    detail_df = pd.DataFrame(segment_rows)
    detail_df.to_csv(detail_path, index=False)
    _write_grid(stress_rows, stress_path)
    stress_df = pd.DataFrame(stress_rows).sort_values(
        ["stable_over_stage1_pass", "annual_prepass", "selection_score", "delta_full_ann"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    pass_df = stress_df[stress_df["stable_over_stage1_pass"] == True]
    locked_candidate = pass_df.iloc[0].to_dict() if not pass_df.empty else None
    summary = {
        "run_mode": "stage1_plus_practical_risk_targeted_stress_validation_only",
        "selection_scope": "validation_only_no_test_loaded",
        "test_usage_policy": "test_not_loaded_not_used_for_selection",
        "hard_constraints": {"account": int(mod.ACCOUNT), "topk": int(mod.TOPK), "universe": mod.INSTRUMENTS},
        "incumbent_strategy": args.incumbent_strategy,
        "target_signal": args.signal,
        "risk_degrees": risk_degrees,
        "validation_summary_path": str(args.validation_summary_path),
        "validation_cache_paths": [str(path) for path in valid_cache_paths],
        "valid_common_rows": int(valid_rows),
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
        "stress_grid_path": str(stress_path),
        "segment_detail_path": str(detail_path),
        "baseline_stress": baseline_summary,
        "stable_pass_count": int(len(pass_df)),
        "locked_candidate_validation_only": locked_candidate,
        "top_stress": stress_df.to_dict(orient="records"),
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
