from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import h200_strict_validation_selector as strict_selector
import h200_validation_stress_audit as stress_audit
import run_rank_ensemble_tra_alpha360_once as rank_runner
import run_stage2_signal_cv_search as cv
import run_stage2_signal_search_fast as fast
import tune_alpha360_tra_fundamental_xgb_weight_refine as stage2


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp/h200_quarter_stability_selector_20260706_1"
DEFAULT_TRIAL_NAMES = [
    "baseline_seq__prac_m000_hold5_r08",
    "baseline_seq__prac_m000_hold5_r085",
    "baseline_seq__prac_m000_hold5_r09",
    "baseline_seq__prac_m000_hold6_r08",
    "baseline_seq__prac_m000_hold6_r085",
    "baseline_seq__prac_m000_hold6_r09",
    "baseline_seq__prac_m000_hold8_r08",
    "baseline_seq__prac_m000_hold8_r085",
    "baseline_seq__prac_m000_hold8_r09",
    "value_quality_hibonus_hq08_b002__prac_m000_hold5_r08",
    "value_quality_hibonus_hq08_b002__prac_m000_hold5_r085",
    "value_quality_hibonus_hq08_b002__prac_m000_hold6_r08",
    "value_quality_hibonus_hq08_b002__prac_m000_hold6_r085",
    "value_quality_hibonus_hq08_b002__prac_m000_hold8_r08",
    "value_quality_hibonus_hq08_b002__prac_m000_hold8_r085",
    "value_quality_twoside_lq02_p01_hq085_b002__prac_m000_hold5_r08",
    "value_quality_twoside_lq02_p01_hq085_b002__prac_m000_hold5_r085",
    "value_quality_twoside_lq02_p01_hq085_b002__prac_m000_hold6_r08",
    "value_quality_twoside_lq02_p01_hq085_b002__prac_m000_hold6_r085",
    "cash_quality_z_tw001__prac_m000_hold7_r085",
    "cash_quality_z_tw001__prac_m000_hold7_r09",
]
SORT_COLUMNS = [
    "stress_gate_pass",
    "stress_score",
    "quarter_ann_min",
    "quarter_ann_positive_ratio",
    "year_ann_min",
    "full_ann",
]


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_combo(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _split_csv(value))


def _rank_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(
        [column for column in SORT_COLUMNS if column in df.columns],
        ascending=[False] * len([column for column in SORT_COLUMNS if column in df.columns]),
        na_position="last",
    )


def _candidate_by_name(seq_pred: pd.DataFrame, seq_label: pd.DataFrame, profile: str) -> dict[str, dict[str, Any]]:
    return {candidate["signal_name"]: candidate for candidate in cv._build_cv_signal_candidates(seq_pred, seq_label, profile=profile)}


def _evaluate_candidate(
    *,
    combo_text: str,
    valid_rows: int,
    tra_validation: dict[str, Any],
    signal: dict[str, Any],
    strategy: dict[str, Any],
    trial_name: str,
    segments: dict[str, tuple[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    signal_name, strategy_name = trial_name.split("__", 1)
    selector_key = f"{combo_text}::{trial_name}"
    for segment_name, date_range in segments.items():
        metrics = cv._evaluate_segment(
            tra_validation,
            strategy,
            signal["pred"].sort_index(),
            signal["label"].sort_index(),
            segment=date_range,
        )
        details.append(
            {
                "selector_key": selector_key,
                "combo": combo_text,
                "trial_name": trial_name,
                "signal_name": signal_name,
                "strategy_trial": strategy_name,
                "strategy_profile": strategy.get("strategy_profile"),
                "segment_name": segment_name,
                "segment_group": stress_audit._group_for_segment(segment_name),
                "start": date_range[0],
                "end": date_range[1],
                **metrics,
            }
        )

    summary: dict[str, Any] = {
        "selector_key": selector_key,
        "combo": combo_text,
        "trial_name": trial_name,
        "signal_name": signal_name,
        "strategy_trial": strategy_name,
        "strategy_profile": strategy.get("strategy_profile"),
        "kind": signal.get("kind"),
        "valid_common_rows": int(valid_rows),
        "tab_weight": signal.get("tab_weight", pd.NA),
        "low_q": signal.get("low_q", pd.NA),
        "penalty": signal.get("penalty", pd.NA),
        "high_q": signal.get("high_q", pd.NA),
        "bonus": signal.get("bonus", pd.NA),
        "ann_weight": signal.get("ann_weight", pd.NA),
    }
    for group in ("full", "year", "half", "quarter", "month"):
        summary.update(stress_audit._summarize_group(details, group))
    summary["full_ann"] = summary.get("full_ann_mean")
    summary["year_ann_min"] = summary.get("year_ann_min")
    summary["stress_score"] = (
        float(summary.get("year_ann_min", -9.0)) * 500
        + float(summary.get("half_ann_min", -9.0)) * 250
        + float(summary.get("quarter_ann_positive_ratio", 0.0)) * 100
        + float(summary.get("quarter_rankic_positive_ratio", 0.0)) * 50
        - max(0.0, -float(summary.get("quarter_ann_min", 0.0))) * 150
    )
    summary["stress_gate_pass"] = stress_audit._stress_gate(summary, stress_audit.DEFAULT_STRESS_GATE)
    return summary, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="H200 validation-only selector with quarter stability in the selection loop.")
    parser.add_argument("--valid-summary-path", action="append", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--combo", action="append", required=True)
    parser.add_argument("--signal-profile", default="robust-small")
    parser.add_argument("--strategy-profiles", default="stable-core,stable-quarter,stable-cash-quick")
    parser.add_argument("--trial-name", action="append", default=None)
    parser.add_argument("--all-profile-trials", action="store_true")
    parser.add_argument("--segment-levels", default="year,half,quarter")
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    grid_path = args.output_prefix.with_name(f"{args.output_prefix.name}_grid.csv")
    detail_path = args.output_prefix.with_name(f"{args.output_prefix.name}_segment_detail.csv")
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.json")

    if not args.force_recompute and (grid_path.exists() or detail_path.exists() or summary_path.exists()):
        raise FileExistsError("quarter stability output already exists; use --force-recompute to overwrite")

    valid_summary_paths = args.valid_summary_path or strict_selector.DEFAULT_VALID_SUMMARIES
    valid_seed_runs = stress_audit._load_valid_seed_runs(valid_summary_paths)
    strategies = stress_audit._strategy_by_name(_split_csv(args.strategy_profiles))
    segments = stress_audit._date_segments(_split_csv(args.segment_levels))
    trial_names = args.trial_name or DEFAULT_TRIAL_NAMES
    mod = rank_runner.load_module()
    tra_validation = {
        "window_key": "w1",
        "train": ["2012-01-01", "2019-12-31"],
        "valid": ["2020-01-01", "2022-12-31"],
        "test": ["2023-01-01", "2026-04-01"],
        "handler_kwargs_extra": stage2.TRA_RANK_ENSEMBLE_HANDLER_KWARGS,
    }

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for combo in [_parse_combo(item) for item in args.combo]:
        missing = [seed for seed in combo if seed not in valid_seed_runs]
        if missing:
            raise RuntimeError(f"combo missing validation seed runs: combo={combo} missing={missing}")
        combo_text = ",".join(str(seed) for seed in combo)
        valid_pred, valid_label, valid_rows = strict_selector._rank_ensemble(mod, valid_seed_runs, combo)
        candidates = _candidate_by_name(valid_pred.sort_index(), valid_label.sort_index(), args.signal_profile)
        active_trials = (
            [f"{signal_name}__{strategy_name}" for signal_name in candidates for strategy_name in strategies]
            if args.all_profile_trials
            else trial_names
        )
        for trial_name in active_trials:
            signal_name, strategy_name = trial_name.split("__", 1)
            if signal_name not in candidates:
                print(f"skip missing signal in {args.signal_profile}: {trial_name}")
                continue
            if strategy_name not in strategies:
                print(f"skip missing strategy in {args.strategy_profiles}: {trial_name}")
                continue
            summary, segment_details = _evaluate_candidate(
                combo_text=combo_text,
                valid_rows=valid_rows,
                tra_validation=tra_validation,
                signal=candidates[signal_name],
                strategy=strategies[strategy_name],
                trial_name=trial_name,
                segments=segments,
            )
            rows.append(summary)
            details.extend(segment_details)
            _rank_rows(pd.DataFrame(rows)).to_csv(grid_path, index=False)
            pd.DataFrame(details).to_csv(detail_path, index=False)
            print(
                f"combo={combo_text} trial={trial_name} stress_pass={summary['stress_gate_pass']} "
                f"full_ann={summary.get('full_ann')} year_min={summary.get('year_ann_min')} "
                f"quarter_min={summary.get('quarter_ann_min')} quarter_pos={summary.get('quarter_ann_positive_ratio')}"
            )

    ranked = _rank_rows(pd.DataFrame(rows))
    ranked.to_csv(grid_path, index=False)
    pd.DataFrame(details).to_csv(detail_path, index=False)
    summary_payload = {
        "run_mode": "h200_quarter_stability_selector",
        "selection_scope": "validation_only_no_test_loaded",
        "test_usage_policy": "test_not_loaded_not_used_for_selection_final_test_may_run_once_after_explicit_lock",
        "valid_summary_paths": [str(path) for path in valid_summary_paths],
        "combos": args.combo,
        "signal_profile": args.signal_profile,
        "strategy_profiles": _split_csv(args.strategy_profiles),
        "all_profile_trials": bool(args.all_profile_trials),
        "trial_names": "all_profile_trials" if args.all_profile_trials else trial_names,
        "segment_levels": _split_csv(args.segment_levels),
        "stress_gate": stress_audit.DEFAULT_STRESS_GATE,
        "row_count": int(len(ranked)),
        "stress_gate_pass_count": int(ranked["stress_gate_pass"].sum()) if not ranked.empty else 0,
        "grid_path": str(grid_path),
        "segment_detail_path": str(detail_path),
        "top_validation_stress": ranked.head(20).to_dict(orient="records") if not ranked.empty else [],
    }
    summary_path.write_text(json.dumps(fast._jsonable(summary_payload), indent=2, ensure_ascii=True), encoding="utf-8")
    print(grid_path)
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
