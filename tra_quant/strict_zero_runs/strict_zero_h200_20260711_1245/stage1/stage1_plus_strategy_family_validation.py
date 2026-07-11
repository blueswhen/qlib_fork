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
from stage1_plus_regime_strategy_validation import _n_drop_tag, _risk_tag, _write_grid


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp/stage1_plus_strategy_family_validation_20260708_1"
CUSTOM_MODULE = "qlib.contrib.strategy.custom_signal_strategy"


def _trial(
    *,
    name: str,
    strategy_class: str,
    n_drop: int = 3,
    risk_degree: float = 0.95,
    hold_thresh: int = 4,
    score_margin: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs_extra = {
        "score_margin": score_margin,
        "hold_thresh": hold_thresh,
        "slot_budget_ratio": 1.0,
    }
    if extra:
        kwargs_extra.update(extra)
    return {
        "trial_name": name,
        "strategy_class": strategy_class,
        "strategy_module_path": CUSTOM_MODULE,
        "n_drop": n_drop,
        "risk_degree": risk_degree,
        "kwargs_extra": kwargs_extra,
    }


def _focused_trials(mod: Any) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for vol_power in (0.5, 1.0):
        trials.append(
            _trial(
                name=f"invvol_w20_p{str(vol_power).replace('.', '')}_drop3_hold4_r095",
                strategy_class="InverseVolPracticalTopkDropoutStrategy",
                extra={"universe": "csi300", "vol_window": 20, "vol_power": vol_power},
            )
        )
    for smooth_alpha, hold_bonus in ((0.70, 0.001), (0.85, 0.001), (0.70, 0.0025)):
        trials.append(
            _trial(
                name=(
                    f"turnover_a{str(smooth_alpha).replace('.', '')}"
                    f"_hb{str(hold_bonus).replace('.', '')}_drop3_hold4_r095"
                ),
                strategy_class="TurnoverAwarePracticalTopkDropoutStrategy",
                extra={"smooth_alpha": smooth_alpha, "hold_bonus": hold_bonus},
            )
        )
    for strength, allocation_blend, use_rank_decay in ((0.25, 0.50, False), (0.50, 0.50, True)):
        trials.append(
            _trial(
                name=(
                    f"multitopk_3-5-10-20_s{str(strength).replace('.', '')}"
                    f"_ab{str(allocation_blend).replace('.', '')}"
                    f"_{'decay' if use_rank_decay else 'flat'}_drop3_hold4_r095"
                ),
                strategy_class="MultiTopkConsensusPracticalTopkDropoutStrategy",
                extra={
                    "topk_list": "3,5,10,20",
                    "consensus_strength": strength,
                    "allocation_blend": allocation_blend,
                    "use_rank_decay": use_rank_decay,
                },
            )
        )
    existing_by_name = getattr(mod, "STRATEGY_TRIAL_BY_NAME", {})
    practical_name = "prac_m000_drop3_hold4_r085"
    practical = existing_by_name.get(practical_name) or mod._prac(practical_name, 3, 0.85, 0.0, 4)
    trials.append(practical)
    return trials


def _expanded_trials(mod: Any) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for risk_degree in (0.85, 0.95):
        for n_drop in (2, 3):
            for hold_thresh in (4,):
                for vol_window in (10, 20, 30):
                    for vol_power in (0.5, 1.0, 1.5):
                        trials.append(
                            _trial(
                                name=(
                                    f"invvol_w{vol_window}_p{str(vol_power).replace('.', '')}"
                                    f"{_n_drop_tag(n_drop)}_hold{hold_thresh}_{_risk_tag(risk_degree)}"
                                ),
                                strategy_class="InverseVolPracticalTopkDropoutStrategy",
                                n_drop=n_drop,
                                risk_degree=risk_degree,
                                hold_thresh=hold_thresh,
                                extra={"universe": "csi300", "vol_window": vol_window, "vol_power": vol_power},
                            )
                        )
                for smooth_alpha in (0.55, 0.70, 0.85):
                    for hold_bonus in (0.001, 0.0025, 0.005):
                        trials.append(
                            _trial(
                                name=(
                                    f"turnover_a{str(smooth_alpha).replace('.', '')}"
                                    f"_hb{str(hold_bonus).replace('.', '')}"
                                    f"{_n_drop_tag(n_drop)}_hold{hold_thresh}_{_risk_tag(risk_degree)}"
                                ),
                                strategy_class="TurnoverAwarePracticalTopkDropoutStrategy",
                                n_drop=n_drop,
                                risk_degree=risk_degree,
                                hold_thresh=hold_thresh,
                                extra={"smooth_alpha": smooth_alpha, "hold_bonus": hold_bonus},
                            )
                        )
    for strength in (0.25, 0.50, 0.75):
        for allocation_blend in (0.0, 0.50):
            for use_rank_decay in (False, True):
                trials.append(
                    _trial(
                        name=(
                            f"multitopk_3-5-10-20_s{str(strength).replace('.', '')}"
                            f"_ab{str(allocation_blend).replace('.', '')}"
                            f"_{'decay' if use_rank_decay else 'flat'}_drop3_hold4_r095"
                        ),
                        strategy_class="MultiTopkConsensusPracticalTopkDropoutStrategy",
                        extra={
                            "topk_list": "3,5,10,20",
                            "consensus_strength": strength,
                            "allocation_blend": allocation_blend,
                            "use_rank_decay": use_rank_decay,
                        },
                    )
                )
    return trials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only Stage1 strategy-family selector.")
    parser.add_argument("--validation-summary-path", type=Path, default=DEFAULT_VALIDATION_SUMMARY)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--incumbent-strategy", default=INCUMBENT_STRATEGY)
    parser.add_argument("--signals", default="baseline_seq,growth_quality_linear_rank_w0001")
    parser.add_argument("--growth-weight", type=float, default=0.001)
    parser.add_argument("--profile", choices=("focused", "expanded"), default="focused")
    parser.add_argument("--max-strategies", type=int, default=0)
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

    strategy_trials = _focused_trials(mod) if args.profile == "focused" else _expanded_trials(mod)
    if args.max_strategies > 0:
        strategy_trials = strategy_trials[: args.max_strategies]

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
        "run_mode": "stage1_plus_strategy_family_validation_only",
        "selection_scope": "validation_only_no_test_loaded",
        "test_usage_policy": "test_not_loaded_not_used_for_selection",
        "hard_constraints": {"account": 150000, "topk": 5, "universe": "csi300"},
        "profile": args.profile,
        "incumbent_strategy": args.incumbent_strategy,
        "validation_summary_path": str(args.validation_summary_path),
        "validation_cache_paths": [str(path) for path in valid_cache_paths],
        "valid_common_rows": int(valid_rows),
        "signal_candidates": [item["candidate"] for item in signal_candidates],
        "strategy_count": len(strategy_trials),
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
