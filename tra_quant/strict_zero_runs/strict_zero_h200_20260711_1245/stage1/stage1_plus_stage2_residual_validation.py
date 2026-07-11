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
import stage1_plus_conservative_fusion_validation as base


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp/stage1_plus_stage2_residual_validation_20260707_1"
SEED_SCREEN_SUMMARIES = (
    BASE_DIR / "tra_global_dual_seed_pattern_summary_stage2_seed_screen_20260622_1_b12000_valid.txt",
    BASE_DIR / "tra_global_dual_seed_pattern_summary_stage2_seed_screen_20260622_2_b12000_valid.txt",
)
STRICT_SUMMARY_GLOB = "tra_global_dual_seed_pattern_summary_stage2_strict_valid_only_new*_valid.txt"


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for record in records:
        key = str(record["cache_path"])
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def _records_from_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = base._parse_summary(path)
    runs = data.get("valid_base_runs")
    if not runs:
        return []
    if isinstance(runs, dict):
        iterator = runs.items()
    else:
        iterator = enumerate(runs)
    records = []
    for seed, row in iterator:
        cache_path = base._localize_path(row["cache_path"])
        if not cache_path.exists():
            continue
        records.append(
            {
                "seed": int(seed) if str(seed).isdigit() else seed,
                "cache_path": cache_path,
                "source_summary": path,
                "Rank_IC": row.get("Rank IC"),
                "IC": row.get("IC"),
                "valid_ann": row.get("with_cost_ann_return"),
                "valid_ir": row.get("with_cost_ir"),
                "valid_mdd": row.get("with_cost_mdd"),
            }
        )
    return records


def _rankic_sorted(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda row: float(row.get("Rank_IC") or -999.0), reverse=True)


def _build_overlay_groups(selected_groups: list[str]) -> dict[str, list[dict[str, Any]]]:
    seed_records = _dedupe_records(
        [record for path in SEED_SCREEN_SUMMARIES for record in _records_from_summary(path)]
    )
    strict_records = _dedupe_records(
        [
            record
            for path in sorted(BASE_DIR.glob(STRICT_SUMMARY_GLOB))
            for record in _records_from_summary(path)
        ]
    )
    seed_by_seed = {int(row["seed"]): row for row in seed_records if str(row["seed"]).isdigit()}
    groups = {
        "seed_screen_all8": seed_records,
        "seed_screen_rankic_top4": _rankic_sorted(seed_records)[:4],
        "seed_screen_99_11_1234": [
            seed_by_seed[seed] for seed in (99, 11, 1234) if seed in seed_by_seed
        ],
        "strict_all": strict_records,
        "strict_rankic_top16": _rankic_sorted(strict_records)[:16],
        "strict_rankic_top32": _rankic_sorted(strict_records)[:32],
        "hybrid_seed_all8_strict_top32": seed_records + _rankic_sorted(strict_records)[:32],
    }
    missing = [name for name in selected_groups if name not in groups]
    if missing:
        raise RuntimeError(f"unknown stage2 overlay group(s): {missing}")
    selected = {name: _dedupe_records(groups[name]) for name in selected_groups}
    empty = [name for name, rows in selected.items() if not rows]
    if empty:
        raise RuntimeError(f"empty stage2 overlay group(s): {empty}")
    return selected


def _candidate_name(group_name: str, mode: str, weight: float) -> str:
    return f"{group_name}_{mode}_w{str(weight).replace('.', '')}"


def _build_candidates(
    mod: Any,
    stage1_pred: pd.DataFrame,
    stage1_label: pd.DataFrame,
    overlay_groups: dict[str, list[dict[str, Any]]],
    weights: list[float],
    modes: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    common = stage1_pred.index.intersection(stage1_label.index).sort_values()
    base_rank = base._daily_rank_pct(stage1_pred.loc[common])
    label = stage1_label.loc[common].sort_index()
    candidates: list[dict[str, Any]] = [
        {
            "candidate": "baseline_seq",
            "kind": "baseline",
            "overlay": None,
            "weight": 0.0,
            "pred": base_rank,
            "label": label,
            "seed_count": 0,
        }
    ]
    metadata: dict[str, Any] = {}
    for group_name, records in overlay_groups.items():
        cache_paths = [record["cache_path"] for record in records]
        overlay_pred, _, overlay_rows = rank_runner.rank_ensemble_signal(mod, cache_paths)
        group_common = common.intersection(overlay_pred.index).sort_values()
        overlay_rank = base._daily_rank_pct(overlay_pred.loc[group_common])
        base_rank_common = base_rank.loc[group_common]
        label_common = label.loc[group_common].sort_index()
        metadata[group_name] = {
            "seed_count": len(records),
            "overlay_common_rows": int(overlay_rows),
            "candidate_common_rows": int(len(group_common)),
            "seeds": [record["seed"] for record in records],
            "rankic_mean": float(pd.Series([record.get("Rank_IC") for record in records], dtype="float64").mean()),
            "rankic_min": float(pd.Series([record.get("Rank_IC") for record in records], dtype="float64").min()),
            "rankic_max": float(pd.Series([record.get("Rank_IC") for record in records], dtype="float64").max()),
            "cache_paths": [str(path) for path in cache_paths],
            "source_summaries": sorted({str(record["source_summary"]) for record in records}),
        }
        for mode in modes:
            if mode not in {"linear", "gated"}:
                raise RuntimeError(f"unsupported residual mode: {mode}")
            for weight in weights:
                if mode == "linear":
                    pred = base_rank_common.mul(1.0 - weight).add(overlay_rank.mul(weight), fill_value=0.0)
                else:
                    confidence_gate = (base_rank_common - 0.5).abs().mul(2.0)
                    pred = base_rank_common.add(overlay_rank.sub(0.5).mul(confidence_gate).mul(weight), fill_value=0.0)
                    pred = base._daily_rank_pct(pred)
                candidates.append(
                    {
                        "candidate": _candidate_name(group_name, mode, weight),
                        "kind": f"stage2_rank_residual_{mode}",
                        "overlay": group_name,
                        "weight": float(weight),
                        "pred": pred.sort_index(),
                        "label": label_common,
                        "seed_count": len(records),
                    }
                )
    return candidates, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only Stage1 + Stage2 residual fusion selector.")
    parser.add_argument("--validation-summary-path", type=Path, default=base.DEFAULT_VALIDATION_SUMMARY)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--strategy-trial", action="append")
    parser.add_argument(
        "--group",
        action="append",
    )
    parser.add_argument("--weights", default="0.001,0.0025,0.005")
    parser.add_argument("--mode", action="append")
    parser.add_argument("--max-quarter-candidates", type=int, default=10)
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
    strategy_trials = args.strategy_trial or [base.INCUMBENT_STRATEGY]
    selected_groups = args.group or ["seed_screen_all8", "strict_rankic_top32", "hybrid_seed_all8_strict_top32"]
    selected_modes = args.mode or ["linear", "gated"]
    strategies = base._strategy_by_name(mod, strategy_trials)
    overlay_groups = _build_overlay_groups(selected_groups)
    valid_cache_paths = base._load_seed_cache_paths(args.validation_summary_path.expanduser().resolve(), "seed_runs_valid")
    stage1_pred, stage1_label, valid_rows = rank_runner.rank_ensemble_signal(mod, valid_cache_paths)
    candidates, overlay_metadata = _build_candidates(
        mod,
        stage1_pred.sort_index(),
        stage1_label.sort_index(),
        overlay_groups,
        weights,
        selected_modes,
    )

    annual_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    baseline_by_strategy: dict[str, dict[str, Any]] = {}
    for strategy_name, strategy in strategies.items():
        baseline_candidate = next(item for item in candidates if item["candidate"] == "baseline_seq")
        baseline_segments = [
            base._evaluate_segment(mod, strategy, baseline_candidate["pred"], baseline_candidate["label"], name, date_range)
            for name, date_range in base.ANNUAL_SEGMENTS.items()
        ]
        segment_rows.extend(
            {**row, "candidate": "baseline_seq", "strategy_trial": strategy_name, "segment_group": "annual"}
            for row in baseline_segments
        )
        baseline_summary = base._summarize_candidate(baseline_candidate, strategy_name, baseline_segments, None, args)
        baseline_by_strategy[strategy_name] = baseline_summary
        annual_rows.append(baseline_summary)

        for candidate in candidates:
            if candidate["candidate"] == "baseline_seq":
                continue
            rows = [
                base._evaluate_segment(mod, strategy, candidate["pred"], candidate["label"], name, date_range)
                for name, date_range in base.ANNUAL_SEGMENTS.items()
            ]
            segment_rows.extend(
                {**row, "candidate": candidate["candidate"], "strategy_trial": strategy_name, "segment_group": "annual"}
                for row in rows
            )
            row = base._summarize_candidate(candidate, strategy_name, rows, baseline_summary, args)
            row["seed_count"] = candidate["seed_count"]
            annual_rows.append(row)
            base._write_csv(annual_rows, annual_path)
            print(
                f"annual candidate={candidate['candidate']} strategy={strategy_name} "
                f"delta_full={row['delta_full_ann']:.6f} "
                f"delta_year_lcb={row['delta_year_ann_lcb']:.6f} "
                f"prepass={row['annual_prepass']}"
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
            base._evaluate_segment(mod, strategy, candidate["pred"], candidate["label"], name, date_range)
            for name, date_range in base.QUARTER_SEGMENTS.items()
        ]
        segment_rows.extend(
            {**row, "candidate": candidate_name, "strategy_trial": strategy_name, "segment_group": "quarter"}
            for row in quarter_detail
        )
        baseline_row = None if candidate_name == "baseline_seq" else baseline_by_strategy[strategy_name]
        row = base._summarize_candidate(candidate, strategy_name, annual_detail + quarter_detail, baseline_row, args)
        row["seed_count"] = candidate["seed_count"]
        stress_rows.append(row)
        base._write_csv(stress_rows, stress_path)
        print(
            f"stress candidate={candidate_name} strategy={strategy_name} "
            f"pass={row['stable_over_stage1_pass']} reason={row['stable_gate_reason']}"
        )

    detail_path.write_text(pd.DataFrame(segment_rows).to_csv(index=False), encoding="utf-8")
    base._write_csv(annual_rows, annual_path)
    base._write_csv(stress_rows, stress_path)
    stress_df = pd.DataFrame(stress_rows).sort_values(
        ["stable_over_stage1_pass", "annual_prepass", "selection_score", "delta_full_ann"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    pass_df = stress_df[stress_df["stable_over_stage1_pass"] == True] if not stress_df.empty else stress_df
    locked_candidate = pass_df.iloc[0].to_dict() if not pass_df.empty else None
    summary = {
        "run_mode": "stage1_plus_stage2_residual_validation_only",
        "selection_scope": "validation_only_no_test_loaded",
        "test_usage_policy": "test_not_loaded_not_used_for_selection",
        "hard_constraints": {"account": 150000, "topk": 5, "universe": "csi300"},
        "strategy_trials": list(strategies),
        "validation_summary_path": str(args.validation_summary_path),
        "stage1_validation_cache_paths": [str(path) for path in valid_cache_paths],
        "valid_common_rows": int(valid_rows),
        "overlay_groups": overlay_metadata,
        "weights": weights,
        "modes": selected_modes,
        "candidate_count": len(candidates),
        "annual_segments": base.ANNUAL_SEGMENTS,
        "quarter_segments": base.QUARTER_SEGMENTS,
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
    summary_path.write_text(json.dumps(base._jsonable(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    print(summary_path)
    if locked_candidate is None:
        print("no validation candidate passed stable-over-stage1 gate")
    else:
        print(f"locked_candidate_validation_only={locked_candidate['candidate']} strategy={locked_candidate['strategy_trial']}")


if __name__ == "__main__":
    main()
