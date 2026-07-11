from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import h200_validation_stress_audit as stress_audit
import run_rank_ensemble_tra_alpha360_once as rank_runner
import run_stage2_signal_cv_search as cv
import run_stage2_signal_search_fast as fast
import tune_alpha360_tra_fundamental_xgb_weight_refine as stage2


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp/h200_validation_gate_aware_annual_scout_20260706_1"
DEFAULT_SENTINEL_TRIAL = "baseline_seq__prac_m000_hold5_r085"
DEFAULT_TRIAL_NAMES = [
    "baseline_seq__prac_m000_hold5_r085",
    "baseline_seq__prac_m000_hold7_r085",
    "cash_quality_z_tw001__prac_m000_hold7_r085",
    "value_quality_hibonus_hq08_b002__prac_m000_hold5_r085",
    "value_quality_hibonus_hq085_b002__prac_m000_hold5_r085",
]


def _jsonable(value: Any) -> Any:
    return fast._jsonable(value)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_combo(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _split_csv(value))


def _load_combo_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if args.combo_proposal_path:
        df = pd.read_csv(args.combo_proposal_path)
        if args.combo_column not in df.columns:
            raise RuntimeError(f"combo column not found: {args.combo_column}")
        for idx, row in df.head(int(args.max_combos)).reset_index(drop=True).iterrows():
            item = row.to_dict()
            item["combo"] = str(row[args.combo_column])
            item["combo_rank"] = int(idx + 1)
            specs.append(item)
    if args.combo:
        for combo in args.combo:
            specs.append({"combo": combo, "combo_rank": len(specs) + 1, "source": "explicit"})
    if not specs:
        raise RuntimeError("no combo proposals supplied")
    seen = set()
    out = []
    for spec in specs:
        combo = ",".join(str(seed) for seed in _parse_combo(str(spec["combo"])))
        if combo in seen:
            continue
        seen.add(combo)
        spec["combo"] = combo
        out.append(spec)
    return out


def _rank_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_cols = [
        "annual_gate_pass",
        "sentinel_gate_pass",
        "year_ann_min",
        "full_ann",
        "stable_score",
        "full_ir",
    ]
    cols = [col for col in sort_cols if col in df.columns]
    return df.sort_values(cols, ascending=[False] * len(cols), na_position="last")


def _trial_parts(trial_name: str) -> tuple[str, str]:
    if "__" not in trial_name:
        raise ValueError(f"trial_name must be signal__strategy: {trial_name}")
    return tuple(trial_name.split("__", 1))  # type: ignore[return-value]


def _annual_gate(row: dict[str, Any], *, full_min: float, year_min: float, year_std_max: float) -> bool:
    return (
        float(row.get("full_ann", float("-inf"))) >= full_min
        and float(row.get("year_ann_min", float("-inf"))) >= year_min
        and float(row.get("year_ann_std", float("inf"))) <= year_std_max
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only gate-aware annual scout for H200 stage2.")
    parser.add_argument("--valid-summary-path", action="append", type=Path, required=True)
    parser.add_argument("--combo-proposal-path", type=Path, default=None)
    parser.add_argument("--combo-column", default="combo")
    parser.add_argument("--combo", action="append", default=None)
    parser.add_argument("--max-combos", type=int, default=24)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--signal-profile", default="robust-small")
    parser.add_argument("--strategy-profiles", default="stable-core,stable-cash-quick,stable-quarter")
    parser.add_argument("--sentinel-trial", default=DEFAULT_SENTINEL_TRIAL)
    parser.add_argument("--trial-name", action="append", default=None)
    parser.add_argument("--sentinel-full-ann-min", type=float, default=0.20)
    parser.add_argument("--sentinel-year-ann-min", type=float, default=0.16)
    parser.add_argument("--annual-full-ann-min", type=float, default=0.24)
    parser.add_argument("--annual-year-ann-min", type=float, default=0.18)
    parser.add_argument("--annual-year-ann-std-max", type=float, default=0.08)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    grid_path = args.output_prefix.with_name(f"{args.output_prefix.name}_grid.csv")
    combo_path = args.output_prefix.with_name(f"{args.output_prefix.name}_combo_status.csv")
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.json")
    if not args.force_recompute and any(path.exists() for path in (grid_path, combo_path, summary_path)):
        raise FileExistsError("gate-aware scout outputs already exist; use --force-recompute")

    combo_specs = _load_combo_specs(args)
    trial_names = args.trial_name or DEFAULT_TRIAL_NAMES
    if args.sentinel_trial not in trial_names:
        trial_names = [args.sentinel_trial] + trial_names
    else:
        trial_names = [args.sentinel_trial] + [trial for trial in trial_names if trial != args.sentinel_trial]

    summary_base = {
        "run_mode": "h200_validation_gate_aware_annual_scout",
        "selection_scope": "validation_only_no_test_loaded",
        "test_usage_policy": "test_not_loaded_not_used_for_selection",
        "valid_summary_paths": [str(path) for path in args.valid_summary_path],
        "combo_proposal_path": str(args.combo_proposal_path) if args.combo_proposal_path else None,
        "combo_count": int(len(combo_specs)),
        "signal_profile": args.signal_profile,
        "strategy_profiles": _split_csv(args.strategy_profiles),
        "sentinel_trial": args.sentinel_trial,
        "trial_names": trial_names,
        "sentinel_gate": {
            "full_ann_min": args.sentinel_full_ann_min,
            "year_ann_min_min": args.sentinel_year_ann_min,
        },
        "annual_gate": {
            "full_ann_min": args.annual_full_ann_min,
            "year_ann_min_min": args.annual_year_ann_min,
            "year_ann_std_max": args.annual_year_ann_std_max,
        },
        "grid_path": str(grid_path),
        "combo_status_path": str(combo_path),
    }
    if args.dry_run:
        payload = dict(summary_base)
        payload["dry_run"] = True
        payload["combos"] = combo_specs
        summary_path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=True), encoding="utf-8")
        print(summary_path)
        return

    valid_seed_runs = stress_audit._load_valid_seed_runs(args.valid_summary_path)
    strategies = stress_audit._strategy_by_name(_split_csv(args.strategy_profiles))
    mod = rank_runner.load_module()
    tra_validation = {
        "window_key": "w1",
        "train": ["2012-01-01", "2019-12-31"],
        "valid": ["2020-01-01", "2022-12-31"],
        "test": ["2023-01-01", "2026-04-01"],
        "handler_kwargs_extra": stage2.TRA_RANK_ENSEMBLE_HANDLER_KWARGS,
    }

    rows: list[dict[str, Any]] = []
    combo_rows: list[dict[str, Any]] = []
    for spec in combo_specs:
        combo_text = str(spec["combo"])
        combo = _parse_combo(combo_text)
        missing = [seed for seed in combo if seed not in valid_seed_runs]
        if missing:
            raise RuntimeError(f"combo missing validation seed runs: combo={combo_text} missing={missing}")

        valid_pred, valid_label, valid_rows = stress_audit._rank_ensemble(mod, valid_seed_runs, combo)
        candidates = stress_audit._candidate_by_name(valid_pred.sort_index(), valid_label.sort_index(), args.signal_profile)
        sentinel_row = None
        combo_status = {
            "combo": combo_text,
            "combo_rank": spec.get("combo_rank"),
            "valid_common_rows": int(valid_rows),
            "sentinel_trial": args.sentinel_trial,
            "sentinel_gate_pass": False,
            "evaluated_trial_count": 0,
            "skipped_after_sentinel": False,
        }
        for trial_name in trial_names:
            signal_name, strategy_name = _trial_parts(trial_name)
            if signal_name not in candidates:
                print(f"skip missing signal in {args.signal_profile}: combo={combo_text} trial={trial_name}")
                continue
            if strategy_name not in strategies:
                print(f"skip missing strategy in {args.strategy_profiles}: combo={combo_text} trial={trial_name}")
                continue

            row = cv._evaluate_cv_candidate(tra_validation, candidates[signal_name], strategies[strategy_name])
            row.update(
                {
                    "selector_key": f"{combo_text}::{trial_name}",
                    "combo": combo_text,
                    "combo_size": len(combo),
                    "combo_rank": spec.get("combo_rank"),
                    "valid_common_rows": int(valid_rows),
                    "strategy_profile": strategies[strategy_name]["strategy_profile"],
                    "sentinel_trial": args.sentinel_trial,
                    "sentinel_gate_pass": None,
                    "annual_gate_pass": _annual_gate(
                        row,
                        full_min=args.annual_full_ann_min,
                        year_min=args.annual_year_ann_min,
                        year_std_max=args.annual_year_ann_std_max,
                    ),
                }
            )
            rows.append(row)
            combo_status["evaluated_trial_count"] = int(combo_status["evaluated_trial_count"]) + 1
            if trial_name == args.sentinel_trial:
                sentinel_pass = (
                    float(row["full_ann"]) >= args.sentinel_full_ann_min
                    and float(row["year_ann_min"]) >= args.sentinel_year_ann_min
                )
                row["sentinel_gate_pass"] = sentinel_pass
                sentinel_row = row
                combo_status.update(
                    {
                        "sentinel_full_ann": float(row["full_ann"]),
                        "sentinel_year_ann_min": float(row["year_ann_min"]),
                        "sentinel_year_ann_std": float(row["year_ann_std"]),
                        "sentinel_gate_pass": bool(sentinel_pass),
                    }
                )
                if not sentinel_pass:
                    combo_status["skipped_after_sentinel"] = True
                    print(
                        f"combo={combo_text} sentinel_fail full_ann={row['full_ann']:.6f} "
                        f"year_min={row['year_ann_min']:.6f}"
                    )
                    break
            else:
                row["sentinel_gate_pass"] = bool(combo_status["sentinel_gate_pass"])

            _rank_rows(pd.DataFrame(rows)).to_csv(grid_path, index=False)
            pd.DataFrame(combo_rows + [combo_status]).to_csv(combo_path, index=False)
            print(
                f"combo={combo_text} trial={trial_name} full_ann={row['full_ann']:.6f} "
                f"year_min={row['year_ann_min']:.6f} annual_gate={row['annual_gate_pass']}"
            )
        if sentinel_row is None:
            combo_status["skipped_after_sentinel"] = True
        combo_rows.append(combo_status)
        _rank_rows(pd.DataFrame(rows)).to_csv(grid_path, index=False)
        pd.DataFrame(combo_rows).to_csv(combo_path, index=False)

    ranked = _rank_rows(pd.DataFrame(rows))
    combo_df = pd.DataFrame(combo_rows)
    ranked.to_csv(grid_path, index=False)
    combo_df.to_csv(combo_path, index=False)
    summary = dict(summary_base)
    summary.update(
        {
            "dry_run": False,
            "row_count": int(len(ranked)),
            "combo_status_count": int(len(combo_df)),
            "sentinel_gate_pass_count": int(combo_df["sentinel_gate_pass"].sum()) if not combo_df.empty else 0,
            "annual_gate_pass_count": int(ranked["annual_gate_pass"].sum()) if not ranked.empty else 0,
            "top_rows": ranked.head(20).to_dict(orient="records") if not ranked.empty else [],
            "combo_status": combo_df.to_dict(orient="records"),
        }
    )
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    print(grid_path)
    print(combo_path)
    print(summary_path)


if __name__ == "__main__":
    main()
