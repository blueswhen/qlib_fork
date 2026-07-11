from __future__ import annotations

import argparse
import ast
import itertools
import json
from pathlib import Path

import pandas as pd

import run_rank_ensemble_tra_alpha360_once as rank_runner
import run_stage2_signal_cv_search as cv
import run_stage2_signal_search_fast as fast
import tune_alpha360_tra_fundamental_xgb_weight_refine as stage2
from tra_local_helpers import localize_value


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = BASE_DIR.parent.parent
DEFAULT_VALID_SUMMARIES = [
    WORKSPACE
    / "tra_quant/runs/stage1_h200_20260621_11_4seed_b12000_fast/stage1/tra_global_dual_seed_pattern_summary_stage1_h200_20260621_11_4seed_b12000_fast_b12000_valid.txt",
    WORKSPACE / "tra_quant/stage1/tra_global_dual_seed_pattern_summary_stage2_seed_screen_20260622_1_b12000_valid.txt",
    WORKSPACE / "tra_quant/stage1/tra_global_dual_seed_pattern_summary_stage2_seed_screen_20260622_2_b12000_valid.txt",
]
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp/h200_strict_validation_selector_20260705_1"
DEFAULT_TRIAL_NAMES = [
    "baseline_seq__prac_m000_hold5_r08",
    "baseline_seq__prac_m000_hold5_r085",
    "baseline_seq__prac_m000_hold5_r09",
    "baseline_seq__prac_m000_hold7_r08",
    "baseline_seq__prac_m000_hold7_r085",
    "baseline_seq__prac_m000_hold7_r09",
    "baseline_seq__prac_m000_hold7_r095",
    "cash_quality_z_tw0005__prac_m000_hold5_r095",
    "cash_quality_z_tw0005__prac_m000_hold7_r095",
    "cash_quality_z_tw001__prac_m000_hold5_r09",
    "cash_quality_z_tw001__prac_m000_hold7_r085",
    "cash_quality_z_tw001__prac_m000_hold7_r09",
]
STRICT_SELECTION_RULE = {
    "full_ann_min": 0.15,
    "year_ann_min_min": 0.0,
    "year_ann_std_max": 0.12,
    "full_mdd_min": -0.30,
    "sort": [
        "strict_pass",
        "year_ann_min",
        "stable_score",
        "full_ann",
        "full_ir",
    ],
}


def _parse_summary(path: Path) -> dict:
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            data[key] = localize_value(ast.literal_eval(value))
        except Exception:
            data[key] = localize_value(value)
    return data


def _seed_runs_from_base_summary(path: Path) -> dict[int, dict]:
    summary = _parse_summary(path)
    runs = summary.get("valid_base_runs", {})
    if isinstance(runs, list):
        return {int(row["seed"]): row for row in runs}
    if isinstance(runs, dict):
        return {int(seed): dict(row, seed=int(seed)) for seed, row in runs.items()}
    raise TypeError(f"unsupported valid_base_runs in {path}: {type(runs).__name__}")


def _load_valid_seed_runs(paths: list[Path]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in paths:
        out.update(_seed_runs_from_base_summary(path))
    return out


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _strategy_by_name(profiles: list[str]) -> dict[str, dict]:
    out = {}
    for profile in profiles:
        for trial in cv._strategy_trials(profile):
            trial_with_profile = dict(trial)
            trial_with_profile.setdefault("strategy_profile", profile)
            out.setdefault(trial["trial_name"], trial_with_profile)
    return out


def _candidate_by_name(seq_pred: pd.DataFrame, seq_label: pd.DataFrame, profile: str) -> dict[str, dict]:
    return {candidate["signal_name"]: candidate for candidate in cv._build_cv_signal_candidates(seq_pred, seq_label, profile=profile)}


def _rank_ensemble(mod, seed_runs: dict[int, dict], combo: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    cache_paths = [Path(seed_runs[seed]["cache_path"]) for seed in combo]
    return rank_runner.rank_ensemble_signal(mod, cache_paths)


def _strict_pass(row: dict) -> bool:
    return (
        float(row["full_ann"]) >= STRICT_SELECTION_RULE["full_ann_min"]
        and float(row["year_ann_min"]) >= STRICT_SELECTION_RULE["year_ann_min_min"]
        and float(row["year_ann_std"]) <= STRICT_SELECTION_RULE["year_ann_std_max"]
        and float(row["full_mdd"]) >= STRICT_SELECTION_RULE["full_mdd_min"]
    )


def _rank_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(
        STRICT_SELECTION_RULE["sort"],
        ascending=[False, False, False, False, False],
        na_position="last",
    )


def _write_validation_summary(seed_runs: dict[int, dict], combo: tuple[int, ...], path: Path) -> None:
    rows = []
    for seed in combo:
        row = dict(seed_runs[seed])
        row["seed"] = int(seed)
        rows.append(row)
    lines = [
        "window_key=w1",
        "train=['2012-01-01', '2019-12-31']",
        "valid=['2020-01-01', '2022-12-31']",
        "test=['2023-01-01', '2026-04-01']",
        f"seed_runs_valid={rows!r}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float_or_neg_inf(value) -> float:
    if value is None:
        return float("-inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict H200 validation-only selector. Does not read test summaries.")
    parser.add_argument("--valid-summary-path", action="append", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--combo-sizes", default="3,4")
    parser.add_argument(
        "--combo",
        action="append",
        default=None,
        help="Explicit comma-separated seed combo to evaluate. May be repeated. Overrides --combo-sizes generation.",
    )
    parser.add_argument("--seed-pool", default="")
    parser.add_argument("--top-valid-seeds", type=int, default=8)
    parser.add_argument("--signal-profile", default="cash-quality-quick")
    parser.add_argument("--strategy-profiles", default="stable-core,stable-cash-quick")
    parser.add_argument("--trial-name", action="append", default=None)
    parser.add_argument(
        "--all-profile-trials",
        action="store_true",
        help="Evaluate every signal candidate from --signal-profile against every strategy from --strategy-profiles.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    grid_path = args.output_prefix.with_name(f"{args.output_prefix.name}_grid.csv")
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.json")
    locked_path = args.output_prefix.with_name(f"{args.output_prefix.name}_locked_manifest.json")
    validation_wrapper_path = args.output_prefix.with_name(f"{args.output_prefix.name}_locked_validation_summary.txt")
    selected_row_path = args.output_prefix.with_name(f"{args.output_prefix.name}_locked_cv_row.csv")
    valid_summary_paths = args.valid_summary_path or DEFAULT_VALID_SUMMARIES
    trial_names = args.trial_name or DEFAULT_TRIAL_NAMES
    strategies = _strategy_by_name(_split_csv(args.strategy_profiles))
    combo_sizes = [int(item) for item in _split_csv(args.combo_sizes)]

    valid_seed_runs = _load_valid_seed_runs(valid_summary_paths)
    if args.seed_pool:
        seed_pool = [int(item) for item in _split_csv(args.seed_pool)]
    else:
        ranked = sorted(
            valid_seed_runs,
            key=lambda seed: (
                _float_or_neg_inf(valid_seed_runs[seed].get("with_cost_ann_return", float("-inf"))),
                _float_or_neg_inf(valid_seed_runs[seed].get("with_cost_ir", float("-inf"))),
            ),
            reverse=True,
        )
        seed_pool = ranked[: int(args.top_valid_seeds)]
    missing = [seed for seed in seed_pool if seed not in valid_seed_runs]
    if missing:
        raise RuntimeError(f"seed pool missing validation cache: {missing}")

    if args.combo:
        combos = [tuple(int(item) for item in _split_csv(combo_text)) for combo_text in args.combo]
        seed_pool = sorted({seed for combo in combos for seed in combo})
    else:
        combos = [
            combo
            for combo_size in combo_sizes
            for combo in itertools.combinations(sorted(seed_pool), combo_size)
        ]
    combos = [combo for idx, combo in enumerate(combos) if idx % args.num_shards == args.shard_index]
    if not combos:
        raise RuntimeError("no combos selected for this shard")

    row_map: dict[str, dict] = {}
    if grid_path.exists() and not args.force_recompute:
        df = pd.read_csv(grid_path)
        row_map = {str(row["selector_key"]): row for row in df.to_dict(orient="records")}

    mod = rank_runner.load_module()
    tra_validation = {
        "window_key": "w1",
        "train": ["2012-01-01", "2019-12-31"],
        "valid": ["2020-01-01", "2022-12-31"],
        "test": ["2023-01-01", "2026-04-01"],
        "handler_kwargs_extra": stage2.TRA_RANK_ENSEMBLE_HANDLER_KWARGS,
    }

    for combo in combos:
        combo_text = ",".join(str(seed) for seed in combo)
        valid_pred, valid_label, valid_rows = _rank_ensemble(mod, valid_seed_runs, combo)
        valid_candidates = _candidate_by_name(valid_pred.sort_index(), valid_label.sort_index(), args.signal_profile)
        active_trial_names = (
            [
                f"{signal_name}__{strategy_name}"
                for signal_name in valid_candidates
                for strategy_name in strategies
            ]
            if args.all_profile_trials
            else trial_names
        )

        for trial_name in active_trial_names:
            if "__" not in trial_name:
                raise ValueError(f"trial_name must be signal__strategy: {trial_name}")
            signal_name, strategy_name = trial_name.split("__", 1)
            selector_key = f"{combo_text}::{trial_name}"
            if selector_key in row_map:
                continue
            if signal_name not in valid_candidates:
                raise KeyError(f"signal not in {args.signal_profile}: {signal_name}")
            if strategy_name not in strategies:
                raise KeyError(f"strategy not in {args.strategy_profiles}: {strategy_name}")

            row = cv._evaluate_cv_candidate(tra_validation, valid_candidates[signal_name], strategies[strategy_name])
            row.update(
                {
                    "selector_key": selector_key,
                    "combo": combo_text,
                    "combo_size": len(combo),
                    "valid_common_rows": int(valid_rows),
                    "strict_pass": _strict_pass(row),
                    "strategy_profile": strategies[strategy_name]["strategy_profile"],
                }
            )
            row_map[selector_key] = row
            _rank_rows(pd.DataFrame(row_map.values())).to_csv(grid_path, index=False)
            print(
                f"combo={combo_text} trial={trial_name} "
                f"full_ann={row['full_ann']:.6f} year_min={row['year_ann_min']:.6f} "
                f"year_std={row['year_ann_std']:.6f} strict_pass={row['strict_pass']}"
            )

    df = _rank_rows(pd.DataFrame(row_map.values()))
    df.to_csv(grid_path, index=False)
    selected = df.iloc[0].to_dict()
    pd.DataFrame([selected]).to_csv(selected_row_path, index=False)
    selected_combo = tuple(int(seed) for seed in str(selected["combo"]).split(","))
    _write_validation_summary(valid_seed_runs, selected_combo, validation_wrapper_path)
    summary = {
        "run_mode": "h200_strict_validation_only_selector",
        "selection_scope": "validation_only_no_test_summary_loaded",
        "test_usage_policy": "no_test_access_before_locked_manifest",
        "valid_summary_paths": [str(path) for path in valid_summary_paths],
        "signal_profile": args.signal_profile,
        "strategy_profiles": _split_csv(args.strategy_profiles),
        "all_profile_trials": bool(args.all_profile_trials),
        "trial_names": "all_profile_trials" if args.all_profile_trials else trial_names,
        "strict_selection_rule": STRICT_SELECTION_RULE,
        "seed_pool": seed_pool,
        "combo_sizes": combo_sizes,
        "explicit_combos": args.combo or None,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "row_count": int(len(df)),
        "strict_pass_count": int(df["strict_pass"].sum()) if not df.empty else 0,
        "selected_row": selected,
        "locked_manifest_path": str(locked_path),
        "locked_validation_summary_path": str(validation_wrapper_path),
        "locked_cv_row_path": str(selected_row_path),
    }
    locked = {
        "locked_at": pd.Timestamp.utcnow().isoformat(),
        "run_mode": "strict_validation_locked_candidate",
        "selection_scope": "validation_only_no_test_summary_loaded",
        "test_usage_policy": "test_may_be_run_once_after_this_manifest",
        "selected_row": selected,
        "cv_grid_path": str(grid_path),
        "locked_cv_row_path": str(selected_row_path),
        "validation_summary_path": str(validation_wrapper_path),
        "final_test_summary_path": None,
        "signal_profile": args.signal_profile,
        "strategy_profile": selected.get("strategy_profile", "stable-core"),
        "target_test_ann": 0.2804402793535583,
    }
    summary_path.write_text(json.dumps(fast._jsonable(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    locked_path.write_text(json.dumps(fast._jsonable(locked), indent=2, ensure_ascii=True), encoding="utf-8")
    print(grid_path)
    print(summary_path)
    print(locked_path)
    print(validation_wrapper_path)
    print(selected_row_path)
    print(f"locked_combo={selected['combo']} locked_trial={selected['trial_name']}")


if __name__ == "__main__":
    main()
