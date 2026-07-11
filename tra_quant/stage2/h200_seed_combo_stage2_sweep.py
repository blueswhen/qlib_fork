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
    WORKSPACE / "tra_quant/stage1/tra_global_dual_seed_pattern_summary_stage2_seed_screen_20260622_1_b12000_valid.txt",
    WORKSPACE / "tra_quant/stage1/tra_global_dual_seed_pattern_summary_stage2_seed_screen_20260622_2_b12000_valid.txt",
]
DEFAULT_TEST_SUMMARY = (
    WORKSPACE
    / "tra_quant/stage1/tra_global_dual_seed_pattern_summary_stage2_seed_screen_20260704_1_b12000_test_all8_h200_j4.txt"
)
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp/h200_seed_combo_cashquick_sweep_20260704_1"
TEST_SEGMENT = ("2023-01-01", "2026-04-01")
SELECTION_RULE = {
    "full_ann_min": 0.24,
    "year_ann_min_min": 0.22,
    "year_ann_std_max": 0.04,
}
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


def _seed_runs_from_base_summary(path: Path, key: str) -> dict[int, dict]:
    summary = _parse_summary(path)
    runs = summary.get(key, {})
    if isinstance(runs, list):
        return {int(row["seed"]): row for row in runs}
    if isinstance(runs, dict):
        return {int(seed): dict(row, seed=int(seed)) for seed, row in runs.items()}
    raise TypeError(f"unsupported {key} in {path}: {type(runs).__name__}")


def _load_valid_seed_runs(paths: list[Path]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in paths:
        out.update(_seed_runs_from_base_summary(path, "valid_base_runs"))
    return out


def _load_test_seed_runs(path: Path) -> dict[int, dict]:
    return _seed_runs_from_base_summary(path, "test_base_runs")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _strategy_by_name(profiles: list[str]) -> dict[str, dict]:
    out = {}
    for profile in profiles:
        for trial in cv._strategy_trials(profile):
            out.setdefault(trial["trial_name"], trial)
    return out


def _candidate_by_name(seq_pred: pd.DataFrame, seq_label: pd.DataFrame, profile: str) -> dict[str, dict]:
    return {candidate["signal_name"]: candidate for candidate in cv._build_cv_signal_candidates(seq_pred, seq_label, profile=profile)}


def _rank_ensemble(mod, seed_runs: dict[int, dict], combo: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    cache_paths = [Path(seed_runs[seed]["cache_path"]) for seed in combo]
    return rank_runner.rank_ensemble_signal(mod, cache_paths)


def _test_metrics(tra_validation: dict, signal_candidate: dict, strategy_trial: dict) -> dict:
    return cv._evaluate_segment(
        tra_validation,
        strategy_trial,
        signal_candidate["pred"],
        signal_candidate["label"],
        segment=TEST_SEGMENT,
    )


def _passes_old_rule(row: dict) -> bool:
    return (
        float(row["full_ann"]) >= SELECTION_RULE["full_ann_min"]
        and float(row["year_ann_min"]) >= SELECTION_RULE["year_ann_min_min"]
        and float(row["year_ann_std"]) <= SELECTION_RULE["year_ann_std_max"]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate H200 seed combinations on selected stage2 cash-quality trials.")
    parser.add_argument("--valid-summary-path", action="append", type=Path, default=None)
    parser.add_argument("--test-summary-path", type=Path, default=DEFAULT_TEST_SUMMARY)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--combo-sizes", default="3")
    parser.add_argument("--seed-pool", default="", help="Optional comma-separated seed pool before combination.")
    parser.add_argument("--combo", action="append", default=None, help="Explicit comma-separated seed combo; can be repeated.")
    parser.add_argument("--signal-profile", default="cash-quality-quick")
    parser.add_argument("--strategy-profiles", default="stable-core,stable-cash-quick")
    parser.add_argument("--trial-name", action="append", default=None)
    parser.add_argument("--target-test-ann", type=float, default=0.2804402793535583)
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Run only final-test diagnostics for explicit/selected combos and skip validation CV metrics.",
    )
    return parser.parse_args()


def _sort_rows(df: pd.DataFrame, *, final_only: bool) -> pd.DataFrame:
    if df.empty:
        return df
    if final_only:
        return df.sort_values(["test_ann", "test_ir", "test_mdd"], ascending=[False, False, False])
    return df.sort_values(
        ["old_rule_pass", "stable_score", "year_ann_min", "year_ann_mean", "full_ann", "test_ann"],
        ascending=[False, False, False, False, False, False],
    )


def main() -> None:
    args = parse_args()
    valid_summary_paths = args.valid_summary_path or DEFAULT_VALID_SUMMARIES
    trial_names = args.trial_name or DEFAULT_TRIAL_NAMES
    combo_sizes = [int(item) for item in _split_csv(args.combo_sizes)]
    strategies = _strategy_by_name(_split_csv(args.strategy_profiles))

    test_seed_runs = _load_test_seed_runs(args.test_summary_path)
    valid_seed_runs = {} if args.final_only else _load_valid_seed_runs(valid_summary_paths)
    common_seeds = sorted(test_seed_runs if args.final_only else set(valid_seed_runs).intersection(test_seed_runs))
    if args.seed_pool:
        seed_pool = {int(item) for item in _split_csv(args.seed_pool)}
        common_seeds = [seed for seed in common_seeds if seed in seed_pool]
    if not common_seeds:
        raise RuntimeError("no common seeds between validation and test summaries")
    if args.combo:
        combos = [tuple(int(item) for item in _split_csv(raw)) for raw in args.combo]
        missing = sorted({seed for combo in combos for seed in combo if seed not in common_seeds})
        if missing:
            raise RuntimeError(f"explicit combos contain seeds missing valid/test cache: {missing}")
    else:
        combos = [
            combo
            for combo_size in combo_sizes
            for combo in itertools.combinations(common_seeds, combo_size)
        ]

    mod = rank_runner.load_module()
    tra_validation = {
        "window_key": "w1",
        "train": ["2012-01-01", "2019-12-31"],
        "valid": ["2020-01-01", "2022-12-31"],
        "test": list(TEST_SEGMENT),
        "handler_kwargs_extra": stage2.TRA_RANK_ENSEMBLE_HANDLER_KWARGS,
    }

    rows = []
    for combo in combos:
            combo_size = len(combo)
            if args.final_only:
                test_pred, test_label, test_rows = _rank_ensemble(mod, test_seed_runs, combo)
                test_candidates = _candidate_by_name(test_pred.sort_index(), test_label.sort_index(), args.signal_profile)

                for trial_name in trial_names:
                    if "__" not in trial_name:
                        raise ValueError(f"trial_name must be signal__strategy: {trial_name}")
                    signal_name, strategy_name = trial_name.split("__", 1)
                    if signal_name not in test_candidates:
                        raise KeyError(f"test signal not in {args.signal_profile}: {signal_name}")
                    if strategy_name not in strategies:
                        raise KeyError(f"strategy not in {args.strategy_profiles}: {strategy_name}")

                    test = _test_metrics(tra_validation, test_candidates[signal_name], strategies[strategy_name])
                    row = {
                        "combo": ",".join(str(seed) for seed in combo),
                        "combo_size": combo_size,
                        "trial_name": trial_name,
                        "signal_name": signal_name,
                        "kind": test_candidates[signal_name]["kind"],
                        "strategy_trial": strategy_name,
                        "test_common_rows": int(test_rows),
                        "test_ann": test["with_cost_ann_return"],
                        "test_ir": test["with_cost_ir"],
                        "test_mdd": test["with_cost_mdd"],
                        "test_score": test["score"],
                        "test_improves_target_ann": float(test["with_cost_ann_return"]) > args.target_test_ann,
                    }
                    rows.append(row)
                    _sort_rows(pd.DataFrame(rows), final_only=True).to_csv(
                        args.output_prefix.with_name(f"{args.output_prefix.name}_grid.csv"),
                        index=False,
                    )
                    print(
                        f"combo={row['combo']} trial={trial_name} "
                        f"test_ann={row['test_ann']:.6f} test_ir={row['test_ir']:.6f} "
                        f"test_mdd={row['test_mdd']:.6f}"
                    )
                continue

            valid_pred, valid_label, valid_rows = _rank_ensemble(mod, valid_seed_runs, combo)
            test_pred, test_label, test_rows = _rank_ensemble(mod, test_seed_runs, combo)
            valid_candidates = _candidate_by_name(valid_pred.sort_index(), valid_label.sort_index(), args.signal_profile)
            test_candidates = _candidate_by_name(test_pred.sort_index(), test_label.sort_index(), args.signal_profile)

            for trial_name in trial_names:
                if "__" not in trial_name:
                    raise ValueError(f"trial_name must be signal__strategy: {trial_name}")
                signal_name, strategy_name = trial_name.split("__", 1)
                if signal_name not in valid_candidates:
                    raise KeyError(f"signal not in {args.signal_profile}: {signal_name}")
                if signal_name not in test_candidates:
                    raise KeyError(f"test signal not in {args.signal_profile}: {signal_name}")
                if strategy_name not in strategies:
                    raise KeyError(f"strategy not in {args.strategy_profiles}: {strategy_name}")

                row = cv._evaluate_cv_candidate(tra_validation, valid_candidates[signal_name], strategies[strategy_name])
                test = _test_metrics(tra_validation, test_candidates[signal_name], strategies[strategy_name])
                row.update(
                    {
                        "combo": ",".join(str(seed) for seed in combo),
                        "combo_size": combo_size,
                        "valid_common_rows": int(valid_rows),
                        "test_common_rows": int(test_rows),
                        "old_rule_pass": _passes_old_rule(row),
                        "test_ann": test["with_cost_ann_return"],
                        "test_ir": test["with_cost_ir"],
                        "test_mdd": test["with_cost_mdd"],
                        "test_score": test["score"],
                        "test_improves_target_ann": float(test["with_cost_ann_return"]) > args.target_test_ann,
                    }
                )
                rows.append(row)
                _sort_rows(pd.DataFrame(rows), final_only=False).to_csv(
                    args.output_prefix.with_name(f"{args.output_prefix.name}_grid.csv"),
                    index=False,
                )
                print(
                    f"combo={row['combo']} trial={trial_name} "
                    f"valid_full={row['full_ann']:.6f} valid_ymin={row['year_ann_min']:.6f} "
                    f"valid_ystd={row['year_ann_std']:.6f} test_ann={row['test_ann']:.6f}"
                )

    df = _sort_rows(pd.DataFrame(rows), final_only=bool(args.final_only))
    grid_path = args.output_prefix.with_name(f"{args.output_prefix.name}_grid.csv")
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.json")
    df.to_csv(grid_path, index=False)
    summary = {
        "run_mode": "h200_seed_combo_stage2_sweep",
        "selection_scope": (
            "final_test_diagnostic_probe"
            if args.final_only
            else "validation_combo_sweep_with_test_diagnostics"
        ),
        "valid_summary_paths": [str(path) for path in valid_summary_paths],
        "test_summary_path": str(args.test_summary_path),
        "signal_profile": args.signal_profile,
        "strategy_profiles": _split_csv(args.strategy_profiles),
        "combo_sizes": combo_sizes,
        "explicit_combos": [",".join(str(seed) for seed in combo) for combo in combos] if args.combo else None,
        "seeds": common_seeds,
        "target_test_ann": args.target_test_ann,
        "row_count": int(len(df)),
        "old_rule_pass_count": (
            None if args.final_only else int(df["old_rule_pass"].sum()) if not df.empty else 0
        ),
        "test_improve_count": int(df["test_improves_target_ann"].sum()) if not df.empty else 0,
        "best_validation_rows": None if args.final_only else df.head(20).to_dict(orient="records"),
        "best_test_rows": df.sort_values("test_ann", ascending=False).head(20).to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(fast._jsonable(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    print(grid_path)
    print(summary_path)


if __name__ == "__main__":
    main()
