from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

import run_stage2_signal_cv_search as cv
import run_stage2_signal_search_fast as fast
import tune_alpha360_tra_fundamental_xgb_weight_refine as stage2


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CV_GRID = BASE_DIR / "tmp" / "stage2_cv_robust_small_stable_core_20260613_1_cv_grid.csv"
DEFAULT_RESULT_SUFFIX = "stage2_cv_cash_quality_z001_hold7r085_final_20260613_1"
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp" / DEFAULT_RESULT_SUFFIX
TARGET_TEST_ANN = 0.22004029352797247
CURRENT_BEST_TEST_ANN = 0.2643467300531994
SELECTION_RULE = {
    "full_ann_min": 0.24,
    "year_ann_min_min": 0.22,
    "year_ann_std_max": 0.04,
    "sort": ["stable_score", "year_ann_min", "year_ann_mean", "full_ann"],
}


def _parse_summary(path: Path) -> dict:
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    for key in ("train", "valid", "test", "seed_runs_test", "seed_runs_valid"):
        if key in data:
            try:
                data[key] = ast.literal_eval(data[key])
            except (SyntaxError, ValueError):
                pass
    return data


def _load_rank_ensemble_test() -> tuple[dict, pd.DataFrame, pd.DataFrame, int, list[Path]]:
    runner = stage2._load_rank_ensemble_runner()
    tra_mod = runner.load_module()
    validation_summary = _parse_summary(stage2.TRA_BEST_VALIDATION_SUMMARY_PATH)
    final_summary = _parse_summary(stage2.TRA_BEST_FINAL_TEST_SUMMARY_PATH)
    test_cache_paths = runner._load_seed_cache_paths(stage2.TRA_BEST_FINAL_TEST_SUMMARY_PATH, "seed_runs_test")
    test_pred, test_label, test_common_rows = runner.rank_ensemble_signal(tra_mod, test_cache_paths)
    tra_validation = {
        "window_key": validation_summary["window_key"],
        "train": validation_summary["train"],
        "valid": validation_summary["valid"],
        "test": validation_summary["test"],
        "handler_kwargs_extra": stage2.TRA_RANK_ENSEMBLE_HANDLER_KWARGS,
        "stage1_source": "rank_ensemble",
        "stage1_validation_summary_path": str(stage2.TRA_BEST_VALIDATION_SUMMARY_PATH),
        "stage1_final_test_summary_path": str(stage2.TRA_BEST_FINAL_TEST_SUMMARY_PATH),
        "stage1_final_test_seed_cache_paths": [str(path) for path in test_cache_paths],
        "stage1_test_common_rows": int(test_common_rows),
        "stage1_final_summary": final_summary,
    }
    return tra_validation, test_pred.sort_index(), test_label.sort_index(), int(test_common_rows), test_cache_paths


def _select_cv_row(cv_grid_path: Path, trial_name: str | None) -> dict:
    df = pd.read_csv(cv_grid_path)
    if trial_name:
        rows = df[df["trial_name"] == trial_name]
        if rows.empty:
            raise ValueError(f"trial_name not found in CV grid: {trial_name}")
        return rows.iloc[0].to_dict()

    selected = df[
        (df["status"] == "success")
        & (df["full_ann"] >= SELECTION_RULE["full_ann_min"])
        & (df["year_ann_min"] >= SELECTION_RULE["year_ann_min_min"])
        & (df["year_ann_std"] <= SELECTION_RULE["year_ann_std_max"])
    ].copy()
    if selected.empty:
        raise RuntimeError(f"no CV rows pass selection rule: {SELECTION_RULE}")
    selected = selected.sort_values(SELECTION_RULE["sort"], ascending=False)
    return selected.iloc[0].to_dict()


def _strategy_from_name(strategy_trial_name: str) -> dict:
    for strategy_trial in cv._strategy_trials("stable-core"):
        if strategy_trial["trial_name"] == strategy_trial_name:
            return strategy_trial
    raise ValueError(f"strategy not found in stable-core profile: {strategy_trial_name}")


def _signal_from_name(
    signal_name: str,
    seq_pred: pd.DataFrame,
    seq_label: pd.DataFrame,
    *,
    signal_profile: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = cv._build_cv_signal_candidates(seq_pred, seq_label, profile=signal_profile)
    for candidate in candidates:
        if candidate["signal_name"] == signal_name:
            return candidate["pred"].sort_index(), candidate["label"].sort_index()
    raise ValueError(f"signal not found in robust-small profile: {signal_name}")


def _write_text_summary(summary: dict, path: Path) -> None:
    lines = []
    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(fast._jsonable(value), ensure_ascii=True, sort_keys=True)
        lines.append(f"{key}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one CV-selected stage2 candidate on final test.")
    parser.add_argument("--cv-grid-path", type=Path, default=DEFAULT_CV_GRID)
    parser.add_argument("--trial-name", default=None)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--result-suffix", default=DEFAULT_RESULT_SUFFIX)
    parser.add_argument(
        "--signal-profile",
        choices=["robust-tiny", "robust-small", "announcement-small", "cash-announcement-small"],
        default="robust-small",
    )
    parser.add_argument("--target-test-ann", type=float, default=CURRENT_BEST_TEST_ANN)
    parser.add_argument("--fast-mode", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    selected_row = _select_cv_row(args.cv_grid_path, args.trial_name)
    strategy_trial = _strategy_from_name(str(selected_row["strategy_trial"]))
    tra_validation, test_seq_pred, test_seq_label, test_common_rows, test_cache_paths = _load_rank_ensemble_test()
    test_pred, test_label = _signal_from_name(
        str(selected_row["signal_name"]),
        test_seq_pred,
        test_seq_label,
        signal_profile=args.signal_profile,
    )

    recorder_name = f"{selected_row['trial_name']}_{args.result_suffix}_test"
    cache_path = stage2._write_fused_cache(recorder_name, test_pred, test_label)
    metrics = stage2._evaluate_signal(
        recorder_name=recorder_name,
        pred=test_pred,
        label=test_label,
        strategy_cfg=stage2._strategy_cfg_from_trial(tra_validation, strategy_trial, eval_segment="test"),
        experiment_name="alpha360_tra_stage2_cv_selected_final_test",
        fast_mode=bool(args.fast_mode),
    )

    summary = {
        "run_mode": "stage2_cv_selected_final_test_once",
        "selection_scope": "validation_cv_only",
        "test_usage_policy": "single_final_evaluation_after_cv_candidate_fixed",
        "target_test_ann": args.target_test_ann,
        "legacy_target_test_ann": TARGET_TEST_ANN,
        "current_best_test_ann": CURRENT_BEST_TEST_ANN,
        "test_improves_target_ann": float(metrics["with_cost_ann_return"]) > args.target_test_ann,
        "selection_rule": SELECTION_RULE,
        "cv_grid_path": str(args.cv_grid_path),
        "selected_cv_row": selected_row,
        "window_key": tra_validation["window_key"],
        "train": tra_validation["train"],
        "valid": tra_validation["valid"],
        "test": tra_validation["test"],
        "base_model_trial": "rank_ensemble_3seed_tra72_old6",
        "stage1_test_common_rows": test_common_rows,
        "stage1_test_seed_cache_paths": [str(path) for path in test_cache_paths],
        "signal_name": selected_row["signal_name"],
        "signal_kind": selected_row["kind"],
        "signal_profile": args.signal_profile,
        "fundamental_feature_path": stage2.FUNDAMENTAL_FEATURE_PATH,
        "announcement_feature_path": (
            str(cv.ANNOUNCEMENT_DECAY_RANKPCT_FEATURE_PATH)
            if "announcement" in str(args.signal_profile)
            else None
        ),
        "strategy_trial": strategy_trial["trial_name"],
        "strategy_class": strategy_trial["strategy_class"],
        "strategy_module_path": strategy_trial["strategy_module_path"],
        "n_drop": strategy_trial["n_drop"],
        "risk_degree": strategy_trial["risk_degree"],
        "kwargs_extra": strategy_trial["kwargs_extra"],
        "test_cache_path": str(cache_path),
        "test_recorder_id": metrics["recorder_id"],
        "test_IC": metrics["IC"],
        "test_Rank_IC": metrics["Rank IC"],
        "test_with_cost_ann_return": metrics["with_cost_ann_return"],
        "test_with_cost_ir": metrics["with_cost_ir"],
        "test_with_cost_mdd": metrics["with_cost_mdd"],
        "test_score": metrics["score"],
    }

    json_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.json")
    txt_path = args.output_prefix.with_name(f"{args.output_prefix.name}.txt")
    json_path.write_text(json.dumps(fast._jsonable(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    _write_text_summary(fast._jsonable(summary), txt_path)
    print(json_path)
    print(txt_path)
    print(
        "selected_trial="
        f"{selected_row['trial_name']} "
        "test_ann="
        f"{float(metrics['with_cost_ann_return'])} "
        f"test_ir={float(metrics['with_cost_ir'])} "
        f"test_mdd={float(metrics['with_cost_mdd'])}"
    )


if __name__ == "__main__":
    main()
