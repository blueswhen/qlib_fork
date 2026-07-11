from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

WORKSPACE_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_FOR_IMPORT))

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

import pandas as pd
import qlib
from qlib.constant import REG_CN

import run_rank_ensemble_tra_alpha360_once as rank_runner
from stage1_plus_conservative_fusion_validation import (
    _build_fusion_candidates,
    _jsonable,
    _load_seed_cache_paths,
)


BASE_DIR = Path(__file__).resolve().parent
RUN_DIR = BASE_DIR.parent.parent / "tra_quant/runs/stage1_h200_20260621_11_4seed_b12000_fast/stage1"
DEFAULT_FINAL_TEST_SUMMARY = (
    RUN_DIR / "tmp/rank_ensemble_h200_final_test_summary_stage1_h200_20260621_11_4seed_b12000_fast.txt"
)
DEFAULT_VALIDATION_LOCK_GRID = (
    BASE_DIR / "tmp/stage1_plus_practical_risk_targeted_stress_20260708_1_stress_grid.csv"
)
DEFAULT_OUTPUT_PATH = (
    BASE_DIR / "tmp/stage1_plus_growth_prac_r0875_final_test_once_20260708_1_summary.json"
)
LOCKED_SIGNAL = "growth_quality_linear_rank_w0001"
LOCKED_STRATEGY = "prac_m000_drop3_hold4_r0875"
LOCKED_RISK_DEGREE = 0.875
LOCKED_GROWTH_WEIGHT = 0.001
REFERENCE_CURRENT_BEST_BASELINE_ANN = 0.26


def _load_validation_lock(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing validation lock grid: {path}")
    df = pd.read_csv(path)
    match = df[(df["candidate"] == LOCKED_SIGNAL) & (df["strategy_trial"] == LOCKED_STRATEGY)]
    if match.empty:
        raise RuntimeError(f"validation lock row not found for {LOCKED_SIGNAL}/{LOCKED_STRATEGY}")
    row = match.iloc[0].to_dict()
    if bool(row.get("stable_over_stage1_pass")) is not True:
        raise RuntimeError(f"locked candidate did not pass validation gate: {row.get('stable_gate_reason')}")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One final test for the validation-locked Stage1+ candidate.")
    parser.add_argument("--final-test-summary-path", type=Path, default=DEFAULT_FINAL_TEST_SUMMARY)
    parser.add_argument("--validation-lock-grid", type=Path, default=DEFAULT_VALIDATION_LOCK_GRID)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--force-final", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.force_final:
        raise FileExistsError(f"final test output already exists; refusing rerun: {output_path}")

    validation_lock_row = _load_validation_lock(args.validation_lock_grid.expanduser().resolve())

    mod = rank_runner.load_module()
    qlib.init(provider_uri=mod.PROVIDER_URI, region=REG_CN)
    test_cache_paths = _load_seed_cache_paths(args.final_test_summary_path.expanduser().resolve(), "seed_runs_test")
    test_pred, test_label, test_rows = rank_runner.rank_ensemble_signal(mod, test_cache_paths)
    signal_candidates = _build_fusion_candidates(
        test_pred.sort_index(),
        test_label.sort_index(),
        [LOCKED_GROWTH_WEIGHT],
        {"growth_quality"},
    )
    signal_by_name = {item["candidate"]: item for item in signal_candidates}
    locked_signal = signal_by_name[LOCKED_SIGNAL]
    locked_strategy = mod._prac(
        LOCKED_STRATEGY,
        3,
        LOCKED_RISK_DEGREE,
        0.0,
        4,
    )
    test_result = rank_runner._evaluate_direct_signal(
        mod,
        locked_strategy,
        locked_signal["pred"],
        locked_signal["label"],
        window_key="w1",
        eval_segment="test",
        backtest_segment="test",
    )
    objective = float(mod._objective(test_result))
    result = {
        "run_mode": "final_test_once_after_validation_lock",
        "selection_scope_before_test": "validation_only",
        "test_usage_policy": "test_loaded_once_after_candidate_lock_no_post_test_tuning",
        "hard_constraints": {"account": int(mod.ACCOUNT), "topk": int(mod.TOPK), "universe": mod.INSTRUMENTS},
        "locked_candidate": {
            "signal": LOCKED_SIGNAL,
            "strategy_trial": LOCKED_STRATEGY,
            "growth_weight": LOCKED_GROWTH_WEIGHT,
            "risk_degree": LOCKED_RISK_DEGREE,
            "validation_lock_row": validation_lock_row,
        },
        "final_test_summary_path": str(args.final_test_summary_path.expanduser().resolve()),
        "test_cache_paths": [str(path) for path in test_cache_paths],
        "test_common_rows": int(test_rows),
        "selected_test_result": {
            "test_score": objective,
            "test_IC": float(test_result["IC"]),
            "test_Rank_IC": float(test_result["Rank IC"]),
            "test_with_cost_ann_return": float(test_result["with_cost_ann_return"]),
            "test_with_cost_ir": float(test_result["with_cost_ir"]),
            "test_with_cost_mdd": float(test_result["with_cost_mdd"]),
        },
        "reference_current_best_baseline_ann_return": REFERENCE_CURRENT_BEST_BASELINE_ANN,
        "beats_reference_current_best_baseline_ann_return": (
            float(test_result["with_cost_ann_return"]) > REFERENCE_CURRENT_BEST_BASELINE_ANN
        ),
        "no_more_test_tuning_allowed": True,
    }
    output_path.write_text(json.dumps(_jsonable(result), indent=2, ensure_ascii=True), encoding="utf-8")
    print(output_path)
    print(
        "final_test_once "
        f"ann={result['selected_test_result']['test_with_cost_ann_return']:.6f} "
        f"ir={result['selected_test_result']['test_with_cost_ir']:.6f} "
        f"mdd={result['selected_test_result']['test_with_cost_mdd']:.6f} "
        f"beats_0.26={result['beats_reference_current_best_baseline_ann_return']}"
    )


if __name__ == "__main__":
    main()
