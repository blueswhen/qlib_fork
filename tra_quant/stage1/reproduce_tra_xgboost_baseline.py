from __future__ import annotations

import argparse
import json
import subprocess
import sys
from ast import literal_eval
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

STAGE1_SCRIPT = BASE_DIR / "tune_tra_alpha360_strict.py"
STAGE2_SCRIPT = BASE_DIR / "tune_alpha360_tra_alpha158_xgb_strict_fusion.py"

STAGE1_VALIDATION_PATH = BASE_DIR / "tra_strict_validation_best.txt"
STAGE1_FINAL_TEST_PATH = BASE_DIR / "tra_strict_final_test.txt"
STAGE2_VALIDATION_PATH = BASE_DIR / "alpha360_tra_alpha158_xgb_strict_validation_best.txt"
STAGE2_FINAL_TEST_PATH = BASE_DIR / "alpha360_tra_alpha158_xgb_strict_final_test.txt"

TRA_DNN_VALIDATION_PATH = BASE_DIR / "alpha360_tra_alpha158_dnn_weight_refine_validation_best.txt"
TRA_DNN_FINAL_TEST_PATH = BASE_DIR / "alpha360_tra_alpha158_dnn_weight_refine_final_test.txt"

MANIFEST_PATH = BASE_DIR / "tra_xgboost_baseline_manifest.json"


def _parse_scalar(value: str):
    try:
        return literal_eval(value)
    except Exception:
        return value


def _parse_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"summary file not found: {path}")
    data: dict[str, object] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = _parse_scalar(value)
    return data


def _run_python(script: Path, extra_args: list[str] | None = None):
    cmd = [sys.executable, str(script)]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, cwd=BASE_DIR, check=True)


def _build_manifest() -> dict:
    stage1_valid = _parse_summary(STAGE1_VALIDATION_PATH)
    stage1_test = _parse_summary(STAGE1_FINAL_TEST_PATH)
    stage2_valid = _parse_summary(STAGE2_VALIDATION_PATH)
    stage2_test = _parse_summary(STAGE2_FINAL_TEST_PATH)
    tra_dnn_valid = _parse_summary(TRA_DNN_VALIDATION_PATH)
    tra_dnn_test = _parse_summary(TRA_DNN_FINAL_TEST_PATH)

    return {
        "baseline_name": "alpha360_tra_plus_alpha158_xgboost_strict_fusion",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(BASE_DIR),
        "reproduction": {
            "stage1_script": str(STAGE1_SCRIPT),
            "stage2_script": str(STAGE2_SCRIPT),
            "stage1_validation_summary": str(STAGE1_VALIDATION_PATH),
            "stage1_final_test_summary": str(STAGE1_FINAL_TEST_PATH),
            "stage2_validation_summary": str(STAGE2_VALIDATION_PATH),
            "stage2_final_test_summary": str(STAGE2_FINAL_TEST_PATH),
        },
        "data_split": {
            "window_key": stage2_valid["window_key"],
            "train": stage2_valid["train"],
            "valid": stage2_valid["valid"],
            "test": stage2_valid["test"],
        },
        "stage1_incumbent": {
            "model_trial": stage1_valid["model_trial"],
            "model_key": stage1_valid["model_key"],
            "step": stage1_valid["step"],
            "handler_kwargs_extra": stage1_valid["handler_kwargs_extra"],
            "strategy_trial": stage1_valid["strategy_trial"],
            "strategy_class": stage1_valid["strategy_class"],
            "strategy_module_path": stage1_valid["strategy_module_path"],
            "n_drop": stage1_valid["n_drop"],
            "risk_degree": stage1_valid["risk_degree"],
            "kwargs_extra": stage1_valid["kwargs_extra"],
            "validation": {
                "IC": stage1_valid["validation_IC"],
                "Rank_IC": stage1_valid["validation_Rank_IC"],
                "with_cost_ann_return": stage1_valid["validation_with_cost_ann_return"],
                "with_cost_ir": stage1_valid["validation_with_cost_ir"],
                "with_cost_mdd": stage1_valid["validation_with_cost_mdd"],
                "score": stage1_valid["validation_score"],
                "cache_path": stage1_valid["validation_cache_path"],
                "result_path": stage1_valid["validation_result_path"],
            },
            "final_test": {
                "IC": stage1_test["test_IC"],
                "Rank_IC": stage1_test["test_Rank_IC"],
                "with_cost_ann_return": stage1_test["test_with_cost_ann_return"],
                "with_cost_ir": stage1_test["test_with_cost_ir"],
                "with_cost_mdd": stage1_test["test_with_cost_mdd"],
                "result_path": stage1_test["test_result_path"],
                "cache_path": stage1_test["test_cache_path"],
                "recorder_id": stage1_test["test_recorder_id"],
            },
        },
        "xgboost_fusion_baseline": {
            "selection_scope": stage2_valid["selection_scope"],
            "base_model_trial": stage2_valid["base_model_trial"],
            "base_strategy_trial": stage2_valid["base_strategy_trial"],
            "tabular_trial": stage2_valid["tabular_trial"],
            "tabular_handler_class": stage2_valid["tabular_handler_class"],
            "tabular_model_key": stage2_valid["tabular_model_key"],
            "fusion_mode": stage2_valid["fusion_mode"],
            "seq_weight": stage2_valid["seq_weight"],
            "tab_weight": stage2_valid["tab_weight"],
            "validation": {
                "IC": stage2_valid["validation_IC"],
                "Rank_IC": stage2_valid["validation_Rank_IC"],
                "with_cost_ann_return": stage2_valid["validation_with_cost_ann_return"],
                "with_cost_ir": stage2_valid["validation_with_cost_ir"],
                "with_cost_mdd": stage2_valid["validation_with_cost_mdd"],
                "score": stage2_valid["validation_score"],
                "cache_path": stage2_valid["validation_cache_path"],
                "recorder_id": stage2_valid["validation_recorder_id"],
                "tabular_result_path": stage2_valid["tabular_validation_result_path"],
                "tabular_recorder_id": stage2_valid["tabular_validation_recorder_id"],
            },
            "final_test": {
                "IC": stage2_test["test_IC"],
                "Rank_IC": stage2_test["test_Rank_IC"],
                "with_cost_ann_return": stage2_test["test_with_cost_ann_return"],
                "with_cost_ir": stage2_test["test_with_cost_ir"],
                "with_cost_mdd": stage2_test["test_with_cost_mdd"],
                "cache_path": stage2_test["test_cache_path"],
                "recorder_id": stage2_test["test_recorder_id"],
                "tabular_result_path": stage2_test["tabular_test_result_path"],
                "tabular_recorder_id": stage2_test["tabular_test_recorder_id"],
            },
            "improvement_vs_stage1_incumbent": {
                "ann_return_delta": float(stage2_test["test_with_cost_ann_return"]) - float(stage1_test["test_with_cost_ann_return"]),
                "ir_delta": float(stage2_test["test_with_cost_ir"]) - float(stage1_test["test_with_cost_ir"]),
                "mdd_delta": float(stage2_test["test_with_cost_mdd"]) - float(stage1_test["test_with_cost_mdd"]),
            },
        },
        "tra_dnn_reference": {
            "base_model_trial": tra_dnn_valid["base_model_trial"],
            "base_strategy_trial": tra_dnn_valid["base_strategy_trial"],
            "tabular_trial": tra_dnn_valid["tabular_trial"],
            "tabular_handler_class": tra_dnn_valid["tabular_handler_class"],
            "tabular_model_key": tra_dnn_valid["tabular_model_key"],
            "fusion_mode": tra_dnn_valid["fusion_mode"],
            "seq_weight": tra_dnn_valid["seq_weight"],
            "tab_weight": tra_dnn_valid["tab_weight"],
            "validation": {
                "with_cost_ann_return": tra_dnn_valid["validation_with_cost_ann_return"],
                "with_cost_ir": tra_dnn_valid["validation_with_cost_ir"],
                "with_cost_mdd": tra_dnn_valid["validation_with_cost_mdd"],
                "score": tra_dnn_valid["validation_score"],
            },
            "final_test": {
                "with_cost_ann_return": tra_dnn_test["test_with_cost_ann_return"],
                "with_cost_ir": tra_dnn_test["test_with_cost_ir"],
                "with_cost_mdd": tra_dnn_test["test_with_cost_mdd"],
            },
            "comparison_vs_tra_dnn": {
                "ann_return_delta": float(stage2_test["test_with_cost_ann_return"]) - float(tra_dnn_test["test_with_cost_ann_return"]),
                "ir_delta": float(stage2_test["test_with_cost_ir"]) - float(tra_dnn_test["test_with_cost_ir"]),
                "mdd_delta": float(stage2_test["test_with_cost_mdd"]) - float(tra_dnn_test["test_with_cost_mdd"]),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce and register the archived TRA + XGBoost strict fusion baseline."
    )
    parser.add_argument("--force-stage1", action="store_true", help="rerun stage1 strict TRA search and final test")
    parser.add_argument("--force-stage2", action="store_true", help="rerun stage2 TRA+XGBoost strict fusion search and final test")
    parser.add_argument("--manifest-path", default=str(MANIFEST_PATH), help="where to write the machine-readable baseline manifest")
    args = parser.parse_args()

    if args.force_stage1 or not (STAGE1_VALIDATION_PATH.exists() and STAGE1_FINAL_TEST_PATH.exists()):
        _run_python(STAGE1_SCRIPT)

    if args.force_stage2 or not (STAGE2_VALIDATION_PATH.exists() and STAGE2_FINAL_TEST_PATH.exists()):
        _run_python(STAGE2_SCRIPT)

    manifest = _build_manifest()
    manifest_path = Path(args.manifest_path).resolve()
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"baseline manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
