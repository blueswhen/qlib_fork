from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parents[1]
OUTPUT_PATH = BASE_DIR / "tmp" / "repro_old_rank_ensemble_paths.json"


RESTORE_SOURCES = {
    BASE_DIR / "tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_strict_tra_lstm_s240_h5_tra18_residual_step1_20260506_1.pkl": (
        BASE_DIR / "mlruns/198111276539745747/e3ab04501c21438e8ca480b8ac85a6ab/artifacts/pred.pkl",
        BASE_DIR / "mlruns/198111276539745747/e3ab04501c21438e8ca480b8ac85a6ab/artifacts/label.pkl",
    ),
    BASE_DIR / "tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_dualseed_s2026_valid_dualseed_formal_seedxroll_b12000_20260514_1.pkl": (
        REPO_ROOT / "mlruns/943174324210839437/8a5d3ba4bd374b3a9fc297fbe1ff0a7c/artifacts/pred.pkl",
        REPO_ROOT / "mlruns/943174324210839437/8a5d3ba4bd374b3a9fc297fbe1ff0a7c/artifacts/label.pkl",
    ),
    BASE_DIR / "tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_dualseed_s3407_valid_dualseed_formal_seedxroll_b12000_20260514_1.pkl": (
        REPO_ROOT / "mlruns/872892592345132733/c9f9fc21c69e4991bcd2d5f321ce2029/artifacts/pred.pkl",
        REPO_ROOT / "mlruns/872892592345132733/c9f9fc21c69e4991bcd2d5f321ce2029/artifacts/label.pkl",
    ),
}


def _load_runner_module():
    for path in (str(REPO_ROOT), str(BASE_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)

    script_path = BASE_DIR / "run_rank_ensemble_tra_alpha360_once.py"
    spec = importlib.util.spec_from_file_location("rank_ensemble_once_repro_old", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = _load_runner_module()
    tra = module.load_module()
    trial = next(t for t in module.build_candidates(tra) if t["trial_name"] == "prac_m000_hold3_r085")

    valid_cache_paths = [
        BASE_DIR / "tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_strict_tra_lstm_s240_h5_tra18_residual_step1_20260506_1.pkl",
        BASE_DIR / "tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_dualseed_s2026_valid_dualseed_formal_seedxroll_b12000_20260514_1.pkl",
        BASE_DIR / "tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_dualseed_s3407_valid_dualseed_formal_seedxroll_b12000_20260514_1.pkl",
    ]
    test_cache_paths = [
        BASE_DIR / "tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r085_strict_tra_lstm_s240_h5_final_test_tra18_residual_step1_20260506_1.pkl",
        BASE_DIR / "tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_dualseed_s2026_test_dualseed_formal_seedxroll_b12000_20260514_1.pkl",
        BASE_DIR / "tra_cache/rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_dualseed_s3407_test_dualseed_formal_seedxroll_b12000_20260514_1.pkl",
    ]

    for cache_path, (pred_path, label_path) in RESTORE_SOURCES.items():
        if cache_path.exists():
            continue
        if not pred_path.exists() or not label_path.exists():
            raise FileNotFoundError(f"missing restore source for {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"pred": pd.read_pickle(pred_path), "label": pd.read_pickle(label_path)}, cache_path)

    missing = [str(path) for path in valid_cache_paths + test_cache_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("\n".join(missing))

    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        valid_pred, valid_label, valid_rows = module.rank_ensemble_signal(tra, valid_cache_paths)
        valid_result = tra._evaluate_strategy(
            "rank_ensemble_seed42_2026_3407_old_valid",
            trial,
            valid_pred,
            valid_label,
            window_key="w1",
        )
        test_pred, test_label, test_rows = module.rank_ensemble_signal(tra, test_cache_paths)
        test_result = tra._evaluate_signal(
            "rank_ensemble_seed42_2026_3407_old_test",
            trial,
            test_pred,
            test_label,
            window_key="w1",
            eval_segment="test",
            backtest_segment="test",
            recorder_name="rank_ensemble_seed42_2026_3407__prac_m000_hold3_r085__repro_restore_paths",
        )

    payload = {
        "strategy_trial": trial["trial_name"],
        "valid_common_rows": int(valid_rows),
        "validation": {
            "score": float(valid_result["score"]),
            "ann": float(valid_result["with_cost_ann_return"]),
            "ir": float(valid_result["with_cost_ir"]),
            "mdd": float(valid_result["with_cost_mdd"]),
            "IC": float(valid_result["IC"]),
            "Rank IC": float(valid_result["Rank IC"]),
        },
        "test_common_rows": int(test_rows),
        "test": {
            "ann": float(test_result["with_cost_ann_return"]),
            "ir": float(test_result["with_cost_ir"]),
            "mdd": float(test_result["with_cost_mdd"]),
            "IC": float(test_result["IC"]),
            "Rank IC": float(test_result["Rank IC"]),
        },
    }

    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()