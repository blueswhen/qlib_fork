from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from tra_local_helpers import localize_value


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = BASE_DIR.parent.parent
DEFAULT_LOCKED_MANIFEST = BASE_DIR / "tmp/h200_strict_validation_selector_cashq_top8_20260705_1_locked_manifest.json"
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp/stage2_strict_h200_cashq_top8_20260705_1_final_once"
DEFAULT_TEST_SUMMARIES = [
    WORKSPACE
    / "tra_quant/runs/stage1_h200_20260621_11_4seed_b12000_fast/stage1/tra_global_dual_seed_pattern_summary_stage1_h200_20260621_11_4seed_b12000_fast_b12000_test.txt",
    WORKSPACE / "tra_quant/stage1/tra_global_dual_seed_pattern_summary_stage2_seed_screen_20260704_1_b12000_test_all8_h200_j4.txt",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _parse_summary(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            data[key] = localize_value(ast.literal_eval(value))
        except Exception:
            data[key] = localize_value(value)
    return data


def _seed_runs_from_test_summary(path: Path) -> dict[int, dict[str, Any]]:
    summary = _parse_summary(path)
    runs = summary.get("seed_runs_test") or summary.get("test_base_runs") or {}
    if isinstance(runs, list):
        return {int(row["seed"]): row for row in runs}
    if isinstance(runs, dict):
        return {int(seed): dict(row, seed=int(seed)) for seed, row in runs.items()}
    raise TypeError(f"unsupported test runs in {path}: {type(runs).__name__}")


def _load_test_seed_runs(paths: list[Path]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in paths:
        out.update(_seed_runs_from_test_summary(path))
    return out


def _write_test_summary(seed_runs: dict[int, dict[str, Any]], combo: tuple[int, ...], path: Path) -> None:
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
        f"seed_runs_test={rows!r}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one locked strict H200 final test after validation-only selection.")
    parser.add_argument("--locked-manifest", type=Path, default=DEFAULT_LOCKED_MANIFEST)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--result-suffix", default=None)
    parser.add_argument("--test-summary-path", action="append", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--recover-interrupted-report",
        action="store_true",
        help="Reuse the frozen fused cache left by a final-test reporting failure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    locked_path = args.locked_manifest.expanduser().resolve()
    if not locked_path.exists():
        raise FileNotFoundError(f"missing locked manifest: {locked_path}")

    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    if locked.get("run_mode") != "strict_validation_locked_candidate":
        raise RuntimeError(f"unexpected locked manifest run_mode: {locked.get('run_mode')}")
    if locked.get("final_test_summary_path"):
        raise RuntimeError("locked manifest already records a final_test_summary_path")

    output_prefix = args.output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    result_suffix = args.result_suffix or output_prefix.name
    summary_json_path = output_prefix.with_name(f"{output_prefix.name}_summary.json")
    summary_txt_path = output_prefix.with_name(f"{output_prefix.name}.txt")
    final_audit_path = output_prefix.with_name(f"{output_prefix.name}_final_audit.json")
    final_test_wrapper_path = output_prefix.with_name(f"{output_prefix.name}_stage1_final_test_summary.txt")
    if summary_json_path.exists() or summary_txt_path.exists() or final_audit_path.exists():
        raise FileExistsError("final output already exists; refusing to run test more than once")

    selected = locked["selected_row"]
    selected_combo = tuple(int(seed) for seed in str(selected["combo"]).split(","))
    test_summary_paths = args.test_summary_path or DEFAULT_TEST_SUMMARIES
    test_seed_runs = _load_test_seed_runs(test_summary_paths)
    missing = [seed for seed in selected_combo if seed not in test_seed_runs]
    if missing:
        raise RuntimeError(f"selected combo missing test seed runs: {missing}")
    if final_test_wrapper_path.exists():
        if not args.recover_interrupted_report:
            raise FileExistsError("final-test wrapper already exists; use explicit report recovery if appropriate")
    else:
        _write_test_summary(test_seed_runs, selected_combo, final_test_wrapper_path)

    locked_cv_row_path_value = locked.get("locked_cv_row_path")
    if locked_cv_row_path_value:
        locked_cv_row_path = Path(locked_cv_row_path_value).expanduser().resolve()
    else:
        locked_cv_row_path = output_prefix.with_name(f"{output_prefix.name}_locked_cv_row.csv")
        pd.DataFrame([selected]).to_csv(locked_cv_row_path, index=False)
    validation_summary_path = Path(locked["validation_summary_path"]).expanduser().resolve()
    signal_profile = str(locked.get("signal_profile", "cash-quality-quick"))
    strategy_profile = str(locked.get("strategy_profile") or selected.get("strategy_profile") or "stable-core")
    combo_text = "_".join(str(seed) for seed in selected_combo)
    base_model_trial = f"rank_ensemble_h200_strict_validation_locked_{combo_text}_b12000"

    command = [
        args.python,
        "evaluate_stage2_cv_selected_final.py",
        "--cv-grid-path",
        str(locked_cv_row_path),
        "--trial-name",
        str(selected["trial_name"]),
        "--output-prefix",
        str(output_prefix),
        "--result-suffix",
        result_suffix,
        "--stage1-validation-summary-path",
        str(validation_summary_path),
        "--stage1-final-test-summary-path",
        str(final_test_wrapper_path),
        "--signal-profile",
        signal_profile,
        "--strategy-profile",
        strategy_profile,
        "--selection-scope",
        "strict_validation_only_locked_before_test",
        "--test-usage-policy",
        "single_test_after_locked_manifest",
        "--base-model-trial",
        base_model_trial,
        "--target-test-ann",
        str(float(locked.get("target_test_ann", 0.2643467300531994))),
    ]
    frozen_fused_cache_path = (
        BASE_DIR
        / "fusion_cache"
        / f"{selected['trial_name']}_{result_suffix}_test.pkl"
    )
    if args.recover_interrupted_report:
        if not frozen_fused_cache_path.exists():
            raise FileNotFoundError(f"missing frozen fused cache for report recovery: {frozen_fused_cache_path}")
        command.extend(["--existing-fused-cache-path", str(frozen_fused_cache_path)])

    env = os.environ.copy()
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    env["PYTHONPATH"] = f"{WORKSPACE}:{env.get('PYTHONPATH', '')}"
    provider_uri = WORKSPACE / "training_data/cn_data_latest"
    env.setdefault("QLIB_PROVIDER_URI", str(provider_uri))

    completed = subprocess.run(command, cwd=BASE_DIR, env=env, text=True, check=True)
    audit = {
        "run_mode": "strict_h200_final_test_once_audit",
        "locked_manifest_path": str(locked_path),
        "locked_at": locked.get("locked_at"),
        "final_evaluated_at": pd.Timestamp.utcnow().isoformat(),
        "selection_scope": "strict_validation_only_locked_before_test",
        "test_usage_policy": (
            "single_final_candidate_fault_recovery_same_frozen_cache"
            if args.recover_interrupted_report
            else "single_test_after_locked_manifest"
        ),
        "report_recovery": bool(args.recover_interrupted_report),
        "report_recovery_reason": (
            "initial backtest completed but summary serialization failed on missing selected-row kind metadata"
            if args.recover_interrupted_report
            else None
        ),
        "reused_frozen_fused_cache_path": (
            str(frozen_fused_cache_path) if args.recover_interrupted_report else None
        ),
        "test_summary_paths_loaded_after_lock": [str(path) for path in test_summary_paths],
        "stage1_final_test_summary_path": str(final_test_wrapper_path),
        "final_summary_path": str(summary_json_path),
        "final_txt_path": str(summary_txt_path),
        "selected_row": selected,
        "command": command,
        "subprocess_returncode": completed.returncode,
    }
    final_audit_path.write_text(json.dumps(_jsonable(audit), indent=2, ensure_ascii=True), encoding="utf-8")
    locked["final_test_summary_path"] = str(summary_json_path)
    locked["final_test_audit_path"] = str(final_audit_path)
    locked_path.write_text(json.dumps(_jsonable(locked), indent=2, ensure_ascii=True), encoding="utf-8")
    print(final_audit_path)


if __name__ == "__main__":
    main()
