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
CONSUMED_FIXED_TEST_END = pd.Timestamp("2026-04-01")


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


def _runs_from_holdout_summary(path: Path) -> dict[int, dict[str, Any]]:
    summary = _parse_summary(path)
    runs = (
        summary.get("seed_runs_holdout")
        or summary.get("holdout_base_runs")
        or summary.get("seed_runs_test")
        or summary.get("test_base_runs")
    )
    if isinstance(runs, list):
        return {int(row["seed"]): dict(row, seed=int(row["seed"])) for row in runs}
    if isinstance(runs, dict):
        return {int(seed): dict(row, seed=int(seed)) for seed, row in runs.items()}
    raise TypeError(f"unsupported holdout runs in {path}: {type(runs).__name__}")


def _load_holdout_seed_runs(paths: list[Path]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in paths:
        out.update(_runs_from_holdout_summary(path))
    return out


def _holdout_window(paths: list[Path], min_start: pd.Timestamp) -> list[str]:
    windows: list[list[str]] = []
    for path in paths:
        summary = _parse_summary(path)
        window = summary.get("holdout") or summary.get("test")
        if not isinstance(window, list) or len(window) != 2:
            raise RuntimeError(f"holdout summary has no holdout/test date window: {path}")
        start = pd.Timestamp(window[0])
        if start < min_start:
            raise RuntimeError(
                f"holdout starts before allowed future window: {path} start={window[0]} min_start={min_start.date()}"
            )
        windows.append([str(window[0]), str(window[1])])
    first = windows[0]
    if any(window != first for window in windows):
        raise RuntimeError(f"holdout summary paths disagree on holdout window: {windows}")
    return first


def _write_holdout_summary(seed_runs: dict[int, dict[str, Any]], combo: tuple[int, ...], window: list[str], path: Path) -> None:
    rows = []
    for seed in combo:
        row = dict(seed_runs[seed])
        row["seed"] = int(seed)
        rows.append(row)
    lines = [
        "window_key=w1",
        "train=['2012-01-01', '2019-12-31']",
        "valid=['2020-01-01', '2022-12-31']",
        f"test={window!r}",
        f"seed_runs_test={rows!r}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_holdout_validation_context(original_validation_summary: Path, window: list[str], path: Path) -> None:
    summary = _parse_summary(original_validation_summary)
    seed_runs_valid = summary.get("seed_runs_valid") or summary.get("valid_base_runs") or []
    lines = [
        f"window_key={summary.get('window_key', 'w1')}",
        f"train={summary.get('train', ['2012-01-01', '2019-12-31'])!r}",
        f"valid={summary.get('valid', ['2020-01-01', '2022-12-31'])!r}",
        f"test={window!r}",
        f"seed_runs_valid={seed_runs_valid!r}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one future holdout evaluation after validation-only lock.")
    parser.add_argument("--locked-manifest", type=Path, required=True)
    parser.add_argument("--holdout-summary-path", action="append", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--result-suffix", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--min-holdout-start", default="2026-04-02")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    locked_path = args.locked_manifest.expanduser().resolve()
    locked = json.loads(locked_path.read_text(encoding="utf-8"))
    if locked.get("run_mode") != "strict_validation_locked_candidate":
        raise RuntimeError(f"unexpected locked manifest run_mode: {locked.get('run_mode')}")
    if locked.get("final_test_summary_path"):
        raise RuntimeError("locked manifest already records a fixed-test/final summary; future holdout requires an untested lock")
    if locked.get("future_holdout_summary_path"):
        raise RuntimeError("locked manifest already records a future_holdout_summary_path")

    output_prefix = args.output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    result_suffix = args.result_suffix or output_prefix.name
    summary_json_path = output_prefix.with_name(f"{output_prefix.name}_summary.json")
    summary_txt_path = output_prefix.with_name(f"{output_prefix.name}.txt")
    holdout_audit_path = output_prefix.with_name(f"{output_prefix.name}_future_holdout_audit.json")
    holdout_wrapper_path = output_prefix.with_name(f"{output_prefix.name}_stage1_future_holdout_summary.txt")
    validation_context_path = output_prefix.with_name(f"{output_prefix.name}_validation_context_with_future_holdout.txt")
    if summary_json_path.exists() or summary_txt_path.exists() or holdout_audit_path.exists():
        raise FileExistsError("future holdout output already exists; refusing to run holdout more than once")

    min_start = pd.Timestamp(args.min_holdout_start)
    if min_start <= CONSUMED_FIXED_TEST_END:
        raise RuntimeError(f"min holdout start must be after consumed fixed test end {CONSUMED_FIXED_TEST_END.date()}")
    holdout_summary_paths = [path.expanduser().resolve() for path in args.holdout_summary_path]
    holdout_window = _holdout_window(holdout_summary_paths, min_start)

    selected = locked["selected_row"]
    selected_combo = tuple(int(seed) for seed in str(selected["combo"]).split(","))
    holdout_seed_runs = _load_holdout_seed_runs(holdout_summary_paths)
    missing = [seed for seed in selected_combo if seed not in holdout_seed_runs]
    if missing:
        raise RuntimeError(f"selected combo missing future holdout seed runs: {missing}")
    _write_holdout_summary(holdout_seed_runs, selected_combo, holdout_window, holdout_wrapper_path)
    _write_holdout_validation_context(Path(locked["validation_summary_path"]).expanduser().resolve(), holdout_window, validation_context_path)

    locked_cv_row_path = Path(locked["locked_cv_row_path"]).expanduser().resolve()
    signal_profile = str(locked.get("signal_profile", "robust-small"))
    strategy_profile = str(locked.get("strategy_profile") or selected.get("strategy_profile") or "stable-core")
    combo_text = "_".join(str(seed) for seed in selected_combo)
    base_model_trial = f"rank_ensemble_h200_future_holdout_locked_{combo_text}"

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
        str(validation_context_path),
        "--stage1-final-test-summary-path",
        str(holdout_wrapper_path),
        "--signal-profile",
        signal_profile,
        "--strategy-profile",
        strategy_profile,
        "--selection-scope",
        "strict_validation_only_locked_before_future_holdout",
        "--test-usage-policy",
        "single_future_holdout_after_locked_manifest",
        "--base-model-trial",
        base_model_trial,
        "--target-test-ann",
        str(float(locked.get("target_test_ann", 0.2643467300531994))),
    ]

    env = os.environ.copy()
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    env["PYTHONPATH"] = f"{WORKSPACE}:{env.get('PYTHONPATH', '')}"
    env.setdefault("QLIB_PROVIDER_URI", str(WORKSPACE / "training_data/cn_data_latest"))

    completed = subprocess.run(command, cwd=BASE_DIR, env=env, text=True, check=True)
    audit = {
        "run_mode": "h200_future_holdout_once_audit",
        "locked_manifest_path": str(locked_path),
        "locked_at": locked.get("locked_at"),
        "future_holdout_evaluated_at": pd.Timestamp.utcnow().isoformat(),
        "selection_scope": "strict_validation_only_locked_before_future_holdout",
        "test_usage_policy": "single_future_holdout_after_locked_manifest",
        "holdout_window": holdout_window,
        "min_holdout_start": args.min_holdout_start,
        "holdout_summary_paths_loaded_after_lock": [str(path) for path in holdout_summary_paths],
        "stage1_future_holdout_summary_path": str(holdout_wrapper_path),
        "validation_context_with_future_holdout_path": str(validation_context_path),
        "future_holdout_summary_path": str(summary_json_path),
        "future_holdout_txt_path": str(summary_txt_path),
        "selected_row": selected,
        "command": command,
        "subprocess_returncode": completed.returncode,
    }
    holdout_audit_path.write_text(json.dumps(_jsonable(audit), indent=2, ensure_ascii=True), encoding="utf-8")
    locked["future_holdout_summary_path"] = str(summary_json_path)
    locked["future_holdout_audit_path"] = str(holdout_audit_path)
    locked_path.write_text(json.dumps(_jsonable(locked), indent=2, ensure_ascii=True), encoding="utf-8")
    print(holdout_audit_path)


if __name__ == "__main__":
    main()
