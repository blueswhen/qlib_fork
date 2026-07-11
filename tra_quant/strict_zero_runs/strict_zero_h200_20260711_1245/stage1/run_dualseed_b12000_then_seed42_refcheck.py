from __future__ import annotations

import argparse
import contextlib
import importlib.util
import subprocess
import sys
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
PYTHON = Path(sys.executable)

FORMAL_SUFFIX = "dualseed_formal_seedxroll_b12000_20260514_1"
SEED42_SUFFIX = "seed42_refcheck_tra18_b16384_20260514_1"

DUAL_LOG = BASE_DIR / f"{FORMAL_SUFFIX}.log"
SEED42_LOG = BASE_DIR / f"{SEED42_SUFFIX}.log"
COMPARE_LOG = BASE_DIR / f"{SEED42_SUFFIX}_comparison.txt"
DEFAULT_VALIDATION_REFERENCE_SUMMARY_PATH = BASE_DIR / "tra_strict_validation_best.txt"


def now() -> str:
    return datetime.now().strftime("%F %T")


def parse_summary(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def as_float(data: dict[str, str], key: str) -> float | None:
    value = data.get(key)
    if value in (None, "None"):
        return None
    return float(value)


def _resolve_result_path(summary_path: Path, key: str) -> Path:
    summary = parse_summary(summary_path)
    result_path = summary.get(key)
    if not result_path:
        raise KeyError(f"{key} not found in {summary_path}")
    path = Path(result_path)
    if not path.exists():
        raise FileNotFoundError(f"missing result path referenced by {summary_path}: {path}")
    return path


def _find_generated_result_path(eval_segment: str) -> Path:
    matches = sorted(BASE_DIR.glob(f"rolling_result_tra_*_{SEED42_SUFFIX}_{eval_segment}.txt"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one generated result for {SEED42_SUFFIX} {eval_segment}, found {len(matches)}: {matches}"
        )
    return matches[0]


def write_comparison(validation_reference_summary_path: Path, test_reference_summary_path: Path | None) -> None:
    pairs = [
        (
            "valid_base_r07",
            _resolve_result_path(validation_reference_summary_path, "validation_result_path"),
            _find_generated_result_path("valid"),
            {
                "IC": "IC",
                "Rank IC": "Rank IC",
                "with_cost_ann_return": "with_cost_ann_return",
                "with_cost_ir": "with_cost_ir",
                "with_cost_mdd": "with_cost_mdd",
            },
        ),
    ]
    if test_reference_summary_path is not None:
        pairs.append(
            (
                "test_reference",
                _resolve_result_path(test_reference_summary_path, "test_result_path"),
                _find_generated_result_path("test"),
                {
                    "test_IC": "IC",
                    "test_Rank IC": "Rank IC",
                    "test_with_cost_ann_return": "with_cost_ann_return",
                    "test_with_cost_ir": "with_cost_ir",
                    "test_with_cost_mdd": "with_cost_mdd",
                },
            )
        )
    lines: list[str] = []
    for name, old_path, new_path, key_map in pairs:
        old = parse_summary(old_path)
        new = parse_summary(new_path)
        lines.extend([f"[{name}]", f"old={old_path}", f"new={new_path}"])
        for old_key, new_key in key_map.items():
            old_value = as_float(old, old_key)
            new_value = as_float(new, new_key)
            delta = None if old_value is None or new_value is None else new_value - old_value
            lines.append(f"{old_key} -> {new_key}: old={old_value} new={new_value} delta={delta}")
        lines.append("")
    COMPARE_LOG.write_text("\n".join(lines), encoding="utf-8")


def load_tuner():
    path = BASE_DIR / "tune_tra_alpha360_global.py"
    spec = importlib.util.spec_from_file_location("tune_tra_alpha360_global_seed42_refcheck", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load tuner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_seed42_refcheck() -> None:
    tuner = load_tuner()
    model_trial = tuner.copy.deepcopy(tuner.MODEL_TRIALS[0])
    common = dict(
        model_family="tra",
        model_trial=model_trial,
        seed=42,
        gpu_slots="0,1",
        deterministic_runtime=True,
        window_key=tuner.WINDOW_KEY,
        n_drop=1,
        strategy_class=None,
        strategy_module_path=None,
        strategy_kwargs_extra=None,
        dataset_kwargs_extra={"batch_size": 16384},
        rolling_task_limit=None,
        parallel_mode="fork_readonly",
    )
    valid = tuner._run_family_trial(
        result_suffix=f"{SEED42_SUFFIX}_valid",
        risk_degree=0.70,
        eval_segment="valid",
        backtest_segment="valid",
        **common,
    )
    print("seed42_valid_result=", valid)
    test = tuner._run_family_trial(
        result_suffix=f"{SEED42_SUFFIX}_test",
        risk_degree=0.85,
        eval_segment="test",
        backtest_segment="test",
        **common,
    )
    print("seed42_test_result=", test)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dual-seed formal study and compare against summary-driven references.")
    parser.add_argument("--validation-reference-summary-path", default=str(DEFAULT_VALIDATION_REFERENCE_SUMMARY_PATH))
    parser.add_argument("--test-reference-summary-path")
    args = parser.parse_args()

    validation_reference_summary_path = Path(args.validation_reference_summary_path).expanduser().resolve()
    test_reference_summary_path = (
        Path(args.test_reference_summary_path).expanduser().resolve() if args.test_reference_summary_path else None
    )

    dual_command = [
        str(PYTHON),
        str(BASE_DIR / "tune_tra_alpha360_global.py"),
        "--dual-seed-study",
        "--result-suffix",
        FORMAL_SUFFIX,
        "--batch-size",
        "12000",
        "--gpu-slots",
        "0,1",
        "--max-concurrent-seeds",
        "2",
        "--min-available-ram-gb",
        "6",
        "--max-extra-swap-gb",
        "48",
        "--resource-poll-seconds",
        "2",
    ]
    with DUAL_LOG.open("w", encoding="utf-8") as log:
        log.write(f"[{now()}] START dual-seed formal suffix={FORMAL_SUFFIX}\n")
        log.flush()
        status = subprocess.run(dual_command, cwd=ROOT_DIR, stdout=log, stderr=subprocess.STDOUT).returncode
        log.write(f"[{now()}] END dual-seed formal status={status}\n")
    if status != 0:
        SEED42_LOG.write_text("dual-seed formal failed; seed42 refcheck not started\n", encoding="utf-8")
        return status

    with SEED42_LOG.open("w", encoding="utf-8") as log:
        log.write(f"[{now()}] START seed42 base refcheck suffix={SEED42_SUFFIX}\n")
        log.flush()
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            run_seed42_refcheck()
        log.write(f"[{now()}] END seed42 base refcheck status=0\n")

    write_comparison(validation_reference_summary_path, test_reference_summary_path)
    with SEED42_LOG.open("a", encoding="utf-8") as log:
        log.write(f"[{now()}] comparison saved to {COMPARE_LOG}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())