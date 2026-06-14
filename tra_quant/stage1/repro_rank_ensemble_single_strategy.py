from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RUNNER_PATH = BASE_DIR / "run_rank_ensemble_tra_alpha360_once.py"
DEFAULT_OUTPUT_PATH = BASE_DIR / "tmp" / "repro_rank_ensemble_single_strategy.json"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("rank_ensemble_single_strategy_repro", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _resolve_cache_paths(module, explicit_paths: list[str], summary_path: str | None, seed_runs_key: str) -> list[Path]:
    if explicit_paths:
        paths = [Path(path).expanduser().resolve() for path in explicit_paths]
    elif summary_path:
        paths = module._load_seed_cache_paths(Path(summary_path).expanduser().resolve(), seed_runs_key)
    else:
        raise ValueError(f"either explicit cache paths or summary path is required for {seed_runs_key}")

    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("\n".join(missing))
    return paths


def _find_trial(module, strategy_trial: str) -> dict:
    for trial in module.build_candidates(module.load_module()):
        if trial["trial_name"] == strategy_trial:
            return trial
    raise KeyError(f"unknown strategy_trial: {strategy_trial}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce one TRA rank-ensemble strategy on validation and test.")
    parser.add_argument("--strategy-trial", required=True)
    parser.add_argument("--validation-cache-path", action="append", default=[])
    parser.add_argument("--test-cache-path", action="append", default=[])
    parser.add_argument("--validation-summary-path")
    parser.add_argument("--final-test-summary-path")
    parser.add_argument("--validation-recorder-name", default="rank_ensemble_single_strategy_repro_valid")
    parser.add_argument("--test-recorder-name", default=None)
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    runner = _load_runner_module()
    tra = runner.load_module()
    trial = next(t for t in runner.build_candidates(tra) if t["trial_name"] == args.strategy_trial)

    valid_cache_paths = _resolve_cache_paths(
        runner,
        args.validation_cache_path,
        args.validation_summary_path,
        "seed_runs_valid",
    )
    test_cache_paths = _resolve_cache_paths(
        runner,
        args.test_cache_path,
        args.final_test_summary_path,
        "seed_runs_test",
    )

    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            valid_pred, valid_label, valid_rows = runner.rank_ensemble_signal(tra, valid_cache_paths)
            valid_result = tra._evaluate_strategy(
                args.validation_recorder_name,
                trial,
                valid_pred,
                valid_label,
                window_key="w1",
            )

            test_pred, test_label, test_rows = runner.rank_ensemble_signal(tra, test_cache_paths)
            test_result = tra._evaluate_signal(
                args.test_recorder_name or f"rank_ensemble_single_strategy__{trial['trial_name']}__test",
                trial,
                test_pred,
                test_label,
                window_key="w1",
                eval_segment="test",
                backtest_segment="test",
                recorder_name=args.test_recorder_name or f"rank_ensemble_single_strategy__{trial['trial_name']}__test",
            )

    payload = {
        "strategy_trial": trial["trial_name"],
        "hold_thresh": trial["kwargs_extra"]["hold_thresh"],
        "score_margin": trial["kwargs_extra"]["score_margin"],
        "n_drop": trial["n_drop"],
        "risk_degree": trial["risk_degree"],
        "validation_cache_paths": [str(path) for path in valid_cache_paths],
        "test_cache_paths": [str(path) for path in test_cache_paths],
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

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()