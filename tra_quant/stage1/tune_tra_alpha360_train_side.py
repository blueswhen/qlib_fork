"""Training-side TRA Alpha360 tuner.

Goal: recover a stable TRA h5 baseline quickly before widening any strategy search.

This script keeps the data path, horizon, and evaluation strategy fixed and runs
a fast baseline-recovery search around ``tra_lstm_base`` on the validation
segment. Validation intentionally uses fewer rolling tasks than the final
confirmation run so we can identify a reproducible baseline model faster.

Selection target:
    score = ann*1000 + ir*50 + mdd*20 + IC*200 + RankIC*100

The strategy is intentionally fixed to the historical baseline-equivalent setup:
``PracticalTopkDropoutStrategy(score_margin=0.005, hold_thresh=5, risk=0.70)``.
"""

from __future__ import annotations

import argparse
import copy
from ast import literal_eval
from pathlib import Path

import pandas as pd

from rolling_tra_alpha360_latest import PROVIDER_URI, WINDOWS, run


BASE_DIR = Path(__file__).resolve().parent
VALIDATION_RESULT_PATH = BASE_DIR / "tra_train_validation_results.csv"
VALIDATION_BEST_PATH = BASE_DIR / "tra_train_validation_best.txt"
FINAL_TEST_PATH = BASE_DIR / "tra_train_final_test.txt"

ACCOUNT = 150000
TOPK = 5
WINDOW_KEY = "w1"
BENCHMARK = "SH000300"
INSTRUMENTS = "csi300"
GPU_SLOTS = "0,1"
MODEL_KEY = "tra_lstm_base"
STEP = 240
DATASET_KWARGS = {"batch_size": 16384}
EVAL_FREQ = 5
VALIDATION_ROLLING_TASK_LIMIT = 2

H5_LABEL = {"label": ["Ref($close, -6) / Ref($close, -1) - 1"]}

FIXED_STRATEGY = {
    "trial_name": "prac_m050_hold5_r07",
    "strategy_class": "PracticalTopkDropoutStrategy",
    "strategy_module_path": "qlib.contrib.strategy.custom_signal_strategy",
    "n_drop": 1,
    "risk_degree": 0.70,
    "kwargs_extra": {
        "score_margin": 0.005,
        "hold_thresh": 5,
        "slot_budget_ratio": 1.0,
    },
}

OBJECTIVE_FORMULA = "ann*1000 + ir*50 + mdd*20 + IC*200 + RankIC*100"


def _trial(
    trial_name: str,
    *,
    model_hidden: int = 64,
    model_layers: int = 2,
    model_dropout: float = 0.0,
    tra_states: int = 3,
    tra_hidden: int = 32,
    tra_layers: int = 1,
    tra_dropout: float = 0.0,
    lr: float = 1e-3,
    n_epochs: int = 100,
    early_stop: int = 20,
    update_freq: int = 1,
    lamb: float = 1.0,
    rho: float = 0.99,
    alpha: float = 0.5,
    eval_freq: int = EVAL_FREQ,
    pretrain: bool = True,
):
    return {
        "trial_name": trial_name,
        "model_key": MODEL_KEY,
        "step": STEP,
        "handler_kwargs_extra": H5_LABEL,
        "model_kwargs": {
            "model_config": {
                "input_size": 6,
                "hidden_size": model_hidden,
                "num_layers": model_layers,
                "rnn_arch": "LSTM",
                "use_attn": True,
                "dropout": model_dropout,
            },
            "tra_config": {
                "num_states": tra_states,
                "rnn_arch": "LSTM",
                "hidden_size": tra_hidden,
                "num_layers": tra_layers,
                "dropout": tra_dropout,
                "tau": 1.0,
                "src_info": "LR_TPE",
            },
            "model_type": "RNN",
            "lr": lr,
            "n_epochs": n_epochs,
            "early_stop": early_stop,
            "update_freq": update_freq,
            "eval_freq": eval_freq,
            "lamb": lamb,
            "rho": rho,
            "alpha": alpha,
            "pretrain": pretrain,
            "transport_method": "router",
            "memory_mode": "sample",
        },
    }


MODEL_TRIALS = [
    _trial("base_ref"),
    _trial("base_longer_140", n_epochs=140, early_stop=30),
    _trial("base_lr5e4", lr=5e-4, n_epochs=140, early_stop=30),
    _trial("base_no_pretrain", pretrain=False, n_epochs=140, early_stop=30),
]


def _objective(metrics: dict) -> float:
    ann = float(metrics["with_cost_ann_return"])
    ir = float(metrics["with_cost_ir"])
    mdd = float(metrics["with_cost_mdd"])
    ic = float(metrics["IC"])
    rank_ic = float(metrics["Rank IC"])
    return ann * 1000 + ir * 50 + mdd * 20 + ic * 200 + rank_ic * 100


def _suffix_path(path: Path, result_suffix: str) -> Path:
    if not result_suffix:
        return path
    return path.with_name(f"{path.stem}_{result_suffix}{path.suffix}")


def _trial_suffix(trial_name: str, result_suffix: str) -> str:
    if not result_suffix:
        return trial_name
    return f"{trial_name}_{result_suffix}"


def _persist(rows: list[dict], path: Path):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    if "score" not in df.columns:
        df["score"] = pd.NA
    df.sort_values(by=["status", "score"], ascending=[True, False], na_position="last").to_csv(path, index=False)


def _load_existing_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if df.empty:
        return {}

    row_map = {}
    for row in df.to_dict(orient="records"):
        row_map[row["model_trial"]] = row
    return row_map


def _parse_optional_literal(value: str):
    if value in ("None", "", None):
        return None
    return literal_eval(value)


def _write_best_validation(
    best_row: dict,
    *,
    output_path: Path,
    window_key: str,
    seed: int,
    deterministic_runtime: bool,
):
    window = WINDOWS[window_key]
    lines = [
        f"window_key={window_key}",
        f"train={window['train']}",
        f"valid={window['valid']}",
        f"test={window['test']}",
        f"seed={seed}",
        f"deterministic_runtime={deterministic_runtime}",
        f"selection_target={OBJECTIVE_FORMULA}",
        f"dataset_kwargs_extra={DATASET_KWARGS}",
        f"validation_rolling_task_limit={VALIDATION_ROLLING_TASK_LIMIT}",
        f"fixed_strategy_trial={FIXED_STRATEGY['trial_name']}",
        f"fixed_strategy_class={FIXED_STRATEGY['strategy_class']}",
        f"fixed_strategy_module_path={FIXED_STRATEGY['strategy_module_path']}",
        f"fixed_strategy_n_drop={FIXED_STRATEGY['n_drop']}",
        f"fixed_strategy_risk_degree={FIXED_STRATEGY['risk_degree']}",
        f"fixed_strategy_kwargs_extra={FIXED_STRATEGY['kwargs_extra']}",
        f"model_trial={best_row['model_trial']}",
        f"model_key={best_row['model_key']}",
        f"step={best_row['step']}",
        f"handler_kwargs_extra={best_row['handler_kwargs_extra']}",
        f"model_kwargs_override={best_row['model_kwargs_override']}",
        f"validation_IC={best_row['IC']}",
        f"validation_Rank_IC={best_row['Rank IC']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"validation_score={best_row['score']}",
        f"validation_result_path={best_row['result_path']}",
        f"validation_cache_path={best_row['cache_path']}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_test(
    best_row: dict,
    test_result: dict,
    *,
    output_path: Path,
    window_key: str,
    seed: int,
    deterministic_runtime: bool,
):
    window = WINDOWS[window_key]
    lines = [
        f"window_key={window_key}",
        f"train={window['train']}",
        f"valid={window['valid']}",
        f"test={window['test']}",
        f"seed={seed}",
        f"deterministic_runtime={deterministic_runtime}",
        f"selection_source=validation_only",
        f"selection_target={OBJECTIVE_FORMULA}",
        f"dataset_kwargs_extra={DATASET_KWARGS}",
        f"validation_rolling_task_limit={VALIDATION_ROLLING_TASK_LIMIT}",
        f"final_test_rolling_task_limit=None",
        f"fixed_strategy_trial={FIXED_STRATEGY['trial_name']}",
        f"fixed_strategy_class={FIXED_STRATEGY['strategy_class']}",
        f"fixed_strategy_module_path={FIXED_STRATEGY['strategy_module_path']}",
        f"fixed_strategy_n_drop={FIXED_STRATEGY['n_drop']}",
        f"fixed_strategy_risk_degree={FIXED_STRATEGY['risk_degree']}",
        f"fixed_strategy_kwargs_extra={FIXED_STRATEGY['kwargs_extra']}",
        f"model_trial={best_row['model_trial']}",
        f"model_key={best_row['model_key']}",
        f"step={best_row['step']}",
        f"handler_kwargs_extra={best_row['handler_kwargs_extra']}",
        f"model_kwargs_override={best_row['model_kwargs_override']}",
        f"validation_score={best_row['score']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"test_result_path={test_result['result_path']}",
        f"test_cache_path={test_result['cache_path']}",
        f"test_recorder_id={test_result['recorder_id']}",
        f"test_IC={test_result['IC']}",
        f"test_Rank_IC={test_result['Rank IC']}",
        f"test_with_cost_ann_return={test_result['with_cost_ann_return']}",
        f"test_with_cost_ir={test_result['with_cost_ir']}",
        f"test_with_cost_mdd={test_result['with_cost_mdd']}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selected_trials(trial_filter: str, max_trials: int | None) -> list[dict]:
    trials = MODEL_TRIALS
    if trial_filter:
        lowered = trial_filter.lower()
        trials = [trial for trial in trials if lowered in trial["trial_name"].lower()]
    if max_trials is not None:
        trials = trials[:max_trials]
    return trials


def main(
    skip_final_if_exists: bool = True,
    *,
    seed: int = 42,
    gpu_slots: str = GPU_SLOTS,
    result_suffix: str = "",
    deterministic_runtime: bool = True,
    window_key: str = WINDOW_KEY,
    run_final_test: bool = True,
    trial_filter: str = "",
    max_trials: int | None = None,
):
    validation_result_path = _suffix_path(VALIDATION_RESULT_PATH, result_suffix)
    validation_best_path = _suffix_path(VALIDATION_BEST_PATH, result_suffix)
    final_test_path = _suffix_path(FINAL_TEST_PATH, result_suffix)
    active_trials = _selected_trials(trial_filter, max_trials)

    if not active_trials:
        raise RuntimeError("no training trials selected")

    print(
        f"training-side TRA tuning seed={seed} window_key={window_key} gpu_slots={gpu_slots} "
        f"deterministic_runtime={deterministic_runtime} result_suffix={result_suffix or '<default>'} "
        f"#model_trials={len(active_trials)} validation_rolling_task_limit={VALIDATION_ROLLING_TASK_LIMIT} "
        f"dataset_kwargs={DATASET_KWARGS} "
        f"fixed_strategy={FIXED_STRATEGY['trial_name']}"
    )

    row_map = _load_existing_rows(validation_result_path)
    if row_map:
        completed_count = sum(1 for row in row_map.values() if row.get("status") == "success")
        print(f"resume existing validation file: {validation_result_path} success_rows={completed_count}/{len(row_map)}")

    for model_trial in active_trials:
        existing_row = row_map.get(model_trial["trial_name"])
        if existing_row and existing_row.get("status") == "success":
            print(f"skip completed training trial {model_trial['trial_name']}")
            continue

        print(
            f"\n=== validation training trial {model_trial['trial_name']}: "
            f"model={model_trial['model_key']} step={model_trial['step']} ==="
        )
        model_kwargs_override = copy.deepcopy(model_trial["model_kwargs"])
        model_kwargs_override["seed"] = seed

        row = {
            "model_trial": model_trial["trial_name"],
            "model_key": model_trial["model_key"],
            "step": model_trial["step"],
            "handler_kwargs_extra": str(model_trial["handler_kwargs_extra"]),
            "model_kwargs_override": str(model_kwargs_override),
            "strategy_trial": FIXED_STRATEGY["trial_name"],
            "strategy_class": FIXED_STRATEGY["strategy_class"],
            "strategy_module_path": FIXED_STRATEGY["strategy_module_path"],
            "n_drop": FIXED_STRATEGY["n_drop"],
            "risk_degree": FIXED_STRATEGY["risk_degree"],
            "kwargs_extra": str(FIXED_STRATEGY["kwargs_extra"]),
            "seed": seed,
            "deterministic_runtime": deterministic_runtime,
            "status": "running",
        }
        try:
            result = run(
                step=model_trial["step"],
                topk=TOPK,
                n_drop=FIXED_STRATEGY["n_drop"],
                window_key=window_key,
                model_key=model_trial["model_key"],
                exp_suffix=_trial_suffix(model_trial["trial_name"], result_suffix),
                handler_class="Alpha360",
                handler_module_path="qlib.contrib.data.handler",
                account=ACCOUNT,
                risk_degree=FIXED_STRATEGY["risk_degree"],
                provider_uri=PROVIDER_URI,
                benchmark=BENCHMARK,
                instruments=INSTRUMENTS,
                model_kwargs_override=model_kwargs_override,
                handler_kwargs_extra=model_trial["handler_kwargs_extra"],
                dataset_kwargs_extra=DATASET_KWARGS,
                strategy_class=FIXED_STRATEGY["strategy_class"],
                strategy_module_path=FIXED_STRATEGY["strategy_module_path"],
                strategy_kwargs_extra=FIXED_STRATEGY["kwargs_extra"],
                gpu_slots=gpu_slots,
                eval_segment="valid",
                backtest_segment="valid",
                rolling_task_limit=VALIDATION_ROLLING_TASK_LIMIT,
                deterministic_runtime=deterministic_runtime,
            )
            row.update(result)
            row["score"] = _objective(result)
            row["status"] = "success"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)

        row_map[model_trial["trial_name"]] = row
        _persist(list(row_map.values()), validation_result_path)

    rows = [row for row in row_map.values() if row["model_trial"] in {trial["trial_name"] for trial in active_trials}]
    if not rows:
        raise RuntimeError("no validation rows produced")

    success_rows = [row for row in rows if row["status"] == "success"]
    if not success_rows:
        raise RuntimeError("all validation trials failed")

    best_row = max(success_rows, key=lambda item: item["score"])
    _write_best_validation(
        best_row,
        output_path=validation_best_path,
        window_key=window_key,
        seed=seed,
        deterministic_runtime=deterministic_runtime,
    )

    if not run_final_test:
        print("skip final test by request; validation winner has been written")
        return

    if skip_final_if_exists and final_test_path.exists():
        print(f"final test summary already exists at {final_test_path}, skip rerunning final test")
        return

    print(f"\n=== final test with validation-selected training config: {best_row['model_trial']} ===")
    test_result = run(
        step=int(best_row["step"]),
        topk=TOPK,
        n_drop=FIXED_STRATEGY["n_drop"],
        window_key=window_key,
        model_key=best_row["model_key"],
        exp_suffix=_trial_suffix(f"{best_row['model_trial']}_final_test", result_suffix),
        handler_class="Alpha360",
        handler_module_path="qlib.contrib.data.handler",
        account=ACCOUNT,
        risk_degree=FIXED_STRATEGY["risk_degree"],
        provider_uri=PROVIDER_URI,
        benchmark=BENCHMARK,
        instruments=INSTRUMENTS,
        model_kwargs_override=_parse_optional_literal(best_row["model_kwargs_override"]),
        handler_kwargs_extra=_parse_optional_literal(best_row["handler_kwargs_extra"]),
        dataset_kwargs_extra=DATASET_KWARGS,
        strategy_class=FIXED_STRATEGY["strategy_class"],
        strategy_module_path=FIXED_STRATEGY["strategy_module_path"],
        strategy_kwargs_extra=FIXED_STRATEGY["kwargs_extra"],
        gpu_slots=gpu_slots,
        eval_segment="test",
        backtest_segment="test",
        rolling_task_limit=None,
        deterministic_runtime=deterministic_runtime,
    )
    _write_final_test(
        best_row,
        test_result,
        output_path=final_test_path,
        window_key=window_key,
        seed=seed,
        deterministic_runtime=deterministic_runtime,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training-side TRA Alpha360 tuning: fix h5 label and baseline strategy, search TRA training hyperparameters."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-slots", default=GPU_SLOTS)
    parser.add_argument("--result-suffix", default="")
    parser.add_argument("--window-key", default=WINDOW_KEY, choices=sorted(WINDOWS))
    parser.add_argument("--trial-filter", default="", help="Run only trials whose names contain this substring.")
    parser.add_argument("--max-trials", type=int, help="Run only the first N selected trials.")
    parser.add_argument("--rerun-final", action="store_true", help="Rerun final test even if a final summary already exists.")
    parser.add_argument("--skip-final", action="store_true", help="Run validation tuning only and skip test evaluation.")
    parser.add_argument("--legacy-runtime", action="store_true", help="Disable deterministic TRA runtime controls.")
    args = parser.parse_args()

    main(
        skip_final_if_exists=not args.rerun_final,
        seed=args.seed,
        gpu_slots=args.gpu_slots,
        result_suffix=args.result_suffix,
        deterministic_runtime=not args.legacy_runtime,
        window_key=args.window_key,
        run_final_test=not args.skip_final,
        trial_filter=args.trial_filter,
        max_trials=args.max_trials,
    )