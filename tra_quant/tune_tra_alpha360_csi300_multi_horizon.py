from __future__ import annotations

import argparse
import copy
from ast import literal_eval
from pathlib import Path

import pandas as pd

from rolling_tra_alpha360_latest import MODEL_PRESETS, PROVIDER_URI, WINDOWS, run


BASE_DIR = Path(__file__).resolve().parent
VALIDATION_RESULT_PATH = BASE_DIR / "tra_csi300_multi_horizon_validation_results.csv"
VALIDATION_BEST_PATH = BASE_DIR / "tra_csi300_multi_horizon_validation_best.txt"
FINAL_TEST_PATH = BASE_DIR / "tra_csi300_multi_horizon_final_test.txt"
STRICT_VALIDATION_BEST_PATH = BASE_DIR / "tra_strict_validation_best.txt"
STRICT_FINAL_TEST_PATH = BASE_DIR / "tra_strict_final_test.txt"

ACCOUNT = 150000
TOPK = 5
WINDOW_KEY = "w1"
BENCHMARK = "SH000300"
INSTRUMENTS = "csi300"
GPU_SLOTS = "0,1"

DEFAULT_LABEL = ["Ref($close, -4) / Ref($close, -1) - 1"]
H5_LABEL = ["Ref($close, -6) / Ref($close, -1) - 1"]
H10_LABEL = ["Ref($close, -11) / Ref($close, -1) - 1"]

STRATEGY = {
    "strategy_class": "PracticalTopkDropoutStrategy",
    "strategy_module_path": "qlib.contrib.strategy.custom_signal_strategy",
    "n_drop": 1,
    "risk_degree": 0.70,
    "kwargs_extra": {"score_margin": 0.005, "hold_thresh": 5, "slot_budget_ratio": 1.0},
}


def _parse_scalar(value: str):
    try:
        return literal_eval(value)
    except Exception:
        return value


def _parse_summary_file(path: Path) -> dict:
    data: dict[str, object] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = _parse_scalar(value)
    return data


def _register_model_presets():
    if "mh_tra_lstm_base" not in MODEL_PRESETS:
        preset = copy.deepcopy(MODEL_PRESETS["tra_lstm_base"])
        preset["class"] = "MultiHorizonTRAModel"
        preset["module_path"] = "qlib.contrib.model.pytorch_multi_horizon_tra"
        preset["kwargs"]["output_dim"] = 2
        preset["kwargs"]["target_weights"] = [0.5, 0.5]
        preset["kwargs"]["target_names"] = ["h3", "h5"]
        MODEL_PRESETS["mh_tra_lstm_base"] = preset


TRIALS = [
    {
        "trial_name": "csi300_mhtra_lstm_s240_h35_eq",
        "family": "multi_horizon",
        "model_key": "mh_tra_lstm_base",
        "step": 240,
        "handler_kwargs_extra": {"label": DEFAULT_LABEL + H5_LABEL},
        "model_kwargs_override": {"output_dim": 2, "target_weights": [0.5, 0.5], "target_names": ["h3", "h5"]},
    },
    {
        "trial_name": "csi300_mhtra_lstm_s240_h3510_front",
        "family": "multi_horizon",
        "model_key": "mh_tra_lstm_base",
        "step": 240,
        "handler_kwargs_extra": {"label": DEFAULT_LABEL + H10_LABEL},
        "model_kwargs_override": {"output_dim": 2, "target_weights": [0.35, 0.65], "target_names": ["h3", "h10"]},
    },
]


def _objective(metrics: dict) -> float:
    ann = float(metrics["with_cost_ann_return"])
    ir = float(metrics["with_cost_ir"])
    mdd = float(metrics["with_cost_mdd"])
    return ann * 1000 + ir * 50 + mdd * 20


def _persist(rows: list[dict], path: Path):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    if "score" not in df.columns:
        df["score"] = pd.NA
    df.sort_values(by=["status", "score"], ascending=[True, False], na_position="last").to_csv(path, index=False)


def _load_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    return df.to_dict("records")


def _incumbent_row() -> dict:
    summary = _parse_summary_file(STRICT_VALIDATION_BEST_PATH)
    return {
        "trial_name": str(summary["model_trial"]),
        "family": "incumbent_single_horizon",
        "model_key": str(summary["model_key"]),
        "step": int(summary["step"]),
        "handler_kwargs_extra": str(summary["handler_kwargs_extra"]),
        "model_kwargs_override": str(None),
        "status": "success",
        "IC": float(summary["validation_IC"]),
        "Rank IC": float(summary["validation_Rank_IC"]),
        "with_cost_ann_return": float(summary["validation_with_cost_ann_return"]),
        "with_cost_ir": float(summary["validation_with_cost_ir"]),
        "with_cost_mdd": float(summary["validation_with_cost_mdd"]),
        "score": float(summary["validation_score"]),
        "result_path": str(summary["validation_result_path"]),
        "cache_path": str(summary["validation_cache_path"]),
        "recorder_id": pd.NA,
    }


def _trial_matches(trial_name: str, trial_pattern: str | None) -> bool:
    if not trial_pattern:
        return True
    import re

    return re.search(trial_pattern, trial_name) is not None


def _run_trial(trial: dict, eval_segment: str) -> dict:
    result = run(
        step=trial["step"],
        topk=TOPK,
        n_drop=STRATEGY["n_drop"],
        window_key=WINDOW_KEY,
        model_key=trial["model_key"],
        exp_suffix=trial["trial_name"],
        handler_class="Alpha360",
        handler_module_path="qlib.contrib.data.handler",
        account=ACCOUNT,
        risk_degree=STRATEGY["risk_degree"],
        provider_uri=PROVIDER_URI,
        benchmark=BENCHMARK,
        instruments=INSTRUMENTS,
        model_kwargs_override=trial["model_kwargs_override"],
        handler_kwargs_extra=trial["handler_kwargs_extra"],
        strategy_class=STRATEGY["strategy_class"],
        strategy_module_path=STRATEGY["strategy_module_path"],
        strategy_kwargs_extra=STRATEGY["kwargs_extra"],
        gpu_slots=GPU_SLOTS,
        eval_segment=eval_segment,
        backtest_segment=eval_segment,
    )
    result["score"] = _objective(result)
    return result


def _write_validation_best(best_row: dict):
    window = WINDOWS[WINDOW_KEY]
    strict_best = _parse_summary_file(STRICT_VALIDATION_BEST_PATH)
    lines = [
        f"selection_scope=csi300_multi_horizon_tra_validation_only",
        f"window_key={WINDOW_KEY}",
        f"train={window['train']}",
        f"valid={window['valid']}",
        f"test={window['test']}",
        f"benchmark={BENCHMARK}",
        f"instruments={INSTRUMENTS}",
        f"trial_name={best_row['trial_name']}",
        f"family={best_row['family']}",
        f"model_key={best_row['model_key']}",
        f"handler_kwargs_extra={best_row['handler_kwargs_extra']}",
        f"model_kwargs_override={best_row['model_kwargs_override']}",
        f"strategy_class={STRATEGY['strategy_class']}",
        f"strategy_kwargs_extra={STRATEGY['kwargs_extra']}",
        f"validation_IC={best_row['IC']}",
        f"validation_Rank_IC={best_row['Rank IC']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"validation_score={best_row['score']}",
        f"validation_result_path={best_row['result_path']}",
        f"validation_cache_path={best_row['cache_path']}",
        f"validation_recorder_id={best_row['recorder_id']}",
        f"strict_incumbent_validation_ann={strict_best['validation_with_cost_ann_return']}",
        f"strict_incumbent_validation_ir={strict_best['validation_with_cost_ir']}",
        f"strict_incumbent_validation_mdd={strict_best['validation_with_cost_mdd']}",
        f"strict_incumbent_validation_score={strict_best['validation_score']}",
    ]
    VALIDATION_BEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_test(best_row: dict, test_result: dict):
    window = WINDOWS[WINDOW_KEY]
    strict_final = _parse_summary_file(STRICT_FINAL_TEST_PATH)
    lines = [
        f"selection_scope=csi300_multi_horizon_tra_validation_only",
        f"window_key={WINDOW_KEY}",
        f"train={window['train']}",
        f"valid={window['valid']}",
        f"test={window['test']}",
        f"benchmark={BENCHMARK}",
        f"instruments={INSTRUMENTS}",
        f"trial_name={best_row['trial_name']}",
        f"family={best_row['family']}",
        f"model_key={best_row['model_key']}",
        f"handler_kwargs_extra={best_row['handler_kwargs_extra']}",
        f"model_kwargs_override={best_row['model_kwargs_override']}",
        f"strategy_class={STRATEGY['strategy_class']}",
        f"strategy_kwargs_extra={STRATEGY['kwargs_extra']}",
        f"validation_score={best_row['score']}",
        f"test_IC={test_result['IC']}",
        f"test_Rank_IC={test_result['Rank IC']}",
        f"test_with_cost_ann_return={test_result['with_cost_ann_return']}",
        f"test_with_cost_ir={test_result['with_cost_ir']}",
        f"test_with_cost_mdd={test_result['with_cost_mdd']}",
        f"test_result_path={test_result['result_path']}",
        f"test_cache_path={test_result['cache_path']}",
        f"test_recorder_id={test_result['recorder_id']}",
        f"strict_incumbent_test_ann={strict_final['test_with_cost_ann_return']}",
        f"strict_incumbent_test_ir={strict_final['test_with_cost_ir']}",
        f"strict_incumbent_test_mdd={strict_final['test_with_cost_mdd']}",
    ]
    FINAL_TEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(trial_pattern: str | None = None, max_trials: int | None = None, skip_final_test: bool = False):
    _register_model_presets()
    rows = _load_existing_rows(VALIDATION_RESULT_PATH)
    rows = [row for row in rows if row.get("trial_name") != _incumbent_row()["trial_name"]]
    rows.append(_incumbent_row())
    done = {row["trial_name"] for row in rows if row.get("status") == "success"}

    pending_trials = [trial for trial in TRIALS if _trial_matches(trial["trial_name"], trial_pattern)]
    if max_trials is not None:
        pending_trials = pending_trials[:max_trials]

    for idx, trial in enumerate(pending_trials, start=1):
        if trial["trial_name"] in done:
            print(f"[skip {idx}/{len(pending_trials)}] {trial['trial_name']} already completed", flush=True)
            continue

        row = {
            "trial_name": trial["trial_name"],
            "family": trial["family"],
            "model_key": trial["model_key"],
            "step": trial["step"],
            "handler_kwargs_extra": str(trial["handler_kwargs_extra"]),
            "model_kwargs_override": str(trial["model_kwargs_override"]),
            "status": "running",
        }
        print(f"[run {idx}/{len(pending_trials)}] {trial['trial_name']}", flush=True)
        try:
            result = _run_trial(trial, eval_segment="valid")
            row.update(
                {
                    "status": "success",
                    "IC": result["IC"],
                    "Rank IC": result["Rank IC"],
                    "with_cost_ann_return": result["with_cost_ann_return"],
                    "with_cost_ir": result["with_cost_ir"],
                    "with_cost_mdd": result["with_cost_mdd"],
                    "score": result["score"],
                    "result_path": result["result_path"],
                    "cache_path": result["cache_path"],
                    "recorder_id": result["recorder_id"],
                }
            )
            print(
                f"[done {idx}/{len(pending_trials)}] {trial['trial_name']} ann={result['with_cost_ann_return']:.6f} "
                f"ir={result['with_cost_ir']:.6f} mdd={result['with_cost_mdd']:.6f}",
                flush=True,
            )
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)[:500]
            print(f"[fail {idx}/{len(pending_trials)}] {trial['trial_name']} error={row['error']}", flush=True)
        rows = [existing for existing in rows if existing.get("trial_name") != trial["trial_name"] and existing.get("family") != "incumbent_single_horizon"] + [_incumbent_row(), row]
        _persist(rows, VALIDATION_RESULT_PATH)

    success_rows = [row for row in rows if row.get("status") == "success"]
    if not success_rows:
        raise RuntimeError("no successful validation rows for csi300 multi-horizon TRA experiment")

    best_row = max(success_rows, key=lambda item: float(item["score"]))
    _write_validation_best(best_row)
    print(
        f"[best-valid] {best_row['trial_name']} ann={float(best_row['with_cost_ann_return']):.6f} "
        f"ir={float(best_row['with_cost_ir']):.6f} score={float(best_row['score']):.6f}",
        flush=True,
    )

    if skip_final_test or best_row["family"] == "incumbent_single_horizon":
        return

    best_trial = next(trial for trial in TRIALS if trial["trial_name"] == best_row["trial_name"])
    test_result = _run_trial(best_trial, eval_segment="test")
    _write_final_test(best_row, test_result)
    print(
        f"[final-test] {best_row['trial_name']} ann={float(test_result['with_cost_ann_return']):.6f} "
        f"ir={float(test_result['with_cost_ir']):.6f} mdd={float(test_result['with_cost_mdd']):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run csi300 Alpha360 + TRA multi-horizon experiments.")
    parser.add_argument("--trial-pattern", default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--skip-final-test", action="store_true")
    args = parser.parse_args()
    main(trial_pattern=args.trial_pattern, max_trials=args.max_trials, skip_final_test=args.skip_final_test)