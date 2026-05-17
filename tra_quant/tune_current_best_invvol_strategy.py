from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord

from rolling_tra_alpha360_latest import PROVIDER_URI, _metric_value, build_base_task


BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "current_best_baseline_manifest.json"
VALIDATION_RESULTS_PATH = BASE_DIR / "alpha360_tra_fundamental_xgb_invvol_strategy_validation_results.csv"
VALIDATION_BEST_PATH = BASE_DIR / "alpha360_tra_fundamental_xgb_invvol_strategy_validation_best.txt"
FINAL_TEST_PATH = BASE_DIR / "alpha360_tra_fundamental_xgb_invvol_strategy_final_test.txt"


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


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _load_signal(cache_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = pd.read_pickle(cache_path)
    pred = cache["pred"].sort_index()
    label = cache["label"].sort_index()
    common_index = pred.index.intersection(label.index)
    return pred.loc[common_index].sort_index(), label.loc[common_index].sort_index()


def _build_trials(manifest: dict) -> list[dict]:
    stage1 = manifest["stage1_incumbent"]
    current_best = manifest["current_best_baseline"]
    common_kwargs = {
        "n_drop": int(stage1["n_drop"]),
        "risk_degree": float(stage1["risk_degree"]),
        "hold_thresh": int(stage1["kwargs_extra"]["hold_thresh"]),
        "score_margin": float(stage1["kwargs_extra"]["score_margin"]),
        "slot_budget_ratio": float(stage1["kwargs_extra"]["slot_budget_ratio"]),
        "universe": "csi300",
    }
    return [
        {
            "trial_name": "baseline_current_strategy",
            "strategy_class": stage1["strategy_class"],
            "strategy_module_path": stage1["strategy_module_path"],
            "n_drop": common_kwargs["n_drop"],
            "risk_degree": common_kwargs["risk_degree"],
            "kwargs_extra": {
                "score_margin": common_kwargs["score_margin"],
                "hold_thresh": common_kwargs["hold_thresh"],
                "slot_budget_ratio": common_kwargs["slot_budget_ratio"],
            },
            "baseline_seq_weight": float(current_best["seq_weight"]),
            "baseline_tab_weight": float(current_best["tab_weight"]),
        },
        {
            "trial_name": "invvol_v10_p10",
            "strategy_class": "InverseVolPracticalTopkDropoutStrategy",
            "strategy_module_path": "qlib.contrib.strategy.custom_signal_strategy",
            "n_drop": common_kwargs["n_drop"],
            "risk_degree": common_kwargs["risk_degree"],
            "kwargs_extra": {
                "score_margin": common_kwargs["score_margin"],
                "hold_thresh": common_kwargs["hold_thresh"],
                "slot_budget_ratio": common_kwargs["slot_budget_ratio"],
                "universe": common_kwargs["universe"],
                "vol_window": 10,
                "vol_power": 1.0,
            },
        },
        {
            "trial_name": "invvol_v20_p10",
            "strategy_class": "InverseVolPracticalTopkDropoutStrategy",
            "strategy_module_path": "qlib.contrib.strategy.custom_signal_strategy",
            "n_drop": common_kwargs["n_drop"],
            "risk_degree": common_kwargs["risk_degree"],
            "kwargs_extra": {
                "score_margin": common_kwargs["score_margin"],
                "hold_thresh": common_kwargs["hold_thresh"],
                "slot_budget_ratio": common_kwargs["slot_budget_ratio"],
                "universe": common_kwargs["universe"],
                "vol_window": 20,
                "vol_power": 1.0,
            },
        },
        {
            "trial_name": "invvol_v40_p10",
            "strategy_class": "InverseVolPracticalTopkDropoutStrategy",
            "strategy_module_path": "qlib.contrib.strategy.custom_signal_strategy",
            "n_drop": common_kwargs["n_drop"],
            "risk_degree": common_kwargs["risk_degree"],
            "kwargs_extra": {
                "score_margin": common_kwargs["score_margin"],
                "hold_thresh": common_kwargs["hold_thresh"],
                "slot_budget_ratio": common_kwargs["slot_budget_ratio"],
                "universe": common_kwargs["universe"],
                "vol_window": 40,
                "vol_power": 1.0,
            },
        },
        {
            "trial_name": "invvol_v20_p05",
            "strategy_class": "InverseVolPracticalTopkDropoutStrategy",
            "strategy_module_path": "qlib.contrib.strategy.custom_signal_strategy",
            "n_drop": common_kwargs["n_drop"],
            "risk_degree": common_kwargs["risk_degree"],
            "kwargs_extra": {
                "score_margin": common_kwargs["score_margin"],
                "hold_thresh": common_kwargs["hold_thresh"],
                "slot_budget_ratio": common_kwargs["slot_budget_ratio"],
                "universe": common_kwargs["universe"],
                "vol_window": 20,
                "vol_power": 0.5,
            },
        },
    ]


def _evaluate_trial(manifest: dict, trial: dict, pred: pd.DataFrame, label: pd.DataFrame, eval_segment: str) -> dict:
    stage1 = manifest["stage1_incumbent"]

    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)
    task = build_base_task(
        topk=5,
        n_drop=int(trial["n_drop"]),
        window_key=manifest["data_split"]["window_key"],
        model_key=stage1["model_key"],
        handler_class="Alpha360",
        handler_module_path="qlib.contrib.data.handler",
        account=150000,
        risk_degree=float(trial["risk_degree"]),
        benchmark="SH000300",
        instruments="csi300",
        handler_kwargs_extra=stage1.get("handler_kwargs_extra"),
        strategy_class=trial["strategy_class"],
        strategy_module_path=trial["strategy_module_path"],
        strategy_kwargs_extra=trial["kwargs_extra"],
        eval_segment=eval_segment,
        backtest_segment=eval_segment,
    )
    port_record = task["record"][2]

    recorder_name = f"current_best_invvol_{eval_segment}_{trial['trial_name']}"
    with R.start(experiment_name=f"current_best_invvol_{eval_segment}_eval", recorder_name=recorder_name):
        rec = R.get_recorder()
        rec.save_objects(**{"pred.pkl": pred, "label.pkl": label})
        SigAnaRecord(recorder=rec, ana_long_short=False, ann_scaler=252).generate()
        PortAnaRecord(recorder=rec, **port_record["kwargs"]).generate()
        metrics = rec.list_metrics()

    result = {
        "recorder_id": rec.id,
        "IC": metrics.get("IC"),
        "Rank IC": metrics.get("Rank IC"),
        "with_cost_ann_return": _metric_value(metrics, rec.id, "1day.excess_return_with_cost.annualized_return"),
        "with_cost_ir": _metric_value(metrics, rec.id, "1day.excess_return_with_cost.information_ratio"),
        "with_cost_mdd": _metric_value(metrics, rec.id, "1day.excess_return_with_cost.max_drawdown"),
    }
    result["score"] = _objective(result)
    return result


def _write_validation_best(best_row: dict, manifest: dict):
    current_best = manifest["current_best_baseline"]
    lines = [
        f"selection_scope=validation_only_strategy_layer",
        f"base_model_trial={current_best['base_model_trial']}",
        f"base_strategy_trial={current_best['base_strategy_trial']}",
        f"tabular_trial={current_best['tabular_trial']}",
        f"fusion_mode={current_best['fusion_mode']}",
        f"seq_weight={current_best['seq_weight']}",
        f"tab_weight={current_best['tab_weight']}",
        f"selected_trial_name={best_row['trial_name']}",
        f"strategy_class={best_row['strategy_class']}",
        f"strategy_module_path={best_row['strategy_module_path']}",
        f"n_drop={best_row['n_drop']}",
        f"risk_degree={best_row['risk_degree']}",
        f"kwargs_extra={best_row['kwargs_extra']}",
        f"validation_IC={best_row['IC']}",
        f"validation_Rank_IC={best_row['Rank IC']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"validation_score={best_row['score']}",
        f"validation_recorder_id={best_row['recorder_id']}",
    ]
    VALIDATION_BEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_test(best_row: dict, test_result: dict, manifest: dict):
    baseline = manifest["current_best_baseline"]["final_test"]
    lines = [
        f"selection_scope=validation_only_strategy_layer",
        f"selected_trial_name={best_row['trial_name']}",
        f"strategy_class={best_row['strategy_class']}",
        f"strategy_module_path={best_row['strategy_module_path']}",
        f"n_drop={best_row['n_drop']}",
        f"risk_degree={best_row['risk_degree']}",
        f"kwargs_extra={best_row['kwargs_extra']}",
        f"validation_score={best_row['score']}",
        f"test_recorder_id={test_result['recorder_id']}",
        f"test_IC={test_result['IC']}",
        f"test_Rank_IC={test_result['Rank IC']}",
        f"test_with_cost_ann_return={test_result['with_cost_ann_return']}",
        f"test_with_cost_ir={test_result['with_cost_ir']}",
        f"test_with_cost_mdd={test_result['with_cost_mdd']}",
        f"baseline_test_with_cost_ann_return={baseline['with_cost_ann_return']}",
        f"baseline_test_with_cost_ir={baseline['with_cost_ir']}",
        f"baseline_test_with_cost_mdd={baseline['with_cost_mdd']}",
        f"delta_ann={float(test_result['with_cost_ann_return']) - float(baseline['with_cost_ann_return'])}",
        f"delta_ir={float(test_result['with_cost_ir']) - float(baseline['with_cost_ir'])}",
        f"delta_mdd={float(test_result['with_cost_mdd']) - float(baseline['with_cost_mdd'])}",
    ]
    FINAL_TEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    manifest = _load_manifest()
    current_best = manifest["current_best_baseline"]
    trials = _build_trials(manifest)

    valid_pred, valid_label = _load_signal(current_best["validation"]["cache_path"])
    test_pred, test_label = _load_signal(current_best["final_test"]["cache_path"])

    rows = _load_existing_rows(VALIDATION_RESULTS_PATH)
    done = {row["trial_name"] for row in rows if row.get("status") == "success"}

    for idx, trial in enumerate(trials, start=1):
        if trial["trial_name"] in done:
            print(f"[skip {idx}/{len(trials)}] {trial['trial_name']} already completed", flush=True)
            continue

        print(f"[run {idx}/{len(trials)}] {trial['trial_name']}", flush=True)
        row = {
            "trial_name": trial["trial_name"],
            "strategy_class": trial["strategy_class"],
            "strategy_module_path": trial["strategy_module_path"],
            "n_drop": trial["n_drop"],
            "risk_degree": trial["risk_degree"],
            "kwargs_extra": str(trial["kwargs_extra"]),
            "status": "running",
        }
        try:
            result = _evaluate_trial(manifest, trial, valid_pred, valid_label, eval_segment="valid")
            row.update(
                {
                    "status": "success",
                    "IC": result["IC"],
                    "Rank IC": result["Rank IC"],
                    "with_cost_ann_return": result["with_cost_ann_return"],
                    "with_cost_ir": result["with_cost_ir"],
                    "with_cost_mdd": result["with_cost_mdd"],
                    "score": result["score"],
                    "recorder_id": result["recorder_id"],
                }
            )
            print(
                f"[done {idx}/{len(trials)}] {trial['trial_name']} ann={result['with_cost_ann_return']:.6f} ir={result['with_cost_ir']:.6f}",
                flush=True,
            )
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)[:500]
            print(f"[fail {idx}/{len(trials)}] {trial['trial_name']} error={row['error']}", flush=True)
        rows = [existing for existing in rows if existing.get("trial_name") != trial["trial_name"]] + [row]
        _persist(rows, VALIDATION_RESULTS_PATH)

    success_rows = [row for row in rows if row["status"] == "success"]
    if not success_rows:
        raise RuntimeError("no successful validation rows for invvol strategy experiment")

    best_row = max(success_rows, key=lambda item: item["score"])
    _write_validation_best(best_row, manifest)
    best_trial = next(trial for trial in trials if trial["trial_name"] == best_row["trial_name"])
    test_result = _evaluate_trial(manifest, best_trial, test_pred, test_label, eval_segment="test")
    _write_final_test(best_row, test_result, manifest)


if __name__ == "__main__":
    main()