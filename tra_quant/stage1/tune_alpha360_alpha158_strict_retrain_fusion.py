from __future__ import annotations

import argparse
from ast import literal_eval
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord

from rolling_alstm_alpha360_latest import PROVIDER_URI, WINDOWS, _metric_value, build_base_task
from rolling_lightgbm_latest import run as run_lightgbm


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "fusion_cache"

ALSTM_VALIDATION_BEST_PATH = BASE_DIR / "alstm_strict_validation_best.txt"
ALSTM_FINAL_TEST_PATH = BASE_DIR / "alstm_strict_final_test.txt"

VALIDATION_RESULTS_PATH = BASE_DIR / "alpha360_alpha158_strict_retrain_validation_results.csv"
VALIDATION_BEST_PATH = BASE_DIR / "alpha360_alpha158_strict_retrain_validation_best.txt"
FINAL_TEST_PATH = BASE_DIR / "alpha360_alpha158_strict_retrain_final_test.txt"

ACCOUNT = 150000
TOPK = 5
WINDOW_KEY = "w1"
BENCHMARK = "SH000300"
INSTRUMENTS = "csi300"

FUSION_MODES = ["rank_pct", "zscore"]
SEQ_WEIGHTS = [0.80, 0.85, 0.90, 0.95]

TABULAR_TRIALS = [
    {
        "trial_name": "rt01_alpha158_cons_plus_s120",
        "handler_class": "Alpha158",
        "handler_module_path": "qlib.contrib.data.handler",
        "model_key": "cons_plus",
        "step": 120,
    },
    {
        "trial_name": "rt02_techA_cons_s120",
        "handler_class": "TechAlpha158A",
        "handler_module_path": "qlib.contrib.data.custom_handler",
        "model_key": "cons",
        "step": 120,
    },
    {
        "trial_name": "rt03_techA_cons_plus_s120",
        "handler_class": "TechAlpha158A",
        "handler_module_path": "qlib.contrib.data.custom_handler",
        "model_key": "cons_plus",
        "step": 120,
    },
    {
        "trial_name": "rt04_techA4D_bal_s120",
        "handler_class": "TechAlpha158A4D",
        "handler_module_path": "qlib.contrib.data.custom_handler",
        "model_key": "bal",
        "step": 120,
    },
    {
        "trial_name": "rt05_techA3D_bal_safe_s120",
        "handler_class": "TechAlpha158A3D",
        "handler_module_path": "qlib.contrib.data.custom_handler",
        "model_key": "bal_safe",
        "step": 120,
    },
    {
        "trial_name": "rt06_techA3D_bal_v2_s120",
        "handler_class": "TechAlpha158A3D",
        "handler_module_path": "qlib.contrib.data.custom_handler",
        "model_key": "bal_v2",
        "step": 120,
    },
    {
        "trial_name": "rt07_techB2D_bal_s120",
        "handler_class": "TechAlpha158B2D",
        "handler_module_path": "qlib.contrib.data.custom_handler",
        "model_key": "bal",
        "step": 120,
    },
    {
        "trial_name": "rt08_techB_bal_safe_s120",
        "handler_class": "TechAlpha158B",
        "handler_module_path": "qlib.contrib.data.custom_handler",
        "model_key": "bal_safe",
        "step": 120,
    },
]


def _parse_scalar(value: str):
    try:
        return literal_eval(value)
    except Exception:
        return value


def _parse_summary_file(path: Path) -> dict:
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = _parse_scalar(value)
    return data


def _persist(rows: list[dict], path: Path):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if "score" not in df.columns:
        df["score"] = pd.NA
    df.sort_values(by=["status", "score"], ascending=[True, False], na_position="last").to_csv(path, index=False)


def _objective(metrics: dict) -> float:
    ann = float(metrics["with_cost_ann_return"])
    ir = float(metrics["with_cost_ir"])
    mdd = float(metrics["with_cost_mdd"])
    return ann * 1000 + ir * 50 + mdd * 20


def _ensure_score_df(df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        return df.to_frame("score")
    if "score" in df.columns:
        return df[["score"]].copy()
    return df.iloc[:, [0]].rename(columns={df.columns[0]: "score"})


def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    def _normalize(group: pd.DataFrame) -> pd.DataFrame:
        score = group["score"]
        std = score.std(ddof=0)
        if std is None or pd.isna(std) or std == 0:
            return pd.DataFrame({"score": pd.Series(0.0, index=group.index)})
        return pd.DataFrame({"score": (score - score.mean()) / std})

    return df.groupby(level="datetime", group_keys=False).apply(_normalize)


def _cross_sectional_rank_pct(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level="datetime", group_keys=False).apply(
        lambda group: pd.DataFrame({"score": group["score"].rank(pct=True)})
    )


def _normalize_scores(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "zscore":
        return _cross_sectional_zscore(df)
    if mode == "rank_pct":
        return _cross_sectional_rank_pct(df)
    raise ValueError(f"unsupported fusion mode: {mode}")


def _artifact_path_from_recorder(recorder_id: str, artifact_name: str) -> Path:
    matches = list((Path.cwd() / "mlruns").glob(f"*/{recorder_id}/artifacts/{artifact_name}"))
    if not matches:
        raise FileNotFoundError(f"artifact `{artifact_name}` not found for recorder `{recorder_id}`")
    return matches[0]


def _load_signal_from_cache(cache_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = pd.read_pickle(cache_path)
    pred = _ensure_score_df(cache["pred"]).sort_index()
    label = cache["label"].sort_index()
    common_index = pred.index.intersection(label.index)
    return pred.loc[common_index].sort_index(), label.loc[common_index].sort_index()


def _load_signal_from_recorder(recorder_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = _artifact_path_from_recorder(recorder_id, "pred.pkl")
    label_path = _artifact_path_from_recorder(recorder_id, "label.pkl")
    pred = _ensure_score_df(pd.read_pickle(pred_path)).sort_index()
    label = pd.read_pickle(label_path).sort_index()
    common_index = pred.index.intersection(label.index)
    return pred.loc[common_index].sort_index(), label.loc[common_index].sort_index()


def _align_signals(
    seq_pred: pd.DataFrame,
    seq_label: pd.DataFrame,
    tab_pred: pd.DataFrame,
    tab_label: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_index = seq_pred.index.intersection(tab_pred.index).intersection(seq_label.index).intersection(tab_label.index)
    common_index = common_index.sort_values()
    return (
        seq_pred.loc[common_index].sort_index(),
        tab_pred.loc[common_index].sort_index(),
        seq_label.loc[common_index].sort_index(),
    )


def _load_alstm_strategy_cfg(validation_summary: dict, eval_segment: str) -> dict:
    return {
        "handler_kwargs_extra": validation_summary["handler_kwargs_extra"],
        "strategy_class": validation_summary["strategy_class"],
        "strategy_module_path": validation_summary["strategy_module_path"],
        "kwargs_extra": validation_summary["kwargs_extra"],
        "n_drop": validation_summary["n_drop"],
        "risk_degree": validation_summary["risk_degree"],
        "eval_segment": eval_segment,
        "backtest_segment": eval_segment,
    }


def _evaluate_signal(
    recorder_name: str,
    pred: pd.DataFrame,
    label: pd.DataFrame,
    strategy_cfg: dict,
    experiment_name: str,
) -> dict:
    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)
    task = build_base_task(
        topk=TOPK,
        n_drop=int(strategy_cfg["n_drop"]),
        window_key=WINDOW_KEY,
        model_key="gru_base",
        handler_class="Alpha360",
        handler_module_path="qlib.contrib.data.handler",
        account=ACCOUNT,
        risk_degree=float(strategy_cfg["risk_degree"]),
        benchmark=BENCHMARK,
        instruments=INSTRUMENTS,
        handler_kwargs_extra=strategy_cfg["handler_kwargs_extra"],
        strategy_class=strategy_cfg["strategy_class"],
        strategy_module_path=strategy_cfg["strategy_module_path"],
        strategy_kwargs_extra=strategy_cfg["kwargs_extra"],
        eval_segment=strategy_cfg["eval_segment"],
        backtest_segment=strategy_cfg["backtest_segment"],
    )
    port_record = task["record"][2]

    with R.start(experiment_name=experiment_name, recorder_name=recorder_name):
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


def _write_fused_cache(cache_name: str, pred: pd.DataFrame, label: pd.DataFrame) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_name}.pkl"
    pd.to_pickle({"pred": pred, "label": label}, cache_path)
    return cache_path


def _result_path_for_tabular(trial: dict, stage_suffix: str) -> Path:
    return BASE_DIR / (
        f"rolling_result_{WINDOW_KEY}_{trial['model_key']}_{trial['handler_class']}_step{int(trial['step'])}_k{TOPK}"
        f"_d1_a{ACCOUNT}_r07_{trial['trial_name']}_{stage_suffix}.txt"
    )


def _run_tabular_trial(trial: dict, eval_segment: str, stage_suffix: str) -> dict:
    result_path = _result_path_for_tabular(trial, stage_suffix)
    run_lightgbm(
        step=int(trial["step"]),
        topk=TOPK,
        n_drop=1,
        window_key=WINDOW_KEY,
        model_key=trial["model_key"],
        exp_suffix=f"{trial['trial_name']}_{stage_suffix}",
        handler_class=trial["handler_class"],
        handler_module_path=trial["handler_module_path"],
        account=ACCOUNT,
        risk_degree=0.70,
        benchmark=BENCHMARK,
        instruments=INSTRUMENTS,
        eval_segment=eval_segment,
        backtest_segment=eval_segment,
    )
    return _parse_summary_file(result_path)


def _write_validation_best(best_row: dict):
    window = WINDOWS[WINDOW_KEY]
    lines = [
        f"window_key={WINDOW_KEY}",
        f"train={window['train']}",
        f"valid={window['valid']}",
        f"test={window['test']}",
        f"base_model_trial={best_row['base_model_trial']}",
        f"base_strategy_trial={best_row['base_strategy_trial']}",
        f"tabular_trial={best_row['tabular_trial']}",
        f"tabular_handler_class={best_row['tabular_handler_class']}",
        f"tabular_model_key={best_row['tabular_model_key']}",
        f"fusion_mode={best_row['fusion_mode']}",
        f"seq_weight={best_row['seq_weight']}",
        f"tab_weight={best_row['tab_weight']}",
        f"validation_IC={best_row['IC']}",
        f"validation_Rank_IC={best_row['Rank IC']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"validation_score={best_row['score']}",
        f"validation_cache_path={best_row['validation_cache_path']}",
        f"validation_recorder_id={best_row['validation_recorder_id']}",
        f"tabular_validation_result_path={best_row['tabular_validation_result_path']}",
        f"tabular_validation_recorder_id={best_row['tabular_validation_recorder_id']}",
    ]
    VALIDATION_BEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_final_test(best_row: dict, test_result: dict, baseline_test: dict):
    window = WINDOWS[WINDOW_KEY]
    improved = float(test_result["with_cost_ann_return"]) > float(baseline_test["test_with_cost_ann_return"])
    lines = [
        f"window_key={WINDOW_KEY}",
        f"train={window['train']}",
        f"valid={window['valid']}",
        f"test={window['test']}",
        f"selection_source=validation_only",
        f"test_improves_incumbent={improved}",
        f"base_model_trial={best_row['base_model_trial']}",
        f"base_strategy_trial={best_row['base_strategy_trial']}",
        f"tabular_trial={best_row['tabular_trial']}",
        f"tabular_handler_class={best_row['tabular_handler_class']}",
        f"tabular_model_key={best_row['tabular_model_key']}",
        f"fusion_mode={best_row['fusion_mode']}",
        f"seq_weight={best_row['seq_weight']}",
        f"tab_weight={best_row['tab_weight']}",
        f"validation_score={best_row['score']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"test_cache_path={test_result['cache_path']}",
        f"test_recorder_id={test_result['recorder_id']}",
        f"test_IC={test_result['IC']}",
        f"test_Rank_IC={test_result['Rank IC']}",
        f"test_with_cost_ann_return={test_result['with_cost_ann_return']}",
        f"test_with_cost_ir={test_result['with_cost_ir']}",
        f"test_with_cost_mdd={test_result['with_cost_mdd']}",
        f"incumbent_test_with_cost_ann_return={baseline_test['test_with_cost_ann_return']}",
        f"incumbent_test_with_cost_ir={baseline_test['test_with_cost_ir']}",
        f"incumbent_test_with_cost_mdd={baseline_test['test_with_cost_mdd']}",
        f"tabular_test_result_path={test_result['tabular_test_result_path']}",
        f"tabular_test_recorder_id={test_result['tabular_test_recorder_id']}",
    ]
    FINAL_TEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    alstm_validation = _parse_summary_file(ALSTM_VALIDATION_BEST_PATH)
    alstm_final = _parse_summary_file(ALSTM_FINAL_TEST_PATH)
    strategy_cfg_valid = _load_alstm_strategy_cfg(alstm_validation, eval_segment="valid")
    strategy_cfg_test = _load_alstm_strategy_cfg(alstm_validation, eval_segment="test")
    seq_valid_pred, seq_valid_label = _load_signal_from_cache(alstm_validation["validation_cache_path"])

    rows: list[dict] = [
        {
            "trial_name": "strict_retrain_fusion_baseline_alstm_only",
            "base_model_trial": alstm_validation["model_trial"],
            "base_strategy_trial": alstm_validation["strategy_trial"],
            "tabular_trial": "none",
            "tabular_handler_class": "none",
            "tabular_model_key": "none",
            "fusion_mode": "baseline",
            "seq_weight": 1.0,
            "tab_weight": 0.0,
            "status": "success",
            "IC": float(alstm_validation["validation_IC"]),
            "Rank IC": float(alstm_validation["validation_Rank_IC"]),
            "with_cost_ann_return": float(alstm_validation["validation_with_cost_ann_return"]),
            "with_cost_ir": float(alstm_validation["validation_with_cost_ir"]),
            "with_cost_mdd": float(alstm_validation["validation_with_cost_mdd"]),
            "score": float(alstm_validation["validation_score"]),
            "validation_cache_path": str(alstm_validation["validation_cache_path"]),
            "validation_recorder_id": pd.NA,
            "tabular_validation_result_path": pd.NA,
            "tabular_validation_recorder_id": pd.NA,
        }
    ]

    for trial in TABULAR_TRIALS:
        validation_result = _run_tabular_trial(trial, eval_segment="valid", stage_suffix="strict_retrain_valid")
        tab_valid_pred, tab_valid_label = _load_signal_from_recorder(validation_result["recorder_id"])
        seq_pred_aligned, tab_pred_aligned, label_aligned = _align_signals(
            seq_valid_pred, seq_valid_label, tab_valid_pred, tab_valid_label
        )

        for mode in FUSION_MODES:
            seq_norm = _normalize_scores(seq_pred_aligned, mode)
            tab_norm = _normalize_scores(tab_pred_aligned, mode)
            for seq_weight in SEQ_WEIGHTS:
                tab_weight = round(1.0 - seq_weight, 2)
                row = {
                    "trial_name": f"strict_retrain_fusion_{trial['trial_name']}_{mode}_sw{str(seq_weight).replace('.', '')}",
                    "base_model_trial": alstm_validation["model_trial"],
                    "base_strategy_trial": alstm_validation["strategy_trial"],
                    "tabular_trial": trial["trial_name"],
                    "tabular_handler_class": trial["handler_class"],
                    "tabular_model_key": trial["model_key"],
                    "fusion_mode": mode,
                    "seq_weight": seq_weight,
                    "tab_weight": tab_weight,
                    "status": "running",
                }
                try:
                    fused_pred = seq_norm.mul(seq_weight).add(tab_norm.mul(tab_weight), fill_value=0.0)
                    validation_cache_path = _write_fused_cache(row["trial_name"], fused_pred, label_aligned)
                    metrics = _evaluate_signal(
                        recorder_name=row["trial_name"],
                        pred=fused_pred,
                        label=label_aligned,
                        strategy_cfg=strategy_cfg_valid,
                        experiment_name="alpha360_alpha158_strict_retrain_validation_eval",
                    )
                    row.update(
                        {
                            "status": "success",
                            "IC": metrics["IC"],
                            "Rank IC": metrics["Rank IC"],
                            "with_cost_ann_return": metrics["with_cost_ann_return"],
                            "with_cost_ir": metrics["with_cost_ir"],
                            "with_cost_mdd": metrics["with_cost_mdd"],
                            "score": metrics["score"],
                            "validation_cache_path": str(validation_cache_path),
                            "validation_recorder_id": metrics["recorder_id"],
                            "tabular_validation_result_path": str(_result_path_for_tabular(trial, "strict_retrain_valid")),
                            "tabular_validation_recorder_id": validation_result["recorder_id"],
                        }
                    )
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = repr(exc)[:500]

                rows = [existing for existing in rows if existing.get("trial_name") != row["trial_name"]] + [row]
                _persist(rows, VALIDATION_RESULTS_PATH)

    success_rows = [row for row in rows if row["status"] == "success"]
    if not success_rows:
        raise RuntimeError("no successful retrain-fusion validation rows")

    best_row = max(success_rows, key=lambda item: item["score"])
    _write_validation_best(best_row)

    if best_row["tabular_trial"] == "none":
        test_result = {
            "cache_path": alstm_final["test_cache_path"],
            "recorder_id": alstm_final["test_recorder_id"],
            "IC": alstm_final["test_IC"],
            "Rank IC": alstm_final["test_Rank_IC"],
            "with_cost_ann_return": alstm_final["test_with_cost_ann_return"],
            "with_cost_ir": alstm_final["test_with_cost_ir"],
            "with_cost_mdd": alstm_final["test_with_cost_mdd"],
            "tabular_test_result_path": pd.NA,
            "tabular_test_recorder_id": pd.NA,
        }
        _write_final_test(best_row, test_result, alstm_final)
        return

    selected_trial = next(trial for trial in TABULAR_TRIALS if trial["trial_name"] == best_row["tabular_trial"])
    tabular_test_result = _run_tabular_trial(selected_trial, eval_segment="test", stage_suffix="strict_retrain_final_test")

    seq_test_pred, seq_test_label = _load_signal_from_cache(alstm_final["test_cache_path"])
    tab_test_pred, tab_test_label = _load_signal_from_recorder(tabular_test_result["recorder_id"])
    seq_test_aligned, tab_test_aligned, test_label_aligned = _align_signals(
        seq_test_pred, seq_test_label, tab_test_pred, tab_test_label
    )
    seq_test_norm = _normalize_scores(seq_test_aligned, best_row["fusion_mode"])
    tab_test_norm = _normalize_scores(tab_test_aligned, best_row["fusion_mode"])
    fused_test_pred = seq_test_norm.mul(best_row["seq_weight"]).add(tab_test_norm.mul(best_row["tab_weight"]), fill_value=0.0)
    test_cache_path = _write_fused_cache(f"{best_row['trial_name']}_final_test", fused_test_pred, test_label_aligned)
    test_metrics = _evaluate_signal(
        recorder_name=f"{best_row['trial_name']}_final_test",
        pred=fused_test_pred,
        label=test_label_aligned,
        strategy_cfg=strategy_cfg_test,
        experiment_name="alpha360_alpha158_strict_retrain_final_test_eval",
    )
    test_metrics.update(
        {
            "cache_path": str(test_cache_path),
            "tabular_test_result_path": str(_result_path_for_tabular(selected_trial, "strict_retrain_final_test")),
            "tabular_test_recorder_id": tabular_test_result["recorder_id"],
        }
    )
    _write_final_test(best_row, test_metrics, alstm_final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrain Alpha158/TechAlpha158 branches, tune fusion on validation, judge only by final test."
    )
    parser.parse_args()
    main()
