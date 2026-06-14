from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord

from rolling_tra_alpha360_latest import PROVIDER_URI, WINDOWS, _metric_value, build_base_task, run as run_tra
from tra_local_helpers import (
    _align_signals,
    _load_signal_from_cache,
    _load_signal_from_recorder,
    _normalize_scores,
    _parse_summary_file,
    _write_fused_cache,
    localize_value,
)


BASE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BASE_DIR / "current_best_baseline_manifest.json"
TRA_VALIDATION_BEST_PATH = BASE_DIR / "tra_strict_validation_best.txt"
TRA_FINAL_TEST_PATH = BASE_DIR / "tra_strict_final_test.txt"

VALIDATION_SUMMARY_PATH = BASE_DIR / "current_best_multiseed_ensemble_validation.txt"
FINAL_TEST_SUMMARY_PATH = BASE_DIR / "current_best_multiseed_ensemble_final_test.txt"

ACCOUNT = 150000
TOPK = 5
WINDOW_KEY = "w1"
BENCHMARK = "SH000300"
INSTRUMENTS = "csi300"
DEFAULT_SEEDS = [42, 2026, 3407]


def _objective(metrics: dict) -> float:
    ann = float(metrics["with_cost_ann_return"])
    ir = float(metrics["with_cost_ir"])
    mdd = float(metrics["with_cost_mdd"])
    return ann * 1000 + ir * 50 + mdd * 20


def _parse_seeds(raw: str | None) -> list[int]:
    if not raw:
        return list(DEFAULT_SEEDS)
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _load_manifest() -> dict:
    return localize_value(json.loads(MANIFEST_PATH.read_text(encoding="utf-8")))


def _seed_runs_key(eval_segment: str) -> str:
    return "seed_runs_valid" if eval_segment == "valid" else "seed_runs_test"


def _existing_seed_run(summary_path: Path, eval_segment: str, seed: int) -> dict | None:
    if not summary_path.exists():
        return None

    summary = _parse_summary_file(summary_path)
    seed_runs = summary.get(_seed_runs_key(eval_segment))
    if seed_runs is None:
        return None
    if isinstance(seed_runs, str):
        seed_runs = ast.literal_eval(seed_runs)

    for seed_run in seed_runs:
        if int(seed_run.get("seed", -1)) != seed:
            continue
        cache_path = seed_run.get("cache_path")
        if not cache_path or not Path(str(cache_path)).exists():
            continue
        return {
            "seed": seed,
            "cache_path": str(cache_path),
            "result_path": str(seed_run.get("result_path")) if seed_run.get("result_path") is not None else None,
            "recorder_id": str(seed_run.get("recorder_id")) if seed_run.get("recorder_id") is not None else None,
        }
    return None


def _strategy_cfg_from_summary(summary: dict, eval_segment: str) -> dict:
    return {
        "handler_kwargs_extra": summary["handler_kwargs_extra"],
        "strategy_class": summary["strategy_class"],
        "strategy_module_path": summary["strategy_module_path"],
        "kwargs_extra": summary["kwargs_extra"],
        "n_drop": int(summary["n_drop"]),
        "risk_degree": float(summary["risk_degree"]),
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
        model_key="tra_lstm_base",
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


def _average_seq_signals(seed_signals: list[tuple[int, pd.DataFrame, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_index = seed_signals[0][1].index
    for _, pred, label in seed_signals:
        common_index = common_index.intersection(pred.index).intersection(label.index)
    common_index = common_index.sort_values()

    pred_sum = None
    label_ref = None
    for _, pred, label in seed_signals:
        pred_slice = pred.loc[common_index].sort_index()
        label_slice = label.loc[common_index].sort_index()
        pred_sum = pred_slice if pred_sum is None else pred_sum.add(pred_slice, fill_value=0.0)
        if label_ref is None:
            label_ref = label_slice

    assert pred_sum is not None and label_ref is not None
    return pred_sum / float(len(seed_signals)), label_ref


def _load_signal_by_summary_or_recorder(result_path: str | Path | None, recorder_id: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if result_path:
        summary_path = Path(str(result_path))
        if summary_path.exists():
            summary = _parse_summary_file(summary_path)
            cache_path = summary.get("cache_path")
            if cache_path:
                return _load_signal_from_cache(cache_path)
    if recorder_id and str(recorder_id) not in {"", "None", "nan", "<NA>"}:
        return _load_signal_from_recorder(str(recorder_id))
    raise FileNotFoundError("no usable result summary or recorder artifact for tabular signal")


def _run_seeded_tra(
    seed: int,
    tra_summary: dict,
    eval_segment: str,
    gpu_slots: str,
    reuse_seed_42: bool,
    existing_summary_path: Path,
) -> dict:
    if reuse_seed_42 and seed == 42:
        summary = tra_summary
        return {
            "seed": seed,
            "cache_path": str(summary[f"{eval_segment}ation_cache_path"]) if eval_segment == "valid" else str(summary["test_cache_path"]),
            "result_path": str(summary[f"{eval_segment}ation_result_path"]) if eval_segment == "valid" else str(summary["test_result_path"]),
            "recorder_id": str(summary[f"{eval_segment}ation_recorder_id"]) if eval_segment == "valid" and "validation_recorder_id" in summary else str(summary.get("test_recorder_id")),
        }

    existing_seed_run = _existing_seed_run(existing_summary_path, eval_segment, seed)
    if existing_seed_run is not None:
        return existing_seed_run

    exp_suffix = f"multiseed_s{seed}_{eval_segment}"
    result = run_tra(
        step=int(tra_summary["step"]),
        topk=TOPK,
        n_drop=int(tra_summary["n_drop"]),
        window_key=WINDOW_KEY,
        model_key=tra_summary["model_key"],
        exp_suffix=exp_suffix,
        handler_class="Alpha360",
        handler_module_path="qlib.contrib.data.handler",
        account=ACCOUNT,
        risk_degree=float(tra_summary["risk_degree"]),
        benchmark=BENCHMARK,
        instruments=INSTRUMENTS,
        handler_kwargs_extra=tra_summary["handler_kwargs_extra"],
        strategy_class=tra_summary["strategy_class"],
        strategy_module_path=tra_summary["strategy_module_path"],
        strategy_kwargs_extra=tra_summary["kwargs_extra"],
        gpu_slots=gpu_slots,
        eval_segment=eval_segment,
        backtest_segment=eval_segment,
        model_kwargs_override={"seed": seed},
    )
    result["seed"] = seed
    return result


def _load_seq_seed_signals(
    seeds: list[int],
    summary: dict,
    eval_segment: str,
    gpu_slots: str,
    reuse_seed_42: bool,
    existing_summary_path: Path,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    seed_runs: list[dict] = []
    seed_signals: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
    for seed in seeds:
        run_result = _run_seeded_tra(
            seed=seed,
            tra_summary=summary,
            eval_segment=eval_segment,
            gpu_slots=gpu_slots,
            reuse_seed_42=reuse_seed_42,
            existing_summary_path=existing_summary_path,
        )
        pred, label = _load_signal_from_cache(run_result["cache_path"])
        seed_runs.append(run_result)
        seed_signals.append((seed, pred, label))
    ensemble_pred, ensemble_label = _average_seq_signals(seed_signals)
    return seed_runs, ensemble_pred, ensemble_label


def _write_summary(path: Path, lines: list[str]):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run current best csi300 baseline with a multi-seed TRA ensemble.")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--gpu-slots", default="0,1")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument(
        "--seed-runs-test-only",
        action="store_true",
        help="Only materialize the test-side seed_runs summary needed by downstream rank-ensemble evaluation.",
    )
    parser.add_argument(
        "--materialize-test-summary",
        action="store_true",
        help="Write the multi-seed test summary even if validation does not improve the current best.",
    )
    parser.add_argument("--force-seed42-rerun", action="store_true")
    parser.add_argument("--tra-validation-best-path", default=str(TRA_VALIDATION_BEST_PATH))
    parser.add_argument("--tra-final-test-path", default=str(TRA_FINAL_TEST_PATH))
    parser.add_argument("--validation-summary-path", default=str(VALIDATION_SUMMARY_PATH))
    parser.add_argument("--final-test-summary-path", default=str(FINAL_TEST_SUMMARY_PATH))
    args = parser.parse_args()

    manifest = _load_manifest()
    current_best = manifest["current_best_baseline"]
    tra_validation_best_path = Path(args.tra_validation_best_path).expanduser().resolve()
    tra_final_test_path = Path(args.tra_final_test_path).expanduser().resolve()
    validation_summary_path = Path(args.validation_summary_path).expanduser().resolve()
    final_test_summary_path = Path(args.final_test_summary_path).expanduser().resolve()
    tra_validation = _parse_summary_file(tra_validation_best_path)
    tra_final = _parse_summary_file(tra_final_test_path)
    seeds = _parse_seeds(args.seeds)
    reuse_seed_42 = not args.force_seed42_rerun

    strategy_cfg_valid = _strategy_cfg_from_summary(tra_validation, eval_segment="valid")

    if args.seed_runs_test_only:
        seed_runs_test, _, _ = _load_seq_seed_signals(
            seeds=seeds,
            summary=tra_final,
            eval_segment="test",
            gpu_slots=args.gpu_slots,
            reuse_seed_42=reuse_seed_42,
            existing_summary_path=final_test_summary_path,
        )
        _write_summary(
            final_test_summary_path,
            [
                f"window_key={WINDOW_KEY}",
                f"train={WINDOWS[WINDOW_KEY]['train']}",
                f"valid={WINDOWS[WINDOW_KEY]['valid']}",
                f"test={WINDOWS[WINDOW_KEY]['test']}",
                f"tra_validation_best_path={tra_validation_best_path}",
                f"tra_final_test_path={tra_final_test_path}",
                f"seeds={seeds}",
                f"reuse_seed_42={reuse_seed_42}",
                "seed_runs_test_only=True",
                f"seed_runs_test={seed_runs_test}",
            ],
        )
        return

    seed_runs_valid, seq_valid_pred, seq_valid_label = _load_seq_seed_signals(
        seeds=seeds,
        summary=tra_validation,
        eval_segment="valid",
        gpu_slots=args.gpu_slots,
        reuse_seed_42=reuse_seed_42,
        existing_summary_path=validation_summary_path,
    )
    seq_valid_metrics = _evaluate_signal(
        recorder_name=f"current_best_multiseed_seq_valid_{'_'.join(str(seed) for seed in seeds)}",
        pred=seq_valid_pred,
        label=seq_valid_label,
        strategy_cfg=strategy_cfg_valid,
        experiment_name="current_best_multiseed_seq_validation_eval",
    )
    seq_valid_cache_path = _write_fused_cache(
        f"current_best_multiseed_seq_valid_{'_'.join(str(seed) for seed in seeds)}",
        seq_valid_pred,
        seq_valid_label,
    )

    tab_valid_pred, tab_valid_label = _load_signal_by_summary_or_recorder(
        current_best["validation"].get("tabular_result_path"),
        current_best["validation"].get("tabular_recorder_id"),
    )
    seq_valid_aligned, tab_valid_aligned, valid_label_aligned = _align_signals(
        seq_valid_pred,
        seq_valid_label,
        tab_valid_pred,
        tab_valid_label,
    )
    seq_valid_norm = _normalize_scores(seq_valid_aligned, current_best["fusion_mode"])
    tab_valid_norm = _normalize_scores(tab_valid_aligned, current_best["fusion_mode"])
    fused_valid_pred = seq_valid_norm.mul(float(current_best["seq_weight"])).add(
        tab_valid_norm.mul(float(current_best["tab_weight"])), fill_value=0.0
    )
    fused_valid_cache_path = _write_fused_cache(
        f"current_best_multiseed_fused_valid_{'_'.join(str(seed) for seed in seeds)}",
        fused_valid_pred,
        valid_label_aligned,
    )
    fused_valid_metrics = _evaluate_signal(
        recorder_name=f"current_best_multiseed_fused_valid_{'_'.join(str(seed) for seed in seeds)}",
        pred=fused_valid_pred,
        label=valid_label_aligned,
        strategy_cfg=strategy_cfg_valid,
        experiment_name="current_best_multiseed_fused_validation_eval",
    )

    validation_improves = float(fused_valid_metrics["score"]) > float(current_best["validation"]["score"])
    _write_summary(
        validation_summary_path,
        [
            f"window_key={WINDOW_KEY}",
            f"train={WINDOWS[WINDOW_KEY]['train']}",
            f"valid={WINDOWS[WINDOW_KEY]['valid']}",
            f"test={WINDOWS[WINDOW_KEY]['test']}",
            f"tra_validation_best_path={tra_validation_best_path}",
            f"tra_final_test_path={tra_final_test_path}",
            f"seeds={seeds}",
            f"reuse_seed_42={reuse_seed_42}",
            f"fusion_mode={current_best['fusion_mode']}",
            f"seq_weight={current_best['seq_weight']}",
            f"tab_weight={current_best['tab_weight']}",
            f"seed_runs_valid={seed_runs_valid}",
            f"seq_validation_cache_path={seq_valid_cache_path}",
            f"seq_validation_recorder_id={seq_valid_metrics['recorder_id']}",
            f"seq_validation_IC={seq_valid_metrics['IC']}",
            f"seq_validation_Rank_IC={seq_valid_metrics['Rank IC']}",
            f"seq_validation_with_cost_ann_return={seq_valid_metrics['with_cost_ann_return']}",
            f"seq_validation_with_cost_ir={seq_valid_metrics['with_cost_ir']}",
            f"seq_validation_with_cost_mdd={seq_valid_metrics['with_cost_mdd']}",
            f"seq_validation_score={seq_valid_metrics['score']}",
            f"fused_validation_cache_path={fused_valid_cache_path}",
            f"fused_validation_recorder_id={fused_valid_metrics['recorder_id']}",
            f"fused_validation_IC={fused_valid_metrics['IC']}",
            f"fused_validation_Rank_IC={fused_valid_metrics['Rank IC']}",
            f"fused_validation_with_cost_ann_return={fused_valid_metrics['with_cost_ann_return']}",
            f"fused_validation_with_cost_ir={fused_valid_metrics['with_cost_ir']}",
            f"fused_validation_with_cost_mdd={fused_valid_metrics['with_cost_mdd']}",
            f"fused_validation_score={fused_valid_metrics['score']}",
            f"current_best_validation_score={current_best['validation']['score']}",
            f"current_best_validation_with_cost_ann_return={current_best['validation']['with_cost_ann_return']}",
            f"current_best_validation_with_cost_ir={current_best['validation']['with_cost_ir']}",
            f"current_best_validation_with_cost_mdd={current_best['validation']['with_cost_mdd']}",
            f"validation_improves_current_best={validation_improves}",
        ],
    )

    if args.validation_only:
        return

    if not validation_improves and not args.materialize_test_summary:
        return

    strategy_cfg_test = _strategy_cfg_from_summary(tra_validation, eval_segment="test")
    seed_runs_test, seq_test_pred, seq_test_label = _load_seq_seed_signals(
        seeds=seeds,
        summary=tra_final,
        eval_segment="test",
        gpu_slots=args.gpu_slots,
        reuse_seed_42=reuse_seed_42,
        existing_summary_path=final_test_summary_path,
    )
    seq_test_metrics = _evaluate_signal(
        recorder_name=f"current_best_multiseed_seq_test_{'_'.join(str(seed) for seed in seeds)}",
        pred=seq_test_pred,
        label=seq_test_label,
        strategy_cfg=strategy_cfg_test,
        experiment_name="current_best_multiseed_seq_final_test_eval",
    )
    seq_test_cache_path = _write_fused_cache(
        f"current_best_multiseed_seq_test_{'_'.join(str(seed) for seed in seeds)}",
        seq_test_pred,
        seq_test_label,
    )

    tab_test_pred, tab_test_label = _load_signal_by_summary_or_recorder(
        current_best["final_test"].get("tabular_result_path"),
        current_best["final_test"].get("tabular_recorder_id"),
    )
    seq_test_aligned, tab_test_aligned, test_label_aligned = _align_signals(
        seq_test_pred,
        seq_test_label,
        tab_test_pred,
        tab_test_label,
    )
    seq_test_norm = _normalize_scores(seq_test_aligned, current_best["fusion_mode"])
    tab_test_norm = _normalize_scores(tab_test_aligned, current_best["fusion_mode"])
    fused_test_pred = seq_test_norm.mul(float(current_best["seq_weight"])).add(
        tab_test_norm.mul(float(current_best["tab_weight"])), fill_value=0.0
    )
    fused_test_cache_path = _write_fused_cache(
        f"current_best_multiseed_fused_test_{'_'.join(str(seed) for seed in seeds)}",
        fused_test_pred,
        test_label_aligned,
    )
    fused_test_metrics = _evaluate_signal(
        recorder_name=f"current_best_multiseed_fused_test_{'_'.join(str(seed) for seed in seeds)}",
        pred=fused_test_pred,
        label=test_label_aligned,
        strategy_cfg=strategy_cfg_test,
        experiment_name="current_best_multiseed_fused_final_test_eval",
    )

    final_improves = float(fused_test_metrics["with_cost_ann_return"]) > float(current_best["final_test"]["with_cost_ann_return"])
    _write_summary(
        final_test_summary_path,
        [
            f"window_key={WINDOW_KEY}",
            f"train={WINDOWS[WINDOW_KEY]['train']}",
            f"valid={WINDOWS[WINDOW_KEY]['valid']}",
            f"test={WINDOWS[WINDOW_KEY]['test']}",
            f"tra_validation_best_path={tra_validation_best_path}",
            f"tra_final_test_path={tra_final_test_path}",
            f"seeds={seeds}",
            f"reuse_seed_42={reuse_seed_42}",
            f"fusion_mode={current_best['fusion_mode']}",
            f"seq_weight={current_best['seq_weight']}",
            f"tab_weight={current_best['tab_weight']}",
            f"seed_runs_test={seed_runs_test}",
            f"seq_test_cache_path={seq_test_cache_path}",
            f"seq_test_recorder_id={seq_test_metrics['recorder_id']}",
            f"seq_test_IC={seq_test_metrics['IC']}",
            f"seq_test_Rank_IC={seq_test_metrics['Rank IC']}",
            f"seq_test_with_cost_ann_return={seq_test_metrics['with_cost_ann_return']}",
            f"seq_test_with_cost_ir={seq_test_metrics['with_cost_ir']}",
            f"seq_test_with_cost_mdd={seq_test_metrics['with_cost_mdd']}",
            f"seq_test_score={seq_test_metrics['score']}",
            f"fused_test_cache_path={fused_test_cache_path}",
            f"fused_test_recorder_id={fused_test_metrics['recorder_id']}",
            f"fused_test_IC={fused_test_metrics['IC']}",
            f"fused_test_Rank_IC={fused_test_metrics['Rank IC']}",
            f"fused_test_with_cost_ann_return={fused_test_metrics['with_cost_ann_return']}",
            f"fused_test_with_cost_ir={fused_test_metrics['with_cost_ir']}",
            f"fused_test_with_cost_mdd={fused_test_metrics['with_cost_mdd']}",
            f"fused_test_score={fused_test_metrics['score']}",
            f"current_best_test_with_cost_ann_return={current_best['final_test']['with_cost_ann_return']}",
            f"current_best_test_with_cost_ir={current_best['final_test']['with_cost_ir']}",
            f"current_best_test_with_cost_mdd={current_best['final_test']['with_cost_mdd']}",
            f"validation_improves_current_best={validation_improves}",
            f"materialized_test_summary_without_validation_improve={args.materialize_test_summary and not validation_improves}",
            f"final_test_improves_current_best={final_improves}",
        ],
    )


if __name__ == "__main__":
    main()