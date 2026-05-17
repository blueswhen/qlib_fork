from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rolling_tra_alpha360_latest import WINDOWS, run as run_tra
from tune_alpha360_alpha158_dnn_strict_fusion import (
    _align_signals,
    _evaluate_signal,
    _load_signal_from_cache,
    _normalize_scores,
    _parse_summary_file,
    _persist,
    _write_fused_cache,
)
import tune_alpha360_tra_fundamental_xgb_weight_refine as fund_xgb_mod


BASE_DIR = Path(__file__).resolve().parent

VALIDATION_RESULTS_PATH = BASE_DIR / "tra_fundxgb_joint_repro_validation_results.csv"
VALIDATION_BEST_PATH = BASE_DIR / "tra_fundxgb_joint_repro_validation_best.txt"
FINAL_TEST_PATH = BASE_DIR / "tra_fundxgb_joint_repro_final_test.txt"

WINDOW_KEY = "w1"
ACCOUNT = 150000
TOPK = 5
BENCHMARK = "SH000300"
INSTRUMENTS = "csi300"
GPU_SLOTS = "0,1"

H5_LABEL = {"label": ["Ref($close, -6) / Ref($close, -1) - 1"]}
DATASET_KWARGS = {"batch_size": 16384}
MODEL_KEY = "tra_lstm_base"
STEP = 240

MODEL_KWARGS_OVERRIDE = {
    "model_config": {
        "input_size": 6,
        "hidden_size": 64,
        "num_layers": 2,
        "rnn_arch": "LSTM",
        "use_attn": True,
        "dropout": 0.0,
    },
    "tra_config": {
        "num_states": 3,
        "rnn_arch": "LSTM",
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.0,
        "tau": 1.0,
        "src_info": "LR_TPE",
    },
    "model_type": "RNN",
    "lr": 1e-3,
    "n_epochs": 100,
    "early_stop": 20,
    "update_freq": 1,
    "eval_freq": 5,
    "lamb": 1.0,
    "rho": 0.99,
    "alpha": 0.5,
    "pretrain": True,
    "transport_method": "router",
    "memory_mode": "sample",
}

STRICT_VALIDATION_PATH = BASE_DIR / "tra_strict_validation_best.txt"
STRICT_FINAL_TEST_PATH = BASE_DIR / "tra_strict_final_test.txt"
CURRENT_BEST_VALIDATION_PATH = BASE_DIR / "alpha360_tra_fundamental_xgb_weight_refine_validation_best.txt"
CURRENT_BEST_FINAL_PATH = BASE_DIR / "alpha360_tra_fundamental_xgb_weight_refine_final_test.txt"

EXACT_STRATEGY = {
    "trial_name": "prac_m050_h5_r07",
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

SEARCH_STRATEGIES = [
    EXACT_STRATEGY,
    {
        "trial_name": "prac_m050_hold3_r085",
        "strategy_class": "PracticalTopkDropoutStrategy",
        "strategy_module_path": "qlib.contrib.strategy.custom_signal_strategy",
        "n_drop": 1,
        "risk_degree": 0.85,
        "kwargs_extra": {
            "score_margin": 0.005,
            "hold_thresh": 3,
            "slot_budget_ratio": 1.0,
        },
    },
    {
        "trial_name": "base_topk",
        "strategy_class": "TopkDropoutStrategy",
        "strategy_module_path": "qlib.contrib.strategy",
        "n_drop": 1,
        "risk_degree": 0.70,
        "kwargs_extra": {},
    },
    {
        "trial_name": "prac_m025_hold3_r07",
        "strategy_class": "PracticalTopkDropoutStrategy",
        "strategy_module_path": "qlib.contrib.strategy.custom_signal_strategy",
        "n_drop": 1,
        "risk_degree": 0.70,
        "kwargs_extra": {
            "score_margin": 0.0025,
            "hold_thresh": 3,
            "slot_budget_ratio": 1.0,
        },
    },
]

TAB_WEIGHTS = [0.0, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]


def _objective(metrics: dict) -> float:
    ann = float(metrics["with_cost_ann_return"])
    ir = float(metrics["with_cost_ir"])
    mdd = float(metrics["with_cost_mdd"])
    return ann * 1000 + ir * 50 + mdd * 20


def _suffix_path(path: Path, result_suffix: str) -> Path:
    if not result_suffix:
        return path
    return path.with_name(f"{path.stem}_{result_suffix}{path.suffix}")


def _exp_suffix(base: str, result_suffix: str) -> str:
    if not result_suffix:
        return base
    return f"{base}_{result_suffix}"


def _tra_result_path(exp_suffix: str) -> Path:
    risk_tag = "07"
    return BASE_DIR / (
        f"rolling_result_tra_{WINDOW_KEY}_{MODEL_KEY}_Alpha360_step{STEP}_k{TOPK}_d1_a{ACCOUNT}_r{risk_tag}_{exp_suffix}.txt"
    )


def _load_existing_rows(path: Path) -> dict[tuple[str, str, str], dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row_map = {}
    for row in df.to_dict(orient="records"):
        row_map[(row["seq_source"], row["strategy_trial"], row["stage2_variant"])] = row
    return row_map


def _strategy_cfg(strategy_trial: dict, *, eval_segment: str) -> dict:
    return {
        "handler_kwargs_extra": H5_LABEL,
        "strategy_class": strategy_trial["strategy_class"],
        "strategy_module_path": strategy_trial["strategy_module_path"],
        "kwargs_extra": strategy_trial["kwargs_extra"],
        "n_drop": strategy_trial["n_drop"],
        "risk_degree": strategy_trial["risk_degree"],
        "eval_segment": eval_segment,
        "backtest_segment": eval_segment,
    }


def _load_seq_source_from_summary(name: str, validation_summary: dict, test_summary: dict) -> dict:
    valid_pred, valid_label = _load_signal_from_cache(validation_summary["validation_cache_path"])
    test_pred, test_label = _load_signal_from_cache(test_summary["test_cache_path"])
    return {
        "source_name": name,
        "source_type": "legacy_cache",
        "seed": pd.NA,
        "validation_summary": validation_summary,
        "test_summary": test_summary,
        "valid_pred": valid_pred,
        "valid_label": valid_label,
        "test_pred": test_pred,
        "test_label": test_label,
    }


def _train_seq_source(seed: int, result_suffix: str) -> dict:
    valid_exp_suffix = _exp_suffix(f"jointrepro_seq_s{seed}_valid", result_suffix)
    test_exp_suffix = _exp_suffix(f"jointrepro_seq_s{seed}_final_test", result_suffix)
    valid_result_path = _tra_result_path(valid_exp_suffix)
    test_result_path = _tra_result_path(test_exp_suffix)

    model_kwargs = dict(MODEL_KWARGS_OVERRIDE)
    model_kwargs["seed"] = seed

    if valid_result_path.exists():
        valid_summary = _parse_summary_file(valid_result_path)
    else:
        valid_summary = run_tra(
            step=STEP,
            topk=TOPK,
            n_drop=1,
            window_key=WINDOW_KEY,
            model_key=MODEL_KEY,
            exp_suffix=valid_exp_suffix,
            handler_class="Alpha360",
            handler_module_path="qlib.contrib.data.handler",
            account=ACCOUNT,
            risk_degree=0.70,
            benchmark=BENCHMARK,
            instruments=INSTRUMENTS,
            model_kwargs_override=model_kwargs,
            handler_kwargs_extra=H5_LABEL,
            dataset_kwargs_extra=DATASET_KWARGS,
            gpu_slots=GPU_SLOTS,
            eval_segment="valid",
            backtest_segment="valid",
            deterministic_runtime=True,
        )

    if test_result_path.exists():
        test_summary = _parse_summary_file(test_result_path)
    else:
        test_summary = run_tra(
            step=STEP,
            topk=TOPK,
            n_drop=1,
            window_key=WINDOW_KEY,
            model_key=MODEL_KEY,
            exp_suffix=test_exp_suffix,
            handler_class="Alpha360",
            handler_module_path="qlib.contrib.data.handler",
            account=ACCOUNT,
            risk_degree=0.70,
            benchmark=BENCHMARK,
            instruments=INSTRUMENTS,
            model_kwargs_override=model_kwargs,
            handler_kwargs_extra=H5_LABEL,
            dataset_kwargs_extra=DATASET_KWARGS,
            gpu_slots=GPU_SLOTS,
            eval_segment="test",
            backtest_segment="test",
            deterministic_runtime=True,
        )

    valid_pred, valid_label = _load_signal_from_cache(valid_summary["cache_path"])
    test_pred, test_label = _load_signal_from_cache(test_summary["cache_path"])
    return {
        "source_name": f"seed_{seed}",
        "source_type": "trained_seed",
        "seed": seed,
        "validation_summary": valid_summary,
        "test_summary": test_summary,
        "valid_pred": valid_pred,
        "valid_label": valid_label,
        "test_pred": test_pred,
        "test_label": test_label,
    }


def _prepare_seq_sources(seed_list: list[int], include_legacy_anchor: bool, result_suffix: str) -> list[dict]:
    sources: list[dict] = []
    if include_legacy_anchor:
        sources.append(
            _load_seq_source_from_summary(
                "legacy_exact_current_best",
                _parse_summary_file(STRICT_VALIDATION_PATH),
                _parse_summary_file(STRICT_FINAL_TEST_PATH),
            )
        )
    for seed in seed_list:
        sources.append(_train_seq_source(seed, result_suffix))
    return sources


def _prepare_xgb_branch(result_suffix: str) -> dict:
    result_path_valid = fund_xgb_mod._result_path_for_xgb(
        fund_xgb_mod.SELECTED_TRIAL,
        fund_xgb_mod.XGB_VALID_STAGE_SUFFIX,
        window_key=WINDOW_KEY,
        result_suffix=result_suffix,
    )
    result_path_test = fund_xgb_mod._result_path_for_xgb(
        fund_xgb_mod.SELECTED_TRIAL,
        fund_xgb_mod.XGB_TEST_STAGE_SUFFIX,
        window_key=WINDOW_KEY,
        result_suffix=result_suffix,
    )
    fallback_valid = fund_xgb_mod._result_path_for_xgb(
        fund_xgb_mod.SELECTED_TRIAL,
        fund_xgb_mod.XGB_VALID_STAGE_SUFFIX,
        window_key=WINDOW_KEY,
        result_suffix="",
    )
    fallback_test = fund_xgb_mod._result_path_for_xgb(
        fund_xgb_mod.SELECTED_TRIAL,
        fund_xgb_mod.XGB_TEST_STAGE_SUFFIX,
        window_key=WINDOW_KEY,
        result_suffix="",
    )

    if result_path_valid.exists():
        valid_summary = _parse_summary_file(result_path_valid)
    elif fallback_valid.exists():
        valid_summary = _parse_summary_file(fallback_valid)
    else:
        valid_summary = fund_xgb_mod._run_xgb_trial(
            fund_xgb_mod.SELECTED_TRIAL,
            eval_segment="valid",
            stage_suffix=fund_xgb_mod.XGB_VALID_STAGE_SUFFIX,
            window_key=WINDOW_KEY,
            result_suffix=result_suffix,
        )

    if result_path_test.exists():
        test_summary = _parse_summary_file(result_path_test)
    elif fallback_test.exists():
        test_summary = _parse_summary_file(fallback_test)
    else:
        test_summary = fund_xgb_mod._run_xgb_trial(
            fund_xgb_mod.SELECTED_TRIAL,
            eval_segment="test",
            stage_suffix=fund_xgb_mod.XGB_TEST_STAGE_SUFFIX,
            window_key=WINDOW_KEY,
            result_suffix=result_suffix,
        )

    valid_pred, valid_label = _load_signal_from_cache(valid_summary["cache_path"])
    test_pred, test_label = _load_signal_from_cache(test_summary["cache_path"])
    return {
        "trial": fund_xgb_mod.SELECTED_TRIAL,
        "valid_summary": valid_summary,
        "test_summary": test_summary,
        "valid_pred": valid_pred,
        "valid_label": valid_label,
        "test_pred": test_pred,
        "test_label": test_label,
    }


def _evaluate_validation(
    seq_sources: list[dict],
    xgb_branch: dict,
    validation_results_path: Path,
    *,
    result_suffix: str,
    workflow_eval: bool,
) -> dict[tuple[str, str, str], dict]:
    row_map = _load_existing_rows(validation_results_path)

    for source in seq_sources:
        for strategy_trial in SEARCH_STRATEGIES:
            strategy_cfg = _strategy_cfg(strategy_trial, eval_segment="valid")
            seq_pred_aligned, tab_pred_aligned, label_aligned = _align_signals(
                source["valid_pred"],
                source["valid_label"],
                xgb_branch["valid_pred"],
                xgb_branch["valid_label"],
            )
            seq_norm = _normalize_scores(seq_pred_aligned, fund_xgb_mod.FUSION_MODE)
            tab_norm = _normalize_scores(tab_pred_aligned, fund_xgb_mod.FUSION_MODE)

            for tab_weight in TAB_WEIGHTS:
                seq_weight = round(1.0 - tab_weight, 2)
                stage2_variant = "baseline_none" if tab_weight == 0.0 else f"fund_xgb_tw{str(tab_weight).replace('.', '')}"
                key = (source["source_name"], strategy_trial["trial_name"], stage2_variant)
                existing = row_map.get(key)
                if existing and existing.get("status") == "success":
                    continue

                row = {
                    "seq_source": source["source_name"],
                    "seq_source_type": source["source_type"],
                    "seq_seed": source["seed"],
                    "strategy_trial": strategy_trial["trial_name"],
                    "strategy_class": strategy_trial["strategy_class"],
                    "strategy_module_path": strategy_trial["strategy_module_path"],
                    "n_drop": strategy_trial["n_drop"],
                    "risk_degree": strategy_trial["risk_degree"],
                    "kwargs_extra": str(strategy_trial["kwargs_extra"]),
                    "stage2_family": "baseline" if tab_weight == 0.0 else "fund_xgb",
                    "stage2_variant": stage2_variant,
                    "tabular_trial": "none" if tab_weight == 0.0 else xgb_branch["trial"]["trial_name"],
                    "tabular_handler_class": "none" if tab_weight == 0.0 else xgb_branch["trial"]["handler_class"],
                    "tabular_model_key": "none" if tab_weight == 0.0 else xgb_branch["trial"]["model_key"],
                    "fusion_mode": "baseline" if tab_weight == 0.0 else fund_xgb_mod.FUSION_MODE,
                    "seq_weight": seq_weight,
                    "tab_weight": tab_weight,
                    "status": "running",
                }

                try:
                    pred_for_eval = seq_pred_aligned
                    cache_name = f"jointrepro_{source['source_name']}_{strategy_trial['trial_name']}_{stage2_variant}"
                    if tab_weight > 0.0:
                        pred_for_eval = seq_norm.mul(seq_weight).add(tab_norm.mul(tab_weight), fill_value=0.0)
                    validation_cache_path = _write_fused_cache(_exp_suffix(cache_name, result_suffix), pred_for_eval, label_aligned)
                    metrics = _evaluate_signal(
                        recorder_name=_exp_suffix(cache_name, result_suffix),
                        pred=pred_for_eval,
                        label=label_aligned,
                        strategy_cfg=strategy_cfg,
                        experiment_name="tra_fundxgb_joint_repro_validation_eval",
                        fast_mode=not workflow_eval,
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
                            "seq_validation_cache_path": source["validation_summary"].get("validation_cache_path", source["validation_summary"].get("cache_path")),
                            "tabular_validation_cache_path": pd.NA if tab_weight == 0.0 else xgb_branch["valid_summary"]["cache_path"],
                        }
                    )
                except Exception as exc:
                    row["status"] = "failed"
                    row["error"] = repr(exc)[:500]

                row_map[key] = row
                _persist(list(row_map.values()), validation_results_path)

    return row_map


def _write_validation_best(best_row: dict, output_path: Path):
    strict_validation = _parse_summary_file(STRICT_VALIDATION_PATH)
    lines = [
        f"window_key={strict_validation['window_key']}",
        f"train={strict_validation['train']}",
        f"valid={strict_validation['valid']}",
        f"test={strict_validation['test']}",
        "selection_scope=tra_fundxgb_joint_repro_validation",
        f"seq_source={best_row['seq_source']}",
        f"seq_source_type={best_row['seq_source_type']}",
        f"seq_seed={best_row['seq_seed']}",
        f"strategy_trial={best_row['strategy_trial']}",
        f"strategy_class={best_row['strategy_class']}",
        f"strategy_module_path={best_row['strategy_module_path']}",
        f"n_drop={best_row['n_drop']}",
        f"risk_degree={best_row['risk_degree']}",
        f"kwargs_extra={best_row['kwargs_extra']}",
        f"stage2_family={best_row['stage2_family']}",
        f"stage2_variant={best_row['stage2_variant']}",
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
        f"seq_validation_cache_path={best_row['seq_validation_cache_path']}",
        f"tabular_validation_cache_path={best_row['tabular_validation_cache_path']}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_final_test(best_row: dict, seq_sources: list[dict], xgb_branch: dict, *, result_suffix: str, workflow_eval: bool) -> dict:
    source = next(item for item in seq_sources if item["source_name"] == best_row["seq_source"])
    strategy_trial = next(item for item in SEARCH_STRATEGIES if item["trial_name"] == best_row["strategy_trial"])
    strategy_cfg = _strategy_cfg(strategy_trial, eval_segment="test")

    seq_pred_aligned, tab_pred_aligned, label_aligned = _align_signals(
        source["test_pred"],
        source["test_label"],
        xgb_branch["test_pred"],
        xgb_branch["test_label"],
    )

    pred_for_eval = seq_pred_aligned
    if float(best_row["tab_weight"]) > 0.0:
        seq_norm = _normalize_scores(seq_pred_aligned, best_row["fusion_mode"])
        tab_norm = _normalize_scores(tab_pred_aligned, best_row["fusion_mode"])
        pred_for_eval = seq_norm.mul(float(best_row["seq_weight"])).add(
            tab_norm.mul(float(best_row["tab_weight"])),
            fill_value=0.0,
        )

    cache_name = f"jointrepro_{best_row['seq_source']}_{best_row['strategy_trial']}_{best_row['stage2_variant']}_final_test"
    test_cache_path = _write_fused_cache(_exp_suffix(cache_name, result_suffix), pred_for_eval, label_aligned)
    metrics = _evaluate_signal(
        recorder_name=_exp_suffix(cache_name, result_suffix),
        pred=pred_for_eval,
        label=label_aligned,
        strategy_cfg=strategy_cfg,
        experiment_name="tra_fundxgb_joint_repro_final_test_eval",
        fast_mode=not workflow_eval,
    )
    metrics.update(
        {
            "cache_path": str(test_cache_path),
            "seq_test_cache_path": source["test_summary"].get("test_cache_path", source["test_summary"].get("cache_path")),
            "tabular_test_cache_path": pd.NA if float(best_row["tab_weight"]) == 0.0 else xgb_branch["test_summary"]["cache_path"],
        }
    )
    return metrics


def _write_final_test(best_row: dict, test_result: dict, output_path: Path):
    strict_final = _parse_summary_file(STRICT_FINAL_TEST_PATH)
    current_best_final = _parse_summary_file(CURRENT_BEST_FINAL_PATH)
    improves_tra = float(test_result["with_cost_ann_return"]) > float(strict_final["test_with_cost_ann_return"])
    improves_current_best = float(test_result["with_cost_ann_return"]) > float(current_best_final["test_with_cost_ann_return"])
    lines = [
        f"window_key={WINDOW_KEY}",
        f"train={WINDOWS[WINDOW_KEY]['train']}",
        f"valid={WINDOWS[WINDOW_KEY]['valid']}",
        f"test={WINDOWS[WINDOW_KEY]['test']}",
        "selection_scope=validation_only",
        f"test_improves_tra_incumbent={improves_tra}",
        f"test_improves_current_best_baseline={improves_current_best}",
        f"seq_source={best_row['seq_source']}",
        f"seq_source_type={best_row['seq_source_type']}",
        f"seq_seed={best_row['seq_seed']}",
        f"strategy_trial={best_row['strategy_trial']}",
        f"stage2_family={best_row['stage2_family']}",
        f"stage2_variant={best_row['stage2_variant']}",
        f"tabular_trial={best_row['tabular_trial']}",
        f"fusion_mode={best_row['fusion_mode']}",
        f"seq_weight={best_row['seq_weight']}",
        f"tab_weight={best_row['tab_weight']}",
        f"validation_score={best_row['score']}",
        f"validation_with_cost_ann_return={best_row['with_cost_ann_return']}",
        f"validation_with_cost_ir={best_row['with_cost_ir']}",
        f"validation_with_cost_mdd={best_row['with_cost_mdd']}",
        f"test_cache_path={test_result['cache_path']}",
        f"seq_test_cache_path={test_result['seq_test_cache_path']}",
        f"tabular_test_cache_path={test_result['tabular_test_cache_path']}",
        f"test_recorder_id={test_result['recorder_id']}",
        f"test_IC={test_result['IC']}",
        f"test_Rank_IC={test_result['Rank IC']}",
        f"test_with_cost_ann_return={test_result['with_cost_ann_return']}",
        f"test_with_cost_ir={test_result['with_cost_ir']}",
        f"test_with_cost_mdd={test_result['with_cost_mdd']}",
        f"tra_incumbent_test_with_cost_ann_return={strict_final['test_with_cost_ann_return']}",
        f"tra_incumbent_test_with_cost_ir={strict_final['test_with_cost_ir']}",
        f"tra_incumbent_test_with_cost_mdd={strict_final['test_with_cost_mdd']}",
        f"current_best_test_with_cost_ann_return={current_best_final['test_with_cost_ann_return']}",
        f"current_best_test_with_cost_ir={current_best_final['test_with_cost_ir']}",
        f"current_best_test_with_cost_mdd={current_best_final['test_with_cost_mdd']}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_seed_list(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main(args: argparse.Namespace):
    validation_results_path = _suffix_path(VALIDATION_RESULTS_PATH, args.result_suffix)
    validation_best_path = _suffix_path(VALIDATION_BEST_PATH, args.result_suffix)
    final_test_path = _suffix_path(FINAL_TEST_PATH, args.result_suffix)

    seq_sources = _prepare_seq_sources(
        _parse_seed_list(args.seed_list),
        include_legacy_anchor=not args.no_legacy_anchor,
        result_suffix=args.result_suffix,
    )
    xgb_branch = _prepare_xgb_branch(args.result_suffix)
    row_map = _evaluate_validation(
        seq_sources,
        xgb_branch,
        validation_results_path,
        result_suffix=args.result_suffix,
        workflow_eval=args.workflow_eval,
    )

    success_rows = [row for row in row_map.values() if row.get("status") == "success"]
    if not success_rows:
        raise RuntimeError("no successful joint validation rows produced")

    best_row = max(success_rows, key=lambda row: float(row["score"]))
    _write_validation_best(best_row, validation_best_path)

    if args.skip_final:
        return

    test_result = _evaluate_final_test(
        best_row,
        seq_sources,
        xgb_branch,
        result_suffix=args.result_suffix,
        workflow_eval=args.workflow_eval,
    )
    _write_final_test(best_row, test_result, final_test_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Narrow joint repro flow: fixed TRA structure + exact fund-XGB branch + explicit legacy anchor."
    )
    parser.add_argument("--seed-list", default="42,2026,3407", help="comma-separated deterministic stage1 seeds")
    parser.add_argument("--no-legacy-anchor", action="store_true", help="do not include archived exact current-best stage1 cache")
    parser.add_argument("--result-suffix", default="")
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--workflow-eval", action="store_true", help="use recorder-based evaluation instead of fast local backtest")
    main(parser.parse_args())