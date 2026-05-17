from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import torch

from tune_alpha360_alpha158_dnn_strict_fusion import (
    ALSTM_FINAL_TEST_PATH,
    ALSTM_VALIDATION_BEST_PATH,
    _align_signals,
    _evaluate_signal,
    _load_alstm_strategy_cfg,
    _load_signal_from_cache,
    _load_signal_from_recorder,
    _normalize_scores,
    _objective,
    _parse_summary_file,
    _persist,
    _write_fused_cache,
)


BASE_DIR = Path(__file__).resolve().parent

SOURCE_VALIDATION_BEST_PATH = BASE_DIR / "alpha360_alpha158_dnn_weight_refine_validation_best.txt"
SOURCE_FINAL_TEST_PATH = BASE_DIR / "alpha360_alpha158_dnn_weight_refine_final_test.txt"

VALIDATION_RESULTS_PATH = BASE_DIR / "alpha360_alpha158_dnn_trainable_alpha_validation_results.csv"
VALIDATION_BEST_PATH = BASE_DIR / "alpha360_alpha158_dnn_trainable_alpha_validation_best.txt"
FINAL_TEST_PATH = BASE_DIR / "alpha360_alpha158_dnn_trainable_alpha_final_test.txt"

FUSION_MODES = ["zscore", "rank_pct"]
TRAIN_OBJECTIVES = ["ic", "mse"]
INIT_TAB_WEIGHT = 0.15
LEARNING_RATE = 0.05
MAX_STEPS = 400
EARLY_STOP_ROUNDS = 50
LOSS_EPS = 1e-12


def _safe_logit(prob: float, eps: float = 1e-6) -> float:
    clipped = min(max(float(prob), eps), 1.0 - eps)
    return math.log(clipped / (1.0 - clipped))


def _label_series(label: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(label, pd.Series):
        return label.astype("float32")
    if "label" in label.columns:
        return label["label"].astype("float32")
    return label.iloc[:, 0].astype("float32")


def _date_slices(index: pd.MultiIndex) -> list[tuple[int, int]]:
    if not isinstance(index, pd.MultiIndex):
        raise TypeError("expected MultiIndex with datetime/instrument")
    datetimes = index.get_level_values("datetime")
    slices: list[tuple[int, int]] = []
    start = 0
    total = len(index)
    while start < total:
        current_dt = datetimes[start]
        end = start + 1
        while end < total and datetimes[end] == current_dt:
            end += 1
        if end - start >= 2:
            slices.append((start, end))
        start = end
    if not slices:
        raise ValueError("no valid per-date slices with at least two instruments")
    return slices


def _ic_loss(pred: torch.Tensor, target: torch.Tensor, slices: list[tuple[int, int]]) -> torch.Tensor:
    ic_terms = []
    for start, end in slices:
        pred_slice = pred[start:end]
        target_slice = target[start:end]
        pred_centered = pred_slice - pred_slice.mean()
        target_centered = target_slice - target_slice.mean()
        pred_scale = torch.sqrt(torch.mean(pred_centered.square()) + LOSS_EPS)
        target_scale = torch.sqrt(torch.mean(target_centered.square()) + LOSS_EPS)
        ic_terms.append(torch.mean(pred_centered * target_centered) / (pred_scale * target_scale))
    return -torch.stack(ic_terms).mean()


def _mse_loss(pred: torch.Tensor, target: torch.Tensor, _slices: list[tuple[int, int]]) -> torch.Tensor:
    return torch.mean((pred - target).square())


def _build_loss(loss_name: str):
    if loss_name == "ic":
        return _ic_loss
    if loss_name == "mse":
        return _mse_loss
    raise ValueError(f"unsupported train objective: {loss_name}")


def _train_global_alpha(
    seq_norm: pd.DataFrame,
    tab_norm: pd.DataFrame,
    label: pd.DataFrame | pd.Series,
    train_objective: str,
    init_tab_weight: float = INIT_TAB_WEIGHT,
    learning_rate: float = LEARNING_RATE,
    max_steps: int = MAX_STEPS,
    early_stop_rounds: int = EARLY_STOP_ROUNDS,
) -> dict:
    label_series = _label_series(label)
    valid_mask = seq_norm["score"].notna() & tab_norm["score"].notna() & label_series.notna()
    seq_norm = seq_norm.loc[valid_mask]
    tab_norm = tab_norm.loc[valid_mask]
    label_series = label_series.loc[valid_mask]
    if len(seq_norm) == 0:
        raise ValueError("no finite samples left after filtering seq/tab/label")

    seq_tensor = torch.tensor(seq_norm["score"].to_numpy(), dtype=torch.float32)
    tab_tensor = torch.tensor(tab_norm["score"].to_numpy(), dtype=torch.float32)
    label_tensor = torch.tensor(label_series.to_numpy(), dtype=torch.float32)
    slices = _date_slices(seq_norm.index)
    loss_fn = _build_loss(train_objective)

    alpha_param = torch.nn.Parameter(torch.tensor(_safe_logit(init_tab_weight), dtype=torch.float32))
    optimizer = torch.optim.Adam([alpha_param], lr=learning_rate)

    best_loss = float("inf")
    best_state = alpha_param.detach().clone()
    best_step = 0
    stop_rounds = 0

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        tab_weight = torch.sigmoid(alpha_param)
        seq_weight = 1.0 - tab_weight
        fused = seq_tensor * seq_weight + tab_tensor * tab_weight
        loss = loss_fn(fused, label_tensor, slices)
        if not torch.isfinite(loss):
            raise ValueError(f"non-finite loss for objective={train_objective}")
        loss.backward()
        optimizer.step()

        current_loss = float(loss.detach().cpu().item())
        if current_loss + 1e-9 < best_loss:
            best_loss = current_loss
            best_state = alpha_param.detach().clone()
            best_step = step
            stop_rounds = 0
        else:
            stop_rounds += 1
            if stop_rounds >= early_stop_rounds:
                break

    if best_step <= 0 or not math.isfinite(best_loss):
        raise ValueError(f"failed to obtain a finite training loss for objective={train_objective}")

    final_tab_weight = float(torch.sigmoid(best_state).cpu().item())
    return {
        "tab_weight": final_tab_weight,
        "seq_weight": 1.0 - final_tab_weight,
        "train_loss": best_loss,
        "train_steps": best_step,
    }


def _write_validation_best(best_row: dict):
    strict_validation = _parse_summary_file(ALSTM_VALIDATION_BEST_PATH)
    lines = [
        f"window_key={strict_validation['window_key']}",
        f"train={strict_validation['train']}",
        f"valid={strict_validation['valid']}",
        f"test={strict_validation['test']}",
        f"selection_scope=trainable_global_alpha",
        f"base_model_trial={best_row['base_model_trial']}",
        f"base_strategy_trial={best_row['base_strategy_trial']}",
        f"tabular_trial={best_row['tabular_trial']}",
        f"tabular_handler_class={best_row['tabular_handler_class']}",
        f"tabular_model_key={best_row['tabular_model_key']}",
        f"fusion_mode={best_row['fusion_mode']}",
        f"train_objective={best_row['train_objective']}",
        f"seq_weight={best_row['seq_weight']}",
        f"tab_weight={best_row['tab_weight']}",
        f"train_loss={best_row['train_loss']}",
        f"train_steps={best_row['train_steps']}",
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


def _write_final_test(best_row: dict, test_result: dict, incumbent_test: dict):
    strict_validation = _parse_summary_file(ALSTM_VALIDATION_BEST_PATH)
    improved = float(test_result["with_cost_ann_return"]) > float(incumbent_test["test_with_cost_ann_return"])
    lines = [
        f"window_key={strict_validation['window_key']}",
        f"train={strict_validation['train']}",
        f"valid={strict_validation['valid']}",
        f"test={strict_validation['test']}",
        f"selection_scope=trainable_global_alpha",
        f"test_improves_incumbent={improved}",
        f"base_model_trial={best_row['base_model_trial']}",
        f"base_strategy_trial={best_row['base_strategy_trial']}",
        f"tabular_trial={best_row['tabular_trial']}",
        f"tabular_handler_class={best_row['tabular_handler_class']}",
        f"tabular_model_key={best_row['tabular_model_key']}",
        f"fusion_mode={best_row['fusion_mode']}",
        f"train_objective={best_row['train_objective']}",
        f"seq_weight={best_row['seq_weight']}",
        f"tab_weight={best_row['tab_weight']}",
        f"train_loss={best_row['train_loss']}",
        f"train_steps={best_row['train_steps']}",
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
        f"incumbent_test_with_cost_ann_return={incumbent_test['test_with_cost_ann_return']}",
        f"incumbent_test_with_cost_ir={incumbent_test['test_with_cost_ir']}",
        f"incumbent_test_with_cost_mdd={incumbent_test['test_with_cost_mdd']}",
        f"tabular_test_result_path={test_result['tabular_test_result_path']}",
        f"tabular_test_recorder_id={test_result['tabular_test_recorder_id']}",
    ]
    FINAL_TEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    alstm_validation = _parse_summary_file(ALSTM_VALIDATION_BEST_PATH)
    alstm_final = _parse_summary_file(ALSTM_FINAL_TEST_PATH)
    source_validation = _parse_summary_file(SOURCE_VALIDATION_BEST_PATH)
    source_test = _parse_summary_file(SOURCE_FINAL_TEST_PATH)
    strategy_cfg_valid = _load_alstm_strategy_cfg(alstm_validation, eval_segment="valid")
    strategy_cfg_test = _load_alstm_strategy_cfg(alstm_validation, eval_segment="test")

    seq_valid_pred, seq_valid_label = _load_signal_from_cache(alstm_validation["validation_cache_path"])
    tab_valid_pred, tab_valid_label = _load_signal_from_recorder(source_validation["tabular_validation_recorder_id"])
    seq_valid_pred, tab_valid_pred, valid_label = _align_signals(
        seq_valid_pred, seq_valid_label, tab_valid_pred, tab_valid_label
    )

    rows: list[dict] = [
        {
            "trial_name": "trainable_alpha_baseline_alstm_only",
            "base_model_trial": alstm_validation["model_trial"],
            "base_strategy_trial": alstm_validation["strategy_trial"],
            "tabular_trial": "none",
            "tabular_handler_class": "none",
            "tabular_model_key": "none",
            "fusion_mode": "baseline",
            "train_objective": "none",
            "seq_weight": 1.0,
            "tab_weight": 0.0,
            "train_loss": pd.NA,
            "train_steps": pd.NA,
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
        },
        {
            "trial_name": "trainable_alpha_reference_fixed_weight",
            "base_model_trial": alstm_validation["model_trial"],
            "base_strategy_trial": alstm_validation["strategy_trial"],
            "tabular_trial": source_validation["tabular_trial"],
            "tabular_handler_class": source_validation["tabular_handler_class"],
            "tabular_model_key": source_validation["tabular_model_key"],
            "fusion_mode": source_validation["fusion_mode"],
            "train_objective": "fixed_reference",
            "seq_weight": float(source_validation["seq_weight"]),
            "tab_weight": float(source_validation["tab_weight"]),
            "train_loss": pd.NA,
            "train_steps": pd.NA,
            "status": "success",
            "IC": float(source_validation["validation_IC"]),
            "Rank IC": float(source_validation["validation_Rank_IC"]),
            "with_cost_ann_return": float(source_validation["validation_with_cost_ann_return"]),
            "with_cost_ir": float(source_validation["validation_with_cost_ir"]),
            "with_cost_mdd": float(source_validation["validation_with_cost_mdd"]),
            "score": float(source_validation["validation_score"]),
            "validation_cache_path": str(source_validation["validation_cache_path"]),
            "validation_recorder_id": source_validation["validation_recorder_id"],
            "tabular_validation_result_path": str(source_validation["tabular_validation_result_path"]),
            "tabular_validation_recorder_id": source_validation["tabular_validation_recorder_id"],
        },
    ]

    for fusion_mode in FUSION_MODES:
        seq_norm = _normalize_scores(seq_valid_pred, fusion_mode)
        tab_norm = _normalize_scores(tab_valid_pred, fusion_mode)
        for train_objective in TRAIN_OBJECTIVES:
            row = {
                "trial_name": f"trainable_alpha_{fusion_mode}_{train_objective}",
                "base_model_trial": alstm_validation["model_trial"],
                "base_strategy_trial": alstm_validation["strategy_trial"],
                "tabular_trial": source_validation["tabular_trial"],
                "tabular_handler_class": source_validation["tabular_handler_class"],
                "tabular_model_key": source_validation["tabular_model_key"],
                "fusion_mode": fusion_mode,
                "train_objective": train_objective,
                "status": "running",
            }
            try:
                trained = _train_global_alpha(
                    seq_norm=seq_norm,
                    tab_norm=tab_norm,
                    label=valid_label,
                    train_objective=train_objective,
                )
                fused_pred = seq_norm.mul(trained["seq_weight"]).add(
                    tab_norm.mul(trained["tab_weight"]), fill_value=0.0
                )
                validation_cache_path = _write_fused_cache(row["trial_name"], fused_pred, valid_label)
                metrics = _evaluate_signal(
                    recorder_name=row["trial_name"],
                    pred=fused_pred,
                    label=valid_label,
                    strategy_cfg=strategy_cfg_valid,
                    experiment_name="alpha360_alpha158_dnn_trainable_alpha_validation_eval",
                )
                row.update(
                    {
                        "status": "success",
                        "seq_weight": trained["seq_weight"],
                        "tab_weight": trained["tab_weight"],
                        "train_loss": trained["train_loss"],
                        "train_steps": trained["train_steps"],
                        "IC": metrics["IC"],
                        "Rank IC": metrics["Rank IC"],
                        "with_cost_ann_return": metrics["with_cost_ann_return"],
                        "with_cost_ir": metrics["with_cost_ir"],
                        "with_cost_mdd": metrics["with_cost_mdd"],
                        "score": metrics["score"],
                        "validation_cache_path": str(validation_cache_path),
                        "validation_recorder_id": metrics["recorder_id"],
                        "tabular_validation_result_path": str(source_validation["tabular_validation_result_path"]),
                        "tabular_validation_recorder_id": source_validation["tabular_validation_recorder_id"],
                    }
                )
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = repr(exc)[:500]
            rows = [existing for existing in rows if existing.get("trial_name") != row["trial_name"]] + [row]
            _persist(rows, VALIDATION_RESULTS_PATH)

    success_rows = [row for row in rows if row["status"] == "success"]
    if not success_rows:
        raise RuntimeError("no successful trainable-alpha rows")
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

    if best_row["train_objective"] == "fixed_reference":
        test_result = {
            "cache_path": source_test["test_cache_path"],
            "recorder_id": source_test["test_recorder_id"],
            "IC": source_test["test_IC"],
            "Rank IC": source_test["test_Rank_IC"],
            "with_cost_ann_return": source_test["test_with_cost_ann_return"],
            "with_cost_ir": source_test["test_with_cost_ir"],
            "with_cost_mdd": source_test["test_with_cost_mdd"],
            "tabular_test_result_path": source_test["tabular_test_result_path"],
            "tabular_test_recorder_id": source_test["tabular_test_recorder_id"],
        }
        _write_final_test(best_row, test_result, alstm_final)
        return

    seq_test_pred, seq_test_label = _load_signal_from_cache(alstm_final["test_cache_path"])
    tab_test_pred, tab_test_label = _load_signal_from_recorder(source_test["tabular_test_recorder_id"])
    seq_test_pred, tab_test_pred, test_label = _align_signals(
        seq_test_pred, seq_test_label, tab_test_pred, tab_test_label
    )
    seq_test_norm = _normalize_scores(seq_test_pred, best_row["fusion_mode"])
    tab_test_norm = _normalize_scores(tab_test_pred, best_row["fusion_mode"])
    fused_test_pred = seq_test_norm.mul(best_row["seq_weight"]).add(
        tab_test_norm.mul(best_row["tab_weight"]), fill_value=0.0
    )
    test_cache_path = _write_fused_cache(f"{best_row['trial_name']}_final_test", fused_test_pred, test_label)
    test_metrics = _evaluate_signal(
        recorder_name=f"{best_row['trial_name']}_final_test",
        pred=fused_test_pred,
        label=test_label,
        strategy_cfg=strategy_cfg_test,
        experiment_name="alpha360_alpha158_dnn_trainable_alpha_final_test_eval",
    )
    test_metrics.update(
        {
            "cache_path": str(test_cache_path),
            "tabular_test_result_path": str(source_test["tabular_test_result_path"]),
            "tabular_test_recorder_id": source_test["tabular_test_recorder_id"],
        }
    )
    _write_final_test(best_row, test_metrics, alstm_final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a frozen-branch global alpha for Alpha360(ALSTM) + Alpha158(DNN).")
    parser.parse_args()
    main()
