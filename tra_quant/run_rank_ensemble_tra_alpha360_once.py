from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
from pathlib import Path

import pandas as pd

from tra_local_helpers import localize_value

logging.disable(logging.CRITICAL)

BASE_DIR = Path(__file__).resolve().parent
MODULE_PATH = BASE_DIR / "tune_tra_alpha360_global.py"
SUMMARY_PATH = BASE_DIR / "rank_ensemble_3seed_300_summary.json"
VALID_GRID_PATH = BASE_DIR / "rank_ensemble_3seed_300_validation_grid.csv"
LEGACY_VALID_GRID_PATH = BASE_DIR / "rank_ensemble_3seed_validation_grid.csv"
VALIDATION_SUMMARY_PATH = BASE_DIR / "current_best_multiseed_ensemble_validation.txt"
FINAL_TEST_SUMMARY_PATH = BASE_DIR / "current_best_multiseed_ensemble_final_test.txt"

SEARCH_HOLDS = (3, 4, 5)
SEARCH_N_DROPS = (1, 2, 3, 4, 5)
SEARCH_RISKS = (0.6, 0.7, 0.85, 0.95, 1.0)


def load_module():
    spec = importlib.util.spec_from_file_location("tune_tra_alpha360_global_rank_ensemble_once", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_summary(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def _load_seed_cache_paths(summary_path: Path, seed_runs_key: str) -> list[Path]:
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary: {summary_path}")

    summary = _parse_summary(summary_path)
    seed_runs_raw = summary.get(seed_runs_key)
    if not seed_runs_raw:
        raise KeyError(f"{seed_runs_key} not found in {summary_path}")

    seed_runs = localize_value(ast.literal_eval(seed_runs_raw))
    cache_paths: list[Path] = []
    for seed_run in seed_runs:
        cache_path = Path(seed_run["cache_path"])
        if not cache_path.exists():
            raise FileNotFoundError(f"missing cache referenced by {summary_path}: {cache_path}")
        cache_paths.append(cache_path)

    if not cache_paths:
        raise ValueError(f"no cache paths found in {summary_path} via {seed_runs_key}")

    return cache_paths


def rank_ensemble_signal(mod, cache_paths):
    preds = []
    labels = []
    common_index = None
    for cache_path in cache_paths:
        pred, label = mod._load_signal(cache_path)
        common_index = pred.index if common_index is None else common_index.intersection(pred.index)
        preds.append(pred)
        labels.append(label)

    preds = [pred.loc[common_index].sort_index() for pred in preds]
    labels = [label.loc[common_index].sort_index() for label in labels]
    ranked_preds = [pred.groupby(level=0, group_keys=False).rank(method="average", pct=True) for pred in preds]
    ensemble_pred = sum(ranked_preds) / len(ranked_preds)
    return ensemble_pred, labels[0].copy(), len(common_index)


def build_candidates(mod):
    candidates = []
    existing_by_name = getattr(mod, "STRATEGY_TRIAL_BY_NAME", {})
    for hold in SEARCH_HOLDS:
        for n_drop in SEARCH_N_DROPS:
            for risk_degree in SEARCH_RISKS:
                for margin_tag, score_margin in mod._SEARCH_MARGINS:
                    trial_name = f"prac_{margin_tag}{mod._n_drop_tag(n_drop)}_hold{hold}_{mod._risk_tag(risk_degree)}"
                    trial = existing_by_name.get(trial_name)
                    if trial is None:
                        trial = mod._prac(
                            trial_name,
                            n_drop,
                            risk_degree,
                            score_margin,
                            hold,
                        )
                    candidates.append(trial)
    return candidates


def _can_reuse_existing_validation_rows(validation_summary_path: Path, valid_rows: int) -> bool:
    if not SUMMARY_PATH.exists():
        return False

    try:
        summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    if summary.get("validation_summary_path") != str(validation_summary_path):
        return False
    if int(summary.get("valid_common_rows", -1)) != int(valid_rows):
        return False
    return True


def load_existing_validation_rows(validation_summary_path: Path, valid_rows: int):
    if not _can_reuse_existing_validation_rows(validation_summary_path, valid_rows):
        return {}

    row_map = {}
    for path in (VALID_GRID_PATH, LEGACY_VALID_GRID_PATH):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or "strategy_trial" not in df.columns:
            continue
        for row in df.to_dict(orient="records"):
            strategy_trial = row.get("strategy_trial")
            if not strategy_trial or strategy_trial in row_map:
                continue
            row_map[strategy_trial] = row
    return row_map


def _persist_validation_rows(valid_rows_out):
    if not valid_rows_out:
        return
    pd.DataFrame(valid_rows_out).sort_values(
        ["validation_score", "validation_with_cost_ann_return", "validation_with_cost_ir"],
        ascending=[False, False, False],
    ).to_csv(VALID_GRID_PATH, index=False)


def evaluate_or_reuse_validation_rows(mod, candidates, valid_pred, valid_label, validation_summary_path: Path, valid_rows: int):
    existing_rows = load_existing_validation_rows(validation_summary_path, valid_rows)
    valid_rows_out = []
    reused_count = 0
    evaluated_count = 0

    for trial in candidates:
        cached_row = existing_rows.get(trial["trial_name"])
        if cached_row is not None:
            valid_rows_out.append(cached_row)
            reused_count += 1
            continue

        vr = mod._evaluate_strategy(
            "rank_ensemble_seed42_2026_3407",
            trial,
            valid_pred,
            valid_label,
            window_key="w1",
        )
        valid_rows_out.append(
            {
                "strategy_trial": trial["trial_name"],
                "hold_thresh": trial["kwargs_extra"]["hold_thresh"],
                "score_margin": trial["kwargs_extra"]["score_margin"],
                "n_drop": trial["n_drop"],
                "risk_degree": trial["risk_degree"],
                "validation_score": float(vr["score"]),
                "validation_base_score": float(vr["base_score"]),
                "validation_segment_mean_score": float(vr["segment_mean_score"]),
                "validation_stable_penalty": float(vr["stable_penalty"]),
                "validation_with_cost_ann_return": float(vr["with_cost_ann_return"]),
                "validation_with_cost_ir": float(vr["with_cost_ir"]),
                "validation_with_cost_mdd": float(vr["with_cost_mdd"]),
                "validation_IC": float(vr["IC"]),
                "validation_Rank_IC": float(vr["Rank IC"]),
                "segment_details": vr["segment_details"],
            }
        )
        evaluated_count += 1
        _persist_validation_rows(valid_rows_out)
        print(
            f"evaluated {evaluated_count} new strategies, reused {reused_count}, "
            f"latest={trial['trial_name']} validation_score={valid_rows_out[-1]['validation_score']:.6f}"
        )

    valid_df = pd.DataFrame(valid_rows_out).sort_values(
        ["validation_score", "validation_with_cost_ann_return", "validation_with_cost_ir"],
        ascending=[False, False, False],
    )
    return valid_df, reused_count, evaluated_count


def main():
    parser = argparse.ArgumentParser(description="Run the 3-seed TRA rank ensemble over the 300-strategy space.")
    parser.add_argument("--force-recompute", action="store_true", help="Ignore any existing validation grid rows and recompute all 300 strategies.")
    parser.add_argument("--validation-only", action="store_true", help="Run only the 300-strategy validation search and skip the final test-once step.")
    parser.add_argument("--validation-summary-path", default=str(VALIDATION_SUMMARY_PATH))
    parser.add_argument("--final-test-summary-path", default=str(FINAL_TEST_SUMMARY_PATH))
    args = parser.parse_args()

    mod = load_module()
    validation_summary_path = Path(args.validation_summary_path).expanduser().resolve()
    valid_cache_paths = _load_seed_cache_paths(validation_summary_path, "seed_runs_valid")
    valid_pred, valid_label, valid_rows = rank_ensemble_signal(mod, valid_cache_paths)

    candidates = build_candidates(mod)
    if args.force_recompute:
        for path in (VALID_GRID_PATH, LEGACY_VALID_GRID_PATH):
            if path.exists():
                path.unlink()
    valid_df, reused_count, evaluated_count = evaluate_or_reuse_validation_rows(
        mod,
        candidates,
        valid_pred,
        valid_label,
        validation_summary_path,
        valid_rows,
    )
    best_row = valid_df.iloc[0]

    summary = {
        "ensemble_method": "rank_ensemble",
        "rank_definition": "per-day cross-sectional percentile rank, averaged equally across the 3 seeds",
        "selection_rule": "search validation only; run test exactly once for the validation winner",
        "search_space": {
            "holds": list(SEARCH_HOLDS),
            "n_drops": list(SEARCH_N_DROPS),
            "risk_degrees": list(SEARCH_RISKS),
            "score_margins": [score_margin for _, score_margin in mod._SEARCH_MARGINS],
        },
        "candidate_count": int(len(valid_df)),
        "reused_validation_count": int(reused_count),
        "evaluated_validation_count": int(evaluated_count),
        "validation_summary_path": str(validation_summary_path),
        "validation_cache_paths": [str(path) for path in valid_cache_paths],
        "valid_common_rows": int(valid_rows),
        "best_validation_selected": best_row.to_dict(),
        "top5_validation": valid_df.head(5).to_dict(orient="records"),
    }

    if args.validation_only:
        summary["run_mode"] = "validation_only"
        summary["final_test_summary_path"] = None
        summary["test_cache_paths"] = None
        summary["test_common_rows"] = None
        summary["selected_test_result"] = None
        summary["test_pending_reason"] = "validation_only_requested"
    else:
        final_test_summary_path = Path(args.final_test_summary_path).expanduser().resolve()
        test_cache_paths = _load_seed_cache_paths(final_test_summary_path, "seed_runs_test")
        test_pred, test_label, test_rows = rank_ensemble_signal(mod, test_cache_paths)
        best_trial = next(trial for trial in candidates if trial["trial_name"] == best_row["strategy_trial"])
        test_result = mod._evaluate_signal(
            "rank_ensemble_seed42_2026_3407",
            best_trial,
            test_pred,
            test_label,
            window_key="w1",
            eval_segment="test",
            backtest_segment="test",
            recorder_name=f"rank_ensemble_seed42_2026_3407__{best_trial['trial_name']}__test_once",
        )
        summary["run_mode"] = "validation_and_test_once"
        summary["final_test_summary_path"] = str(final_test_summary_path)
        summary["test_cache_paths"] = [str(path) for path in test_cache_paths]
        summary["test_common_rows"] = int(test_rows)
        summary["selected_test_result"] = {
            "test_IC": float(test_result["IC"]),
            "test_Rank_IC": float(test_result["Rank IC"]),
            "test_with_cost_ann_return": float(test_result["with_cost_ann_return"]),
            "test_with_cost_ir": float(test_result["with_cost_ir"]),
            "test_with_cost_mdd": float(test_result["with_cost_mdd"]),
        }

    VALID_GRID_PATH.write_text(valid_df.to_csv(index=False), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()