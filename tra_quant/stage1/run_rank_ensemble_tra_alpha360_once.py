from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
from pathlib import Path

import pandas as pd
import qlib
from qlib.backtest import backtest as normal_backtest
from qlib.constant import REG_CN
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.eva.alpha import calc_ic

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
STRATEGY_PROFILE_300 = "300"
STRATEGY_PROFILE_TRA_BEST_72 = "tra-best-72"
SORT_COLUMNS = [
    "validation_score",
    "validation_with_cost_ann_return",
    "validation_with_cost_ir",
    "validation_with_cost_mdd",
]


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


def _risk_tag(risk_degree: float) -> str:
    if risk_degree == 1.0:
        return "r10"
    return f"r{str(risk_degree).replace('.', '')}"


def _n_drop_tag(n_drop: int) -> str:
    return "" if n_drop == 1 else f"_drop{n_drop}"


def _search_space_for_profile(strategy_profile: str) -> dict:
    if strategy_profile == STRATEGY_PROFILE_300:
        return {
            "holds": SEARCH_HOLDS,
            "n_drops": SEARCH_N_DROPS,
            "risk_degrees": SEARCH_RISKS,
            "score_margins": tuple(score_margin for _, score_margin in mod_search_margins()),
        }
    if strategy_profile == STRATEGY_PROFILE_TRA_BEST_72:
        return {
            "holds": SEARCH_HOLDS,
            "n_drops": (1, 2, 3),
            "risk_degrees": (0.85, 0.95),
            "score_margins": tuple(score_margin for _, score_margin in mod_search_margins()),
        }
    raise ValueError(f"unsupported strategy_profile: {strategy_profile}")


def mod_search_margins() -> tuple[tuple[str, float], ...]:
    return (
        ("m000", 0.0),
        ("m025", 0.0025),
        ("m050", 0.005),
        ("m100", 0.010),
    )


def build_candidates(mod, *, strategy_profile: str = STRATEGY_PROFILE_300):
    search_space = _search_space_for_profile(strategy_profile)
    margin_values = set(search_space["score_margins"])
    candidates = []
    existing_by_name = getattr(mod, "STRATEGY_TRIAL_BY_NAME", {})
    for hold in search_space["holds"]:
        for n_drop in search_space["n_drops"]:
            for risk_degree in search_space["risk_degrees"]:
                for margin_tag, score_margin in mod._SEARCH_MARGINS:
                    if score_margin not in margin_values:
                        continue
                    trial_name = f"prac_{margin_tag}{_n_drop_tag(n_drop)}_hold{hold}_{_risk_tag(risk_degree)}"
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


def _effective_prefix(output_prefix: Path, *, num_shards: int, shard_index: int) -> Path:
    if num_shards <= 1:
        return output_prefix
    return output_prefix.with_name(f"{output_prefix.name}_shard{shard_index:02d}of{num_shards:02d}")


def _grid_path(output_prefix: Path) -> Path:
    return output_prefix.with_name(f"{output_prefix.name}_validation_grid.csv")


def _summary_path(output_prefix: Path) -> Path:
    return output_prefix.with_name(f"{output_prefix.name}_summary.json")


def _signal_ic(pred: pd.DataFrame, label: pd.DataFrame) -> tuple[float, float]:
    pred_s = pred.iloc[:, 0] if isinstance(pred, pd.DataFrame) else pred
    label_s = label.iloc[:, 0] if isinstance(label, pd.DataFrame) else label
    ic, rank_ic = calc_ic(pred_s, label_s)
    return float(ic.mean()), float(rank_ic.mean())


def load_existing_validation_rows(output_csv_path: Path):
    row_map = {}
    for path in (output_csv_path, VALID_GRID_PATH, LEGACY_VALID_GRID_PATH):
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


def _persist_validation_rows(valid_rows_out, output_csv_path: Path):
    if not valid_rows_out:
        return
    pd.DataFrame(valid_rows_out).sort_values(
        SORT_COLUMNS,
        ascending=[False, False, False, False],
    ).to_csv(output_csv_path, index=False)


def _evaluate_direct_signal(
    mod,
    strategy_trial: dict,
    pred: pd.DataFrame,
    label: pd.DataFrame,
    *,
    window_key: str,
    eval_segment: str,
    backtest_segment: str,
    eval_range: list[str] | tuple[str, str] | None = None,
    backtest_range: list[str] | tuple[str, str] | None = None,
) -> dict:
    if eval_range is not None:
        eval_pred = mod._slice_signal_to_range(pred, eval_range)
        eval_label = mod._slice_signal_to_range(label, eval_range)
    else:
        eval_pred = pred
        eval_label = label
    ic, rank_ic = _signal_ic(eval_pred, eval_label)

    task = mod.build_base_task(
        topk=mod.TOPK,
        n_drop=int(strategy_trial["n_drop"]),
        window_key=window_key,
        model_key="tra_lstm_base",
        handler_class="Alpha360",
        handler_module_path="qlib.contrib.data.handler",
        account=mod.ACCOUNT,
        risk_degree=float(strategy_trial["risk_degree"]),
        benchmark=mod.BENCHMARK,
        instruments=mod.INSTRUMENTS,
        handler_kwargs_extra=mod.H5_LABEL,
        strategy_class=strategy_trial["strategy_class"],
        strategy_module_path=strategy_trial["strategy_module_path"],
        strategy_kwargs_extra=strategy_trial["kwargs_extra"],
        eval_segment=eval_segment,
        backtest_segment=backtest_segment,
    )
    config = task["record"][2]["kwargs"]["config"]
    config["strategy"]["kwargs"]["signal"] = eval_pred
    if backtest_range is not None:
        config["backtest"]["start_time"] = backtest_range[0]
        config["backtest"]["end_time"] = backtest_range[1]

    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }
    portfolio_metric_dict, _ = normal_backtest(
        executor=executor_config,
        strategy=config["strategy"],
        **config["backtest"],
    )
    report_normal, _ = portfolio_metric_dict["1day"]
    with_cost = risk_analysis(report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq="1day")
    return {
        "recorder_id": None,
        "IC": ic,
        "Rank IC": rank_ic,
        "with_cost_ann_return": float(with_cost.loc["annualized_return", "risk"]),
        "with_cost_ir": float(with_cost.loc["information_ratio", "risk"]),
        "with_cost_mdd": float(with_cost.loc["max_drawdown", "risk"]),
    }


def _evaluate_validation_stability_direct(
    mod,
    strategy_trial: dict,
    pred: pd.DataFrame,
    label: pd.DataFrame,
    *,
    window_key: str,
) -> dict:
    segment_results = []
    for segment_name, segment_range in mod._validation_subsegments(window_key):
        segment_result = _evaluate_direct_signal(
            mod,
            strategy_trial,
            pred,
            label,
            window_key=window_key,
            eval_segment="valid",
            backtest_segment="valid",
            eval_range=segment_range,
            backtest_range=segment_range,
        )
        segment_results.append({
            "name": segment_name,
            "range": segment_range,
            **segment_result,
        })

    ann_values = pd.Series([float(item["with_cost_ann_return"]) for item in segment_results], dtype="float64")
    ir_values = pd.Series([float(item["with_cost_ir"]) for item in segment_results], dtype="float64")
    mdd_values = pd.Series([float(item["with_cost_mdd"]) for item in segment_results], dtype="float64")
    segment_mean_metrics = {
        "with_cost_ann_return": float(ann_values.mean()),
        "with_cost_ir": float(ir_values.mean()),
        "with_cost_mdd": float(mdd_values.mean()),
    }
    segment_mean_score = mod._objective(segment_mean_metrics)
    ann_std = float(ann_values.std(ddof=0))
    ann_spread = float(ann_values.max() - ann_values.min())
    worst_ann = float(ann_values.min())
    stability_penalty = (
        ann_std * mod.VALIDATION_STABILITY_WEIGHTS["ann_std"]
        + ann_spread * mod.VALIDATION_STABILITY_WEIGHTS["ann_spread"]
        + max(0.0, -worst_ann) * mod.VALIDATION_STABILITY_WEIGHTS["negative_worst_ann"]
    )
    segment_summary = "; ".join(
        f"{item['name']}[{item['range'][0]}:{item['range'][1]}]"
        f":ann={float(item['with_cost_ann_return']):.6f},ir={float(item['with_cost_ir']):.6f},mdd={float(item['with_cost_mdd']):.6f}"
        for item in segment_results
    )
    return {
        "score_mode": mod.VALIDATION_SCORE_MODE,
        "segment_count": len(segment_results),
        "segment_mean_score": segment_mean_score,
        "stable_penalty": stability_penalty,
        "segment_ann_std": ann_std,
        "segment_ann_spread": ann_spread,
        "segment_worst_ann_return": worst_ann,
        "segment_details": segment_summary,
        "score": segment_mean_score - stability_penalty,
    }


def _evaluate_strategy_direct(
    mod,
    strategy_trial: dict,
    pred: pd.DataFrame,
    label: pd.DataFrame,
    *,
    window_key: str,
) -> dict:
    result = _evaluate_direct_signal(
        mod,
        strategy_trial,
        pred,
        label,
        window_key=window_key,
        eval_segment="valid",
        backtest_segment="valid",
    )
    result["validation_recorder_id"] = None
    result["base_score"] = mod._objective(result)
    result.update(
        _evaluate_validation_stability_direct(
            mod,
            strategy_trial,
            pred,
            label,
            window_key=window_key,
        )
    )
    return result


def evaluate_or_reuse_validation_rows(
    mod,
    candidates,
    valid_pred,
    valid_label,
    validation_summary_path: Path,
    valid_rows: int,
    output_csv_path: Path,
    *,
    force_recompute: bool,
):
    existing_rows = {} if force_recompute else load_existing_validation_rows(output_csv_path)
    valid_rows_out = []
    reused_count = 0
    evaluated_count = 0

    for trial in candidates:
        cached_row = existing_rows.get(trial["trial_name"])
        if cached_row is not None:
            valid_rows_out.append(cached_row)
            reused_count += 1
            continue

        vr = _evaluate_strategy_direct(
            mod,
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
                "validation_summary_path": str(validation_summary_path),
                "valid_common_rows": int(valid_rows),
            }
        )
        evaluated_count += 1
        _persist_validation_rows(valid_rows_out, output_csv_path)
        print(
            f"evaluated {evaluated_count} new strategies, reused {reused_count}, "
            f"latest={trial['trial_name']} validation_score={valid_rows_out[-1]['validation_score']:.6f}"
        )

    valid_df = pd.DataFrame(valid_rows_out).sort_values(
        SORT_COLUMNS,
        ascending=[False, False, False, False],
    )
    return valid_df, reused_count, evaluated_count


def merge_shards(args):
    mod = load_module()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    frames = []
    shard_paths = []
    for shard_index in range(args.num_shards):
        shard_prefix = _effective_prefix(output_prefix, num_shards=args.num_shards, shard_index=shard_index)
        shard_path = _grid_path(shard_prefix)
        if not shard_path.exists():
            raise FileNotFoundError(f"missing shard grid: {shard_path}")
        shard_paths.append(str(shard_path))
        frames.append(pd.read_csv(shard_path))
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["strategy_trial"], keep="last")
    merged = merged.sort_values(SORT_COLUMNS, ascending=[False, False, False, False])
    output_csv_path = _grid_path(output_prefix)
    output_summary_path = _summary_path(output_prefix)
    output_csv_path.write_text(merged.to_csv(index=False), encoding="utf-8")
    summary = {
        "run_mode": "fast_validation_strategy_search_merged",
        "ensemble_method": "rank_ensemble",
        "num_shards": int(args.num_shards),
        "shard_grid_paths": shard_paths,
        "grid_rows": int(len(merged)),
        "strategy_profile": args.strategy_profile,
        "search_space": {
            key: list(value) for key, value in _search_space_for_profile(args.strategy_profile).items()
        },
        "candidate_count_total": int(len(build_candidates(mod, strategy_profile=args.strategy_profile))),
        "selection_rule": "validation_score, then ann, then ir, then mdd descending",
        "best_validation_selected": merged.iloc[0].to_dict() if not merged.empty else None,
        "top10_validation": merged.head(10).to_dict(orient="records"),
        "output_csv_path": str(output_csv_path),
    }
    output_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(output_summary_path)


def main():
    parser = argparse.ArgumentParser(description="Run the 3-seed TRA rank ensemble over the 300-strategy space.")
    parser.add_argument("--force-recompute", action="store_true", help="Ignore any existing validation grid rows and recompute all 300 strategies.")
    parser.add_argument("--validation-only", action="store_true", help="Run only the 300-strategy validation search and skip the final test-once step.")
    parser.add_argument("--validation-summary-path", default=str(VALIDATION_SUMMARY_PATH))
    parser.add_argument("--final-test-summary-path", default=str(FINAL_TEST_SUMMARY_PATH))
    parser.add_argument("--output-prefix", default=str(BASE_DIR / "rank_ensemble_3seed_300"))
    parser.add_argument("--strategy-trial", action="append", help="Evaluate only the named strategy trial. Repeat for more than one.")
    parser.add_argument(
        "--strategy-profile",
        choices=[STRATEGY_PROFILE_300, STRATEGY_PROFILE_TRA_BEST_72],
        default=STRATEGY_PROFILE_300,
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true", help="Merge shard CSV files for --output-prefix and --num-shards.")
    args = parser.parse_args()

    if args.merge_shards:
        merge_shards(args)
        return
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    mod = load_module()
    qlib.init(provider_uri=mod.PROVIDER_URI, region=REG_CN)
    output_prefix = _effective_prefix(
        Path(args.output_prefix).expanduser().resolve(),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path = _grid_path(output_prefix)
    output_summary_path = _summary_path(output_prefix)

    validation_summary_path = Path(args.validation_summary_path).expanduser().resolve()
    valid_cache_paths = _load_seed_cache_paths(validation_summary_path, "seed_runs_valid")
    valid_pred, valid_label, valid_rows = rank_ensemble_signal(mod, valid_cache_paths)

    candidates = build_candidates(mod, strategy_profile=args.strategy_profile)
    candidate_count_total = len(candidates)
    if args.strategy_trial:
        wanted = set(args.strategy_trial)
        candidates = [trial for trial in candidates if trial["trial_name"] in wanted]
        missing = sorted(wanted - {trial["trial_name"] for trial in candidates})
        if missing:
            raise ValueError(f"unknown strategy_trial values: {missing}")
    shard_candidates = [
        trial for index, trial in enumerate(candidates) if index % args.num_shards == args.shard_index
    ]
    if args.force_recompute:
        for path in (output_csv_path,):
            if path.exists():
                path.unlink()
    valid_df, reused_count, evaluated_count = evaluate_or_reuse_validation_rows(
        mod,
        shard_candidates,
        valid_pred,
        valid_label,
        validation_summary_path,
        valid_rows,
        output_csv_path,
        force_recompute=args.force_recompute,
    )
    best_row = valid_df.iloc[0]

    summary = {
        "run_mode": "fast_validation_strategy_search",
        "ensemble_method": "rank_ensemble",
        "rank_definition": "per-day cross-sectional percentile rank, averaged equally across the 3 seeds",
        "selection_rule": "search validation only; run test exactly once for the validation winner",
        "search_space": {
            key: list(value) for key, value in _search_space_for_profile(args.strategy_profile).items()
        },
        "strategy_profile": args.strategy_profile,
        "candidate_count_total": int(candidate_count_total),
        "candidate_filter": list(args.strategy_trial or []),
        "candidate_count_this_shard": int(len(shard_candidates)),
        "candidate_count": int(len(valid_df)),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "reused_validation_count": int(reused_count),
        "evaluated_validation_count": int(evaluated_count),
        "validation_summary_path": str(validation_summary_path),
        "validation_cache_paths": [str(path) for path in valid_cache_paths],
        "valid_common_rows": int(valid_rows),
        "best_validation_selected": best_row.to_dict(),
        "top5_validation": valid_df.head(5).to_dict(orient="records"),
        "output_csv_path": str(output_csv_path),
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
        all_candidates = build_candidates(mod, strategy_profile=args.strategy_profile)
        best_trial = next(trial for trial in all_candidates if trial["trial_name"] == best_row["strategy_trial"])
        test_result = _evaluate_direct_signal(
            mod,
            best_trial,
            test_pred,
            test_label,
            window_key="w1",
            eval_segment="test",
            backtest_segment="test",
        )
        summary["run_mode"] = "validation_and_test_once"
        summary["final_test_summary_path"] = str(final_test_summary_path)
        summary["test_cache_paths"] = [str(path) for path in test_cache_paths]
        summary["test_common_rows"] = int(test_rows)
        summary["selected_test_result"] = {
            "strategy_trial": best_trial["trial_name"],
            "test_score": float(mod._objective(test_result)),
            "test_IC": float(test_result["IC"]),
            "test_Rank_IC": float(test_result["Rank IC"]),
            "test_with_cost_ann_return": float(test_result["with_cost_ann_return"]),
            "test_with_cost_ir": float(test_result["with_cost_ir"]),
            "test_with_cost_mdd": float(test_result["with_cost_mdd"]),
        }

    output_csv_path.write_text(valid_df.to_csv(index=False), encoding="utf-8")
    output_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    if args.output_prefix == str(BASE_DIR / "rank_ensemble_3seed_300") and args.num_shards == 1:
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
        VALID_GRID_PATH.write_text(valid_df.to_csv(index=False), encoding="utf-8")
    print(output_summary_path)


if __name__ == "__main__":
    main()
