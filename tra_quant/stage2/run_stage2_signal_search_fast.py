from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd

import tune_alpha360_tra_fundamental_xgb_weight_refine as stage2


logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp" / "stage2_signal_fast"
DEFAULT_XGB_VALID_CACHE = (
    BASE_DIR
    / "xgb_cache"
    / "direct_cache_xgb_w1_xgb_base_TechAlpha158A3DFundamental_k5_d1_a150000_r07_"
    "xgb03_techA3d_fund_rankpct_base_s240_tra_fund_rankpct_anchor_valid_"
    "tra_best_stage2_targeted_20260613_1.pkl"
)

SORT_COLUMNS = ["score", "with_cost_ann_return", "with_cost_ir", "with_cost_mdd"]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _effective_prefix(output_prefix: Path, *, num_shards: int, shard_index: int) -> Path:
    if num_shards <= 1:
        return output_prefix
    return output_prefix.with_name(f"{output_prefix.name}_shard{shard_index:02d}of{num_shards:02d}")


def _grid_path(output_prefix: Path) -> Path:
    return output_prefix.with_name(f"{output_prefix.name}_validation_grid.csv")


def _summary_path(output_prefix: Path) -> Path:
    return output_prefix.with_name(f"{output_prefix.name}_summary.json")


def _load_existing_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty or "trial_name" not in df.columns:
        return {}
    return {str(row["trial_name"]): row for row in df.to_dict(orient="records")}


def _persist(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.sort_values(SORT_COLUMNS, ascending=[False, False, False, False], na_position="last").to_csv(
        path, index=False
    )


def _load_rank_ensemble_validation(
    result_suffix: str,
    *,
    validation_summary_path: Path | None = None,
    repro_json_path: Path | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    runner = stage2._load_rank_ensemble_runner()
    tra_mod = runner.load_module()
    validation_summary_path = (
        validation_summary_path.expanduser().resolve()
        if validation_summary_path is not None
        else stage2.TRA_BEST_VALIDATION_SUMMARY_PATH.resolve()
    )
    repro_json_path = (
        repro_json_path.expanduser().resolve()
        if repro_json_path is not None
        else stage2.TRA_BEST_REPRO_JSON_PATH.resolve()
    )
    payload = json.loads(repro_json_path.read_text(encoding="utf-8"))
    validation_wrapper = stage2._parse_summary_file(validation_summary_path)
    valid_cache_paths = runner._load_seed_cache_paths(validation_summary_path, "seed_runs_valid")
    valid_pred, valid_label, valid_common_rows = runner.rank_ensemble_signal(tra_mod, valid_cache_paths)
    if int(valid_common_rows) != int(payload["valid_common_rows"]):
        raise RuntimeError(f"rank-ensemble valid rows changed: {valid_common_rows} != {payload['valid_common_rows']}")

    strategy_trial = stage2._rank_ensemble_strategy_from_payload(payload)
    tra_validation = {
        "window_key": validation_wrapper["window_key"],
        "train": validation_wrapper["train"],
        "valid": validation_wrapper["valid"],
        "test": validation_wrapper["test"],
        "model_trial": "rank_ensemble_3seed_tra72_old6",
        "model_key": "tra_rank_ensemble",
        "handler_kwargs_extra": stage2.TRA_RANK_ENSEMBLE_HANDLER_KWARGS,
        "strategy_trial": strategy_trial["trial_name"],
        "strategy_class": strategy_trial["strategy_class"],
        "strategy_module_path": strategy_trial["strategy_module_path"],
        "n_drop": strategy_trial["n_drop"],
        "risk_degree": strategy_trial["risk_degree"],
        "kwargs_extra": strategy_trial["kwargs_extra"],
        "stage1_source": "rank_ensemble",
        "stage1_repro_json_path": str(repro_json_path),
        "stage1_validation_summary_path": str(validation_summary_path),
        "stage1_validation_seed_cache_paths": [str(path) for path in valid_cache_paths],
        "stage1_valid_common_rows": valid_common_rows,
    }
    return tra_validation, valid_pred.sort_index(), valid_label.sort_index()


def _fundamental_composites(index) -> dict[str, pd.DataFrame]:
    fund = pd.read_parquet(stage2.FUNDAMENTAL_FEATURE_PATH)
    if isinstance(fund.columns, pd.MultiIndex):
        fund.columns = fund.columns.get_level_values(-1)
    fund = fund.loc[index.intersection(fund.index).sort_values()]

    def cols(names):
        return [name for name in names if name in fund.columns]

    def comp(*, plus=(), minus=()):
        parts = []
        for col in cols(plus):
            parts.append(fund[col].astype(float))
        for col in cols(minus):
            parts.append(-fund[col].astype(float))
        if parts:
            score = pd.concat(parts, axis=1).mean(axis=1).fillna(0.0)
        else:
            score = pd.Series(0.0, index=fund.index)
        return pd.DataFrame({"score": score}, index=fund.index)

    return {
        "value_quality": comp(
            plus=["dv_ttm", "dv_ratio", "fi_roe", "fi_roe_dt", "fi_netprofit_margin", "fi_ocf_yoy", "cf_n_cashflow_act"],
            minus=["pe_ttm", "pe", "pb", "ps_ttm", "ps", "fi_debt_to_assets"],
        ),
        "growth_quality": comp(
            plus=[
                "fi_q_sales_yoy",
                "fi_q_op_qoq",
                "fi_ocf_yoy",
                "fi_tr_yoy",
                "fi_or_yoy",
                "fi_assets_yoy",
                "fi_eqt_yoy",
                "fi_roe",
            ],
            minus=["fi_debt_to_assets"],
        ),
        "cheap_dividend": comp(
            plus=["dv_ttm", "dv_ratio", "fi_bps", "cf_n_cashflow_act"],
            minus=["pe_ttm", "pb", "ps_ttm", "total_mv"],
        ),
        "cash_quality": comp(
            plus=["fi_ocf_yoy", "cf_n_cashflow_act", "fi_netprofit_margin", "fi_roe", "fi_assets_turn"],
            minus=["fi_debt_to_assets"],
        ),
    }


def _build_signal_candidates(
    seq_pred: pd.DataFrame,
    seq_label: pd.DataFrame,
    *,
    signal_profile: str,
    xgb_validation_cache_path: Path,
) -> list[dict]:
    candidates = [{"signal_name": "baseline_seq", "kind": "baseline", "pred": seq_pred, "label": seq_label}]

    if signal_profile in {"fundamental-small", "fundamental-filter-fine", "all-small"}:
        composites = _fundamental_composites(seq_pred.index)
        for recipe_name, raw in composites.items():
            common = seq_pred.index.intersection(raw.index).intersection(seq_label.index).sort_values()
            seq = seq_pred.loc[common]
            label = seq_label.loc[common]
            seq_z = stage2._normalize_scores(seq, "zscore")
            raw = raw.loc[common]
            raw_z = stage2._normalize_scores(raw, "zscore")
            raw_rank = stage2._normalize_scores(raw, "rank_pct")
            if signal_profile == "fundamental-filter-fine":
                linear_weights = ()
                filter_specs = ()
                if recipe_name == "value_quality":
                    filter_specs = (
                        (low_q, penalty)
                        for low_q in (0.16, 0.18, 0.20, 0.22, 0.24)
                        for penalty in (0.10, 0.125, 0.15, 0.175, 0.20, 0.25)
                    )
                elif recipe_name == "growth_quality":
                    filter_specs = (
                        (low_q, penalty)
                        for low_q in (0.08, 0.10, 0.12, 0.15, 0.18, 0.20)
                        for penalty in (0.08, 0.10, 0.125, 0.15)
                    )
                elif recipe_name == "cash_quality":
                    linear_weights = (0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)
            else:
                linear_weights = (0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05)
                filter_specs = ((0.10, 0.10), (0.15, 0.10), (0.20, 0.10), (0.20, 0.15), (0.30, 0.10))

            for weight in linear_weights:
                pred = seq_z.mul(1.0 - weight).add(raw_z.mul(weight), fill_value=0.0)
                candidates.append(
                    {
                        "signal_name": f"{recipe_name}_z_tw{str(weight).replace('.', '')}",
                        "kind": recipe_name,
                        "tab_weight": weight,
                        "pred": pred,
                        "label": label,
                    }
                )
            for low_q, penalty in filter_specs:
                pred = seq_z.copy()
                score = pred["score"].copy()
                score = score - (raw_rank["score"] <= low_q).astype(float) * penalty
                pred["score"] = score
                candidates.append(
                    {
                        "signal_name": f"{recipe_name}_filter_lq{str(low_q).replace('.', '')}_p{str(penalty).replace('.', '')}",
                        "kind": f"{recipe_name}_filter",
                        "low_q": low_q,
                        "penalty": penalty,
                        "pred": pred,
                        "label": label,
                    }
                )

    if signal_profile in {"xgb-small", "all-small"}:
        if not xgb_validation_cache_path.exists():
            raise FileNotFoundError(f"missing XGB validation cache: {xgb_validation_cache_path}")
        tab_pred, tab_label = stage2._load_signal_from_cache(xgb_validation_cache_path)
        seq, tab, label = stage2._align_signals(seq_pred, seq_label, tab_pred, tab_label)
        seq_z = stage2._normalize_scores(seq, "zscore")
        tab_z = stage2._normalize_scores(tab, "zscore")
        tab_rank = stage2._normalize_scores(tab, "rank_pct")
        for weight in (0.005, 0.01, 0.02, 0.03, 0.05):
            pred = seq_z.mul(1.0 - weight).add(tab_z.mul(weight), fill_value=0.0)
            candidates.append(
                {
                    "signal_name": f"xgb_z_tw{str(weight).replace('.', '')}",
                    "kind": "xgb_linear",
                    "tab_weight": weight,
                    "pred": pred,
                    "label": label,
                }
            )
        for low_q, penalty in ((0.15, 0.10), (0.20, 0.10), (0.20, 0.25), (0.30, 0.25)):
            pred = seq_z.copy()
            score = pred["score"].copy()
            score = score - (tab_rank["score"] <= low_q).astype(float) * penalty
            pred["score"] = score
            candidates.append(
                {
                    "signal_name": f"xgb_filter_lq{str(low_q).replace('.', '')}_p{str(penalty).replace('.', '')}",
                    "kind": "xgb_filter",
                    "low_q": low_q,
                    "penalty": penalty,
                    "pred": pred,
                    "label": label,
                }
            )

    return candidates


def _strategy_trials(profile: str) -> list[dict]:
    if profile == "hold4r08":
        return [stage2._prac("prac_m000_hold4_r08", 0.8, 0.0, 4, n_drop=1)]
    if profile == "n_drop1_core":
        trials = []
        for margin_tag, score_margin in (("m000", 0.0), ("m025", 0.0025), ("m050", 0.005)):
            for hold_thresh in (3, 4, 5, 7):
                for risk_degree in (0.80, 0.85, 0.90, 0.95):
                    trials.append(
                        stage2._prac(
                            f"prac_{margin_tag}_hold{hold_thresh}_{stage2._risk_tag(risk_degree)}",
                            risk_degree,
                            score_margin,
                            hold_thresh,
                            n_drop=1,
                        )
                    )
        return trials
    raise ValueError(f"unknown strategy profile: {profile}")


def run_shard(args: argparse.Namespace) -> None:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    output_prefix = _effective_prefix(args.output_prefix.resolve(), num_shards=args.num_shards, shard_index=args.shard_index)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path = _grid_path(output_prefix)
    output_summary_path = _summary_path(output_prefix)

    tra_validation, seq_pred, seq_label = _load_rank_ensemble_validation(args.result_suffix)
    signal_candidates = _build_signal_candidates(
        seq_pred,
        seq_label,
        signal_profile=args.signal_profile,
        xgb_validation_cache_path=args.xgb_validation_cache_path.resolve(),
    )
    if args.signal_name:
        wanted = set(args.signal_name)
        signal_candidates = [candidate for candidate in signal_candidates if candidate["signal_name"] in wanted]
        found = {candidate["signal_name"] for candidate in signal_candidates}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"unknown signal_name values for profile {args.signal_profile}: {missing}")
    strategy_trials = _strategy_trials(args.strategy_profile)
    all_candidates = [
        (signal_candidate, strategy_trial)
        for signal_candidate in signal_candidates
        for strategy_trial in strategy_trials
    ]
    shard_candidates = [
        item for index, item in enumerate(all_candidates) if index % args.num_shards == args.shard_index
    ]

    row_map = {} if args.force_recompute else _load_existing_rows(output_csv_path)
    rows = list(row_map.values())
    reused_count = 0
    evaluated_count = 0

    for signal_candidate, strategy_trial in shard_candidates:
        trial_name = f"{signal_candidate['signal_name']}__{strategy_trial['trial_name']}"
        if trial_name in row_map:
            reused_count += 1
            continue

        row = {
            "trial_name": trial_name,
            "signal_name": signal_candidate["signal_name"],
            "kind": signal_candidate["kind"],
            "strategy_trial": strategy_trial["trial_name"],
            "n_drop": strategy_trial["n_drop"],
            "risk_degree": strategy_trial["risk_degree"],
            "kwargs_extra": str(strategy_trial["kwargs_extra"]),
            "tab_weight": signal_candidate.get("tab_weight", pd.NA),
            "low_q": signal_candidate.get("low_q", pd.NA),
            "penalty": signal_candidate.get("penalty", pd.NA),
            "status": "running",
        }
        try:
            metrics = stage2._evaluate_signal(
                recorder_name=trial_name,
                pred=signal_candidate["pred"],
                label=signal_candidate["label"],
                strategy_cfg=stage2._strategy_cfg_from_trial(tra_validation, strategy_trial, eval_segment="valid"),
                experiment_name="alpha360_tra_stage2_signal_fast_validation_eval",
                fast_mode=True,
            )
            row.update({"status": "success", **metrics})
        except Exception as exc:
            row.update({"status": "failed", "error": repr(exc)[:500]})
        row_map[trial_name] = row
        rows = list(row_map.values())
        evaluated_count += 1
        _persist(rows, output_csv_path)
        print(
            f"evaluated={evaluated_count} reused={reused_count} "
            f"trial={trial_name} status={row['status']} score={row.get('score')}"
        )

    _persist(rows, output_csv_path)
    best_rows = [row for row in rows if row.get("status") == "success"]
    best_rows = sorted(
        best_rows,
        key=lambda row: (
            float(row.get("score", float("-inf"))),
            float(row.get("with_cost_ann_return", float("-inf"))),
            float(row.get("with_cost_ir", float("-inf"))),
            float(row.get("with_cost_mdd", float("-inf"))),
        ),
        reverse=True,
    )
    summary = {
        "run_mode": "stage2_signal_fast_validation_shard",
        "selection_scope": "validation_only",
        "signal_profile": args.signal_profile,
        "strategy_profile": args.strategy_profile,
        "candidate_count_total": len(all_candidates),
        "candidate_count_this_shard": len(shard_candidates),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "reused_count": reused_count,
        "evaluated_count": evaluated_count,
        "output_csv_path": str(output_csv_path),
        "best_validation_selected": best_rows[0] if best_rows else None,
        "top10_validation": best_rows[:10],
    }
    output_summary_path.write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    print(output_summary_path)


def merge_shards(args: argparse.Namespace) -> None:
    output_prefix = args.output_prefix.resolve()
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
    merged = merged.drop_duplicates(subset=["trial_name"], keep="last")
    merged = merged.sort_values(SORT_COLUMNS, ascending=[False, False, False, False], na_position="last")
    output_csv_path = _grid_path(output_prefix)
    output_summary_path = _summary_path(output_prefix)
    output_csv_path.write_text(merged.to_csv(index=False), encoding="utf-8")
    success = merged[merged["status"] == "success"]
    summary = {
        "run_mode": "stage2_signal_fast_validation_merged",
        "selection_scope": "validation_only",
        "num_shards": args.num_shards,
        "shard_grid_paths": shard_paths,
        "grid_rows": int(len(merged)),
        "success_rows": int(len(success)),
        "output_csv_path": str(output_csv_path),
        "best_validation_selected": success.iloc[0].to_dict() if not success.empty else None,
        "top10_validation": success.head(10).to_dict(orient="records"),
    }
    output_summary_path.write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    print(output_summary_path)


def launch_shards(args: argparse.Namespace) -> None:
    log_dir = args.log_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_prefix.resolve()
    xgb_validation_cache_path = args.xgb_validation_cache_path.resolve()
    procs = []
    for shard_index in range(args.num_shards):
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output-prefix",
            str(output_prefix),
            "--result-suffix",
            args.result_suffix,
            "--signal-profile",
            args.signal_profile,
            "--strategy-profile",
            args.strategy_profile,
            "--xgb-validation-cache-path",
            str(xgb_validation_cache_path),
            "--num-shards",
            str(args.num_shards),
            "--shard-index",
            str(shard_index),
        ]
        for signal_name in args.signal_name:
            cmd.extend(["--signal-name", signal_name])
        if args.force_recompute:
            cmd.append("--force-recompute")
        log_path = log_dir / f"{output_prefix.name}_shard{shard_index:02d}of{args.num_shards:02d}.log"
        log_file = log_path.open("w", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, cwd=BASE_DIR, stdout=log_file, stderr=subprocess.STDOUT), log_file, log_path))
        print(f"started shard {shard_index}/{args.num_shards}: {log_path}")

    failed = []
    for proc, log_file, log_path in procs:
        return_code = proc.wait()
        log_file.close()
        if return_code != 0:
            failed.append((return_code, log_path))
    if failed:
        raise RuntimeError(f"failed shards: {failed}")
    merge_shards(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast validation-only stage2 signal search with sharding.")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--result-suffix", default="stage2_signal_fast")
    parser.add_argument(
        "--signal-profile",
        choices=["fundamental-small", "fundamental-filter-fine", "xgb-small", "all-small"],
        default="fundamental-small",
    )
    parser.add_argument("--strategy-profile", choices=["hold4r08", "n_drop1_core"], default="hold4r08")
    parser.add_argument("--signal-name", action="append", default=[], help="Restrict search to one or more signal names.")
    parser.add_argument("--xgb-validation-cache-path", type=Path, default=DEFAULT_XGB_VALID_CACHE)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--launch-shards", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=BASE_DIR / "logs" / "stage2_signal_fast")
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.launch_shards:
        launch_shards(args)
    elif args.merge_shards:
        merge_shards(args)
    else:
        run_shard(args)


if __name__ == "__main__":
    main()
