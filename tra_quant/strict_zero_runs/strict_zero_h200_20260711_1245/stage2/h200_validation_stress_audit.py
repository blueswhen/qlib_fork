from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

import run_rank_ensemble_tra_alpha360_once as rank_runner
import run_stage2_signal_cv_search as cv
import run_stage2_signal_search_fast as fast
import tune_alpha360_tra_fundamental_xgb_weight_refine as stage2
from tra_local_helpers import localize_value


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp/h200_validation_stress_audit_20260705_1"
DEFAULT_STRESS_GATE = {
    "gate_version": "v2_no_negative_valid_quarters_20260705",
    "full_ann_min": 0.26,
    "year_ann_min_min": 0.26,
    "half_ann_min_min": 0.05,
    "quarter_ann_min_min": 0.0,
    "quarter_positive_ratio_min": 0.80,
    "quarter_rankic_min": 0.02,
    "quarter_rankic_positive_ratio_min": 1.0,
}


def _parse_summary(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            data[key] = localize_value(ast.literal_eval(value))
        except Exception:
            data[key] = localize_value(value)
    forbidden = [key for key in ("seed_runs_test", "test_base_runs") if data.get(key)]
    if forbidden:
        raise RuntimeError(f"test seed runs are not allowed in validation stress audit: {path} keys={forbidden}")
    return data


def _seed_runs_from_valid_summary(path: Path) -> dict[int, dict[str, Any]]:
    summary = _parse_summary(path)
    runs = summary.get("valid_base_runs")
    if runs is None:
        runs = summary.get("seed_runs_valid")
    if isinstance(runs, list):
        return {int(row["seed"]): row for row in runs}
    if isinstance(runs, dict):
        return {int(seed): dict(row, seed=int(seed)) for seed, row in runs.items()}
    raise TypeError(f"unsupported validation seed runs in {path}: {type(runs).__name__}")


def _load_valid_seed_runs(paths: list[Path]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in paths:
        out.update(_seed_runs_from_valid_summary(path))
    return out


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _strategy_by_name(profiles: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        for trial in cv._strategy_trials(profile):
            item = dict(trial)
            item["strategy_profile"] = profile
            out.setdefault(trial["trial_name"], item)
    return out


def _candidate_by_name(seq_pred: pd.DataFrame, seq_label: pd.DataFrame, profile: str) -> dict[str, dict[str, Any]]:
    return {candidate["signal_name"]: candidate for candidate in cv._build_cv_signal_candidates(seq_pred, seq_label, profile=profile)}


def _rank_ensemble(mod: Any, seed_runs: dict[int, dict[str, Any]], combo: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    cache_paths = [Path(seed_runs[seed]["cache_path"]) for seed in combo]
    return rank_runner.rank_ensemble_signal(mod, cache_paths)


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return bool(str(value).strip()) and str(value).strip().lower() != "nan"


def _parse_weights(row: dict[str, Any], combo: tuple[int, ...]) -> tuple[float, ...] | None:
    if _has_value(row.get("weights")):
        weights = [float(item) for item in _split_csv(str(row["weights"]))]
    elif _has_value(row.get("weight_key")) and ":" in str(row["weight_key"]):
        parts = _split_csv(str(row["weight_key"]))
        parsed = []
        for part in parts:
            seed_text, weight_text = part.split(":", 1)
            parsed.append((int(seed_text), float(weight_text)))
        seeds = tuple(seed for seed, _ in parsed)
        if seeds != combo:
            raise RuntimeError(f"weight_key seed order does not match seeds: weight_key={row['weight_key']} combo={combo}")
        weights = [weight for _, weight in parsed]
    else:
        return None
    if len(weights) != len(combo):
        raise RuntimeError(f"weights length does not match seeds: combo={combo} weights={weights}")
    total = sum(weights)
    if total <= 0:
        raise RuntimeError(f"non-positive weight sum: combo={combo} weights={weights}")
    return tuple(weight / total for weight in weights)


def _weighted_rank_ensemble(
    mod: Any,
    seed_runs: dict[int, dict[str, Any]],
    combo: tuple[int, ...],
    weights: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    pred_by_seed: dict[int, pd.DataFrame] = {}
    label_by_seed: dict[int, pd.DataFrame] = {}
    common = None
    for seed in combo:
        pred_by_seed[seed], label_by_seed[seed] = mod._load_signal(Path(seed_runs[seed]["cache_path"]))
        common = pred_by_seed[seed].index if common is None else common.intersection(pred_by_seed[seed].index)
    common = common.intersection(label_by_seed[combo[0]].index).sort_values()
    weighted = None
    for seed, weight in zip(combo, weights):
        ranked = (
            pred_by_seed[seed]
            .loc[common]
            .sort_index()
            .groupby(level=0, group_keys=False)
            .rank(method="average", pct=True)
        )
        weighted = ranked.mul(weight) if weighted is None else weighted.add(ranked.mul(weight), fill_value=0.0)
    label = label_by_seed[combo[0]].loc[common].sort_index().copy()
    return weighted.sort_index(), label, int(len(common))


def _date_segments(levels: list[str]) -> dict[str, tuple[str, str]]:
    segments: dict[str, tuple[str, str]] = {"valid_all": ("2020-01-01", "2022-12-31")}
    if "year" in levels:
        for year in (2020, 2021, 2022):
            segments[f"Y{year}"] = (f"{year}-01-01", f"{year}-12-31")
    if "half" in levels:
        for year in (2020, 2021, 2022):
            segments[f"H1_{year}"] = (f"{year}-01-01", f"{year}-06-30")
            segments[f"H2_{year}"] = (f"{year}-07-01", f"{year}-12-31")
    if "quarter" in levels:
        for year in (2020, 2021, 2022):
            for quarter, start, end in (
                (1, "01-01", "03-31"),
                (2, "04-01", "06-30"),
                (3, "07-01", "09-30"),
                (4, "10-01", "12-31"),
            ):
                segments[f"Q{quarter}_{year}"] = (f"{year}-{start}", f"{year}-{end}")
    if "month" in levels:
        for dt in pd.date_range("2020-01-01", "2022-12-01", freq="MS"):
            start = dt.strftime("%Y-%m-%d")
            end = (dt + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
            segments[f"M{dt.strftime('%Y%m')}"] = (start, end)
    return segments


def _group_for_segment(name: str) -> str:
    if name == "valid_all":
        return "full"
    if name.startswith("Y"):
        return "year"
    if name.startswith("H"):
        return "half"
    if name.startswith("Q"):
        return "quarter"
    if name.startswith("M"):
        return "month"
    return "other"


def _summarize_group(rows: list[dict[str, Any]], group: str) -> dict[str, Any]:
    group_rows = [row for row in rows if row["segment_group"] == group]
    if not group_rows:
        return {}
    ann = pd.Series([row["with_cost_ann_return"] for row in group_rows], dtype="float64")
    rankic = pd.Series([row["Rank IC"] for row in group_rows], dtype="float64")
    return {
        f"{group}_count": int(len(group_rows)),
        f"{group}_ann_min": float(ann.min()),
        f"{group}_ann_mean": float(ann.mean()),
        f"{group}_ann_median": float(ann.median()),
        f"{group}_ann_std": float(ann.std(ddof=0)),
        f"{group}_ann_positive_ratio": float((ann > 0).mean()),
        f"{group}_rankic_min": float(rankic.min()),
        f"{group}_rankic_mean": float(rankic.mean()),
        f"{group}_rankic_positive_ratio": float((rankic > 0).mean()),
    }


def _stress_gate(row: dict[str, Any], gate: dict[str, Any]) -> bool:
    return (
        float(row.get("full_ann", float("-inf"))) >= gate["full_ann_min"]
        and float(row.get("year_ann_min", float("-inf"))) >= gate["year_ann_min_min"]
        and float(row.get("half_ann_min", float("-inf"))) >= gate["half_ann_min_min"]
        and float(row.get("quarter_ann_min", float("-inf"))) >= gate["quarter_ann_min_min"]
        and float(row.get("quarter_ann_positive_ratio", 0.0)) >= gate["quarter_positive_ratio_min"]
        and float(row.get("quarter_rankic_min", float("-inf"))) >= gate["quarter_rankic_min"]
        and float(row.get("quarter_rankic_positive_ratio", 0.0)) >= gate["quarter_rankic_positive_ratio_min"]
    )


def _candidate_rows_from_grid(
    paths: list[Path],
    max_candidates: int,
    *,
    prefilter_full_ann_min: float | None,
    prefilter_year_ann_min: float | None,
) -> list[dict[str, Any]]:
    frames = [pd.read_csv(path) for path in paths]
    df = pd.concat(frames, ignore_index=True)
    if "selector_key" in df.columns:
        df = df.drop_duplicates("selector_key", keep="first")
    if "status" in df.columns:
        df = df[df["status"].fillna("success") == "success"]
    for column in ("full_ann", "year_ann_min", "year_ann_std", "stable_score", "full_ir"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if prefilter_full_ann_min is not None:
        if "full_ann" not in df.columns:
            raise RuntimeError("grid prefilter requested but full_ann column is missing")
        df = df[df["full_ann"] >= float(prefilter_full_ann_min)]
    if prefilter_year_ann_min is not None:
        if "year_ann_min" not in df.columns:
            raise RuntimeError("grid prefilter requested but year_ann_min column is missing")
        df = df[df["year_ann_min"] >= float(prefilter_year_ann_min)]
    sort_cols = [column for column in ("year_ann_min", "full_ann", "stable_score", "full_ir") if column in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    return df.head(int(max_candidates)).to_dict(orient="records")


def _candidate_rows_from_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = data.get("selected_row")
    if not selected:
        raise RuntimeError(f"locked manifest has no selected_row: {path}")
    return [selected]


def _parse_combo(row: dict[str, Any]) -> tuple[int, ...]:
    if row.get("combo"):
        return tuple(int(seed) for seed in str(row["combo"]).split(","))
    if row.get("seeds"):
        return tuple(int(seed) for seed in str(row["seeds"]).split(","))
    raise RuntimeError(f"candidate row has no combo/seeds: {row}")


def _evaluate_candidate(
    *,
    mod: Any,
    seed_runs: dict[int, dict[str, Any]],
    row: dict[str, Any],
    signal_profile: str,
    strategies: dict[str, dict[str, Any]],
    segments: dict[str, tuple[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    combo = _parse_combo(row)
    missing = [seed for seed in combo if seed not in seed_runs]
    if missing:
        raise RuntimeError(f"candidate missing validation seed runs: combo={combo} missing={missing}")
    trial_name = str(row["trial_name"])
    signal_name, strategy_name = trial_name.split("__", 1)
    if strategy_name not in strategies:
        raise RuntimeError(f"strategy not available in selected profiles: {strategy_name}")

    weights = _parse_weights(row, combo)
    if weights is None:
        seq_pred, seq_label, valid_common_rows = _rank_ensemble(mod, seed_runs, combo)
        weight_key = None
    else:
        seq_pred, seq_label, valid_common_rows = _weighted_rank_ensemble(mod, seed_runs, combo, weights)
        weight_key = ",".join(f"{seed}:{weight:.4f}" for seed, weight in zip(combo, weights))
    candidates = _candidate_by_name(seq_pred.sort_index(), seq_label.sort_index(), signal_profile)
    if signal_name not in candidates:
        raise RuntimeError(f"signal not available in {signal_profile}: {signal_name}")
    signal = candidates[signal_name]
    strategy = strategies[strategy_name]
    tra_validation = {
        "window_key": "w1",
        "train": ["2012-01-01", "2019-12-31"],
        "valid": ["2020-01-01", "2022-12-31"],
        "test": ["2023-01-01", "2026-04-01"],
        "handler_kwargs_extra": stage2.TRA_RANK_ENSEMBLE_HANDLER_KWARGS,
    }

    details: list[dict[str, Any]] = []
    for segment_name, date_range in segments.items():
        metrics = cv._evaluate_segment(
            tra_validation,
            strategy,
            signal["pred"].sort_index(),
            signal["label"].sort_index(),
            segment=date_range,
        )
        details.append(
            {
                "selector_key": str(row.get("selector_key", f"{','.join(str(seed) for seed in combo)}::{trial_name}")),
                "combo": ",".join(str(seed) for seed in combo),
                "weight_key": weight_key,
                "weights": ",".join(f"{weight:.6f}" for weight in weights) if weights is not None else None,
                "trial_name": trial_name,
                "signal_name": signal_name,
                "strategy_trial": strategy_name,
                "strategy_profile": strategy.get("strategy_profile"),
                "segment_name": segment_name,
                "segment_group": _group_for_segment(segment_name),
                "start": date_range[0],
                "end": date_range[1],
                **metrics,
            }
        )

    summary: dict[str, Any] = {
        "selector_key": str(row.get("selector_key", f"{','.join(str(seed) for seed in combo)}::{trial_name}")),
        "combo": ",".join(str(seed) for seed in combo),
        "weight_key": weight_key,
        "weights": ",".join(f"{weight:.6f}" for weight in weights) if weights is not None else None,
        "trial_name": trial_name,
        "signal_name": signal_name,
        "strategy_trial": strategy_name,
        "strategy_profile": strategy.get("strategy_profile"),
        "valid_common_rows": int(valid_common_rows),
        "source_full_ann": row.get("full_ann"),
        "source_year_ann_min": row.get("year_ann_min"),
        "source_year_ann_std": row.get("year_ann_std"),
    }
    for group in ("full", "year", "half", "quarter", "month"):
        summary.update(_summarize_group(details, group))
    summary["full_ann"] = summary.get("full_ann_mean")
    summary["year_ann_min"] = summary.get("year_ann_min")
    summary["stress_score"] = (
        float(summary.get("year_ann_min", -9.0)) * 500
        + float(summary.get("half_ann_min", -9.0)) * 250
        + float(summary.get("quarter_ann_positive_ratio", 0.0)) * 100
        + float(summary.get("quarter_rankic_positive_ratio", 0.0)) * 50
        - max(0.0, -float(summary.get("quarter_ann_min", 0.0))) * 150
    )
    return summary, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only H200 stage2 stress audit. It refuses test seed summaries.")
    parser.add_argument("--valid-summary-path", action="append", type=Path, required=True)
    parser.add_argument("--cv-grid-path", action="append", type=Path, default=None)
    parser.add_argument("--locked-manifest", action="append", type=Path, default=None)
    parser.add_argument("--trial-name", action="append", default=None)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--signal-profile", default="cash-quality-quick")
    parser.add_argument("--strategy-profiles", default="stable-core,stable-cash-quick,stable-focused,stable-quick")
    parser.add_argument("--segment-levels", default="year,half,quarter")
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--prefilter-full-ann-min", type=float, default=0.26)
    parser.add_argument("--prefilter-year-ann-min", type=float, default=0.26)
    parser.add_argument("--gate-full-ann-min", type=float, default=DEFAULT_STRESS_GATE["full_ann_min"])
    parser.add_argument("--gate-year-ann-min", type=float, default=DEFAULT_STRESS_GATE["year_ann_min_min"])
    parser.add_argument("--gate-half-ann-min", type=float, default=DEFAULT_STRESS_GATE["half_ann_min_min"])
    parser.add_argument("--gate-quarter-ann-min", type=float, default=DEFAULT_STRESS_GATE["quarter_ann_min_min"])
    parser.add_argument(
        "--gate-quarter-positive-ratio-min",
        type=float,
        default=DEFAULT_STRESS_GATE["quarter_positive_ratio_min"],
    )
    parser.add_argument("--gate-quarter-rankic-min", type=float, default=DEFAULT_STRESS_GATE["quarter_rankic_min"])
    parser.add_argument(
        "--gate-quarter-rankic-positive-ratio-min",
        type=float,
        default=DEFAULT_STRESS_GATE["quarter_rankic_positive_ratio_min"],
    )
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.cv_grid_path and not args.locked_manifest:
        raise RuntimeError("provide --cv-grid-path or --locked-manifest")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_prefix.with_name(f"{args.output_prefix.name}_segment_detail.csv")
    candidate_path = args.output_prefix.with_name(f"{args.output_prefix.name}_candidate_summary.csv")
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.json")
    if not args.force_recompute and (detail_path.exists() or candidate_path.exists() or summary_path.exists()):
        raise FileExistsError("stress audit output already exists; use --force-recompute to overwrite")

    seed_runs = _load_valid_seed_runs(args.valid_summary_path)
    rows: list[dict[str, Any]] = []
    if args.cv_grid_path:
        rows.extend(
            _candidate_rows_from_grid(
                args.cv_grid_path,
                args.max_candidates,
                prefilter_full_ann_min=args.prefilter_full_ann_min,
                prefilter_year_ann_min=args.prefilter_year_ann_min,
            )
        )
    if args.locked_manifest:
        for path in args.locked_manifest:
            rows.extend(_candidate_rows_from_manifest(path))
    if args.trial_name:
        requested_trials = set(args.trial_name)
        rows = [row for row in rows if str(row["trial_name"]) in requested_trials]
    if not rows:
        raise RuntimeError("no candidate rows selected for stress audit")

    segments = _date_segments(_split_csv(args.segment_levels))
    strategies = _strategy_by_name(_split_csv(args.strategy_profiles))
    mod = rank_runner.load_module()
    stress_gate = {
        "gate_version": "cli_explicit",
        "full_ann_min": float(args.gate_full_ann_min),
        "year_ann_min_min": float(args.gate_year_ann_min),
        "half_ann_min_min": float(args.gate_half_ann_min),
        "quarter_ann_min_min": float(args.gate_quarter_ann_min),
        "quarter_positive_ratio_min": float(args.gate_quarter_positive_ratio_min),
        "quarter_rankic_min": float(args.gate_quarter_rankic_min),
        "quarter_rankic_positive_ratio_min": float(args.gate_quarter_rankic_positive_ratio_min),
    }

    all_details: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    seen = set()
    for row in rows:
        key = (
            str(row.get("selector_key") or row.get("weight_key") or row.get("combo") or row.get("seeds")),
            str(row["trial_name"]),
        )
        if key in seen:
            continue
        seen.add(key)
        candidate_summary, details = _evaluate_candidate(
            mod=mod,
            seed_runs=seed_runs,
            row=row,
            signal_profile=args.signal_profile,
            strategies=strategies,
            segments=segments,
        )
        candidate_summary["stress_gate_pass"] = _stress_gate(candidate_summary, stress_gate)
        candidate_summaries.append(candidate_summary)
        all_details.extend(details)
        pd.DataFrame(all_details).to_csv(detail_path, index=False)
        pd.DataFrame(candidate_summaries).sort_values(
            ["stress_gate_pass", "stress_score", "year_ann_min", "full_ann"],
            ascending=[False, False, False, False],
            na_position="last",
        ).to_csv(candidate_path, index=False)
        print(
            f"candidate={candidate_summary['combo']}::{candidate_summary['trial_name']} "
            f"stress_pass={candidate_summary['stress_gate_pass']} "
            f"full_ann={candidate_summary.get('full_ann')} "
            f"year_min={candidate_summary.get('year_ann_min')} "
            f"quarter_pos={candidate_summary.get('quarter_ann_positive_ratio')}"
        )

    ranked = pd.DataFrame(candidate_summaries).sort_values(
        ["stress_gate_pass", "stress_score", "year_ann_min", "full_ann"],
        ascending=[False, False, False, False],
        na_position="last",
    )
    ranked.to_csv(candidate_path, index=False)
    pd.DataFrame(all_details).to_csv(detail_path, index=False)
    summary = {
        "run_mode": "h200_validation_stress_audit",
        "selection_scope": "validation_only_no_test_loaded",
        "test_usage_policy": "consumed_test_not_loaded_not_used_for_selection",
        "valid_summary_paths": [str(path) for path in args.valid_summary_path],
        "cv_grid_paths": [str(path) for path in args.cv_grid_path or []],
        "locked_manifests": [str(path) for path in args.locked_manifest or []],
        "trial_name_filter": list(args.trial_name or []),
        "signal_profile": args.signal_profile,
        "strategy_profiles": _split_csv(args.strategy_profiles),
        "segment_levels": _split_csv(args.segment_levels),
        "grid_prefilter": {
            "full_ann_min": args.prefilter_full_ann_min,
            "year_ann_min_min": args.prefilter_year_ann_min,
        },
        "stress_gate": stress_gate,
        "candidate_count": int(len(ranked)),
        "stress_gate_pass_count": int(ranked["stress_gate_pass"].sum()) if not ranked.empty else 0,
        "candidate_summary_path": str(candidate_path),
        "segment_detail_path": str(detail_path),
        "top_validation_stress": ranked.head(20).to_dict(orient="records") if not ranked.empty else [],
    }
    summary_path.write_text(json.dumps(fast._jsonable(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    print(candidate_path)
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
