from __future__ import annotations

import argparse
import ast
import itertools
import json
from pathlib import Path

import pandas as pd
from qlib.contrib.eva.alpha import calc_ic

import run_rank_ensemble_tra_alpha360_once as rank_runner
import run_stage2_signal_cv_search as cv
import run_stage2_signal_search_fast as fast
import tune_alpha360_tra_fundamental_xgb_weight_refine as stage2
from tra_local_helpers import localize_value


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp/h200_weighted_validation_selector_20260705_1"
FROZEN_GATE = {
    "full_ann_min": 0.24,
    "year_ann_min_min": 0.22,
    "year_ann_std_max": 0.04,
}
DEFAULT_TRIAL_NAMES = [
    "baseline_seq__prac_m000_hold5_r08",
    "baseline_seq__prac_m000_hold5_r085",
    "baseline_seq__prac_m000_hold7_r08",
    "baseline_seq__prac_m000_hold7_r085",
    "cash_quality_z_tw0005__prac_m000_hold5_r08",
    "cash_quality_z_tw0005__prac_m000_hold5_r085",
    "cash_quality_z_tw001__prac_m000_hold5_r08",
    "cash_quality_z_tw001__prac_m000_hold5_r085",
    "cash_quality_z_tw001__prac_m000_hold7_r08",
    "cash_quality_z_tw001__prac_m000_hold7_r085",
]


def _parse_summary(path: Path) -> dict:
    data = {}
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
        raise RuntimeError(f"test seed runs are not allowed in weighted validation selector: {path} keys={forbidden}")
    return data


def _seed_runs_from_summary(path: Path) -> dict[int, dict]:
    summary = _parse_summary(path)
    runs = summary.get("valid_base_runs")
    if runs is None:
        runs = summary.get("seed_runs_valid")
    if isinstance(runs, list):
        return {int(row["seed"]): row for row in runs}
    if isinstance(runs, dict):
        return {int(seed): dict(row, seed=int(seed)) for seed, row in runs.items()}
    raise TypeError(f"unsupported valid seed runs in {path}: {type(runs).__name__}")


def _load_seed_runs(paths: list[Path]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for path in paths:
        out.update(_seed_runs_from_summary(path))
    return out


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_weighted_candidate(raw: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
    parts = _split_csv(raw)
    seeds = []
    weights = []
    for part in parts:
        if ":" not in part:
            raise ValueError(f"weighted candidate must be seed:weight entries: {raw}")
        seed_text, weight_text = part.split(":", 1)
        seeds.append(int(seed_text))
        weights.append(float(weight_text))
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"non-positive weight sum: {raw}")
    weights = [weight / total for weight in weights]
    return tuple(seeds), tuple(weights)


def _weight_templates(size: int) -> list[tuple[float, ...]]:
    if size == 1:
        return [(1.0,)]
    if size == 2:
        bases = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3)]
    elif size == 3:
        bases = [
            (1 / 3, 1 / 3, 1 / 3),
            (0.5, 0.3, 0.2),
            (0.6, 0.25, 0.15),
            (0.6, 0.3, 0.1),
            (0.7, 0.2, 0.1),
        ]
    elif size == 4:
        bases = [
            (0.25, 0.25, 0.25, 0.25),
            (0.4, 0.3, 0.2, 0.1),
            (0.5, 0.2, 0.2, 0.1),
            (0.5, 0.3, 0.1, 0.1),
            (0.6, 0.2, 0.1, 0.1),
        ]
    else:
        raise ValueError(f"unsupported combo size for default templates: {size}")
    out = []
    seen = set()
    for base in bases:
        for perm in itertools.permutations(base):
            key = tuple(round(v, 8) for v in perm)
            if key in seen:
                continue
            seen.add(key)
            out.append(tuple(float(v) for v in perm))
    return out


def _strategy_by_name(profiles: list[str]) -> dict[str, dict]:
    out = {}
    for profile in profiles:
        for trial in cv._strategy_trials(profile):
            item = dict(trial)
            item["strategy_profile"] = profile
            out.setdefault(trial["trial_name"], item)
    return out


def _candidate_by_name(seq_pred: pd.DataFrame, seq_label: pd.DataFrame, profile: str) -> dict[str, dict]:
    return {candidate["signal_name"]: candidate for candidate in cv._build_cv_signal_candidates(seq_pred, seq_label, profile=profile)}


def _weighted_rank_ensemble(
    pred_by_seed: dict[int, pd.DataFrame],
    label_by_seed: dict[int, pd.DataFrame],
    seeds: tuple[int, ...],
    weights: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    common = None
    for seed in seeds:
        pred = pred_by_seed[seed]
        common = pred.index if common is None else common.intersection(pred.index)
    common = common.sort_values()
    weighted = None
    for seed, weight in zip(seeds, weights):
        ranked = pred_by_seed[seed].loc[common].sort_index().groupby(level=0, group_keys=False).rank(method="average", pct=True)
        weighted = ranked.mul(weight) if weighted is None else weighted.add(ranked.mul(weight), fill_value=0.0)
    label = label_by_seed[seeds[0]].loc[common].sort_index().copy()
    return weighted.sort_index(), label, int(len(common))


def _signal_ic(pred: pd.DataFrame, label: pd.DataFrame, segment: tuple[str, str]) -> tuple[float, float]:
    pred_s = cv._slice_signal(pred, segment).iloc[:, 0]
    label_s = cv._slice_signal(label, segment).iloc[:, 0]
    common = pred_s.index.intersection(label_s.index).sort_values()
    ic, rank_ic = calc_ic(pred_s.loc[common], label_s.loc[common])
    return float(ic.mean()), float(rank_ic.mean())


def _ic_prefilter_row(
    seeds: tuple[int, ...],
    weights: tuple[float, ...],
    pred: pd.DataFrame,
    label: pd.DataFrame,
    rows: int,
) -> dict:
    result = {
        "weight_key": _weight_key(seeds, weights),
        "seeds": ",".join(str(seed) for seed in seeds),
        "weights": ",".join(f"{weight:.4f}" for weight in weights),
        "valid_common_rows": rows,
    }
    rankics = []
    for name, segment in cv.SEGMENTS.items():
        ic, rank_ic = _signal_ic(pred, label, segment)
        result[f"IC_{name}"] = ic
        result[f"RankIC_{name}"] = rank_ic
        if name != "valid_all":
            rankics.append(rank_ic)
    result["rankic_year_min"] = min(rankics)
    result["rankic_year_mean"] = sum(rankics) / len(rankics)
    result["rankic_year_spread"] = max(rankics) - min(rankics)
    result["ic_prefilter_score"] = (
        result["rankic_year_min"] * 1000
        + result["rankic_year_mean"] * 200
        - result["rankic_year_spread"] * 300
        + result["RankIC_valid_all"] * 100
    )
    return result


def _weight_key(seeds: tuple[int, ...], weights: tuple[float, ...]) -> str:
    return ",".join(f"{seed}:{weight:.4f}" for seed, weight in zip(seeds, weights))


def _frozen_gate_pass(row: dict) -> bool:
    return (
        float(row["full_ann"]) >= FROZEN_GATE["full_ann_min"]
        and float(row["year_ann_min"]) >= FROZEN_GATE["year_ann_min_min"]
        and float(row["year_ann_std"]) <= FROZEN_GATE["year_ann_std_max"]
    )


def _rank_cv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(
        ["frozen_gate_pass", "year_ann_min", "stable_score", "full_ann", "full_ir"],
        ascending=[False, False, False, False, False],
        na_position="last",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only weighted H200 stage1 ensemble selector.")
    parser.add_argument("--valid-summary-path", action="append", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--seed-pool", required=True)
    parser.add_argument("--combo-sizes", default="3,4")
    parser.add_argument("--weighted-candidate", action="append", default=None)
    parser.add_argument("--max-ic-candidates", type=int, default=32)
    parser.add_argument("--signal-profile", default="cash-quality-quick")
    parser.add_argument("--strategy-profiles", default="stable-core,stable-cash-quick")
    parser.add_argument("--trial-name", action="append", default=None)
    parser.add_argument("--all-profile-trials", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    ic_grid_path = args.output_prefix.with_name(f"{args.output_prefix.name}_ic_prefilter.csv")
    cv_grid_path = args.output_prefix.with_name(f"{args.output_prefix.name}_cv_grid.csv")
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.json")

    seed_runs = _load_seed_runs(args.valid_summary_path)
    seed_pool = [int(seed) for seed in _split_csv(args.seed_pool)]
    missing = [seed for seed in seed_pool if seed not in seed_runs]
    if missing:
        raise RuntimeError(f"missing seeds in validation summaries: {missing}")

    mod = rank_runner.load_module()
    pred_by_seed = {}
    label_by_seed = {}
    for seed in seed_pool:
        pred_by_seed[seed], label_by_seed[seed] = mod._load_signal(Path(seed_runs[seed]["cache_path"]))

    weighted_specs: list[tuple[tuple[int, ...], tuple[float, ...]]] = []
    if args.weighted_candidate:
        weighted_specs = [_parse_weighted_candidate(raw) for raw in args.weighted_candidate]
    else:
        for size in [int(item) for item in _split_csv(args.combo_sizes)]:
            for seeds in itertools.combinations(seed_pool, size):
                for weights in _weight_templates(size):
                    weighted_specs.append((tuple(seeds), tuple(weights)))

    ic_rows = []
    signal_by_key: dict[str, tuple[pd.DataFrame, pd.DataFrame, int, tuple[int, ...], tuple[float, ...]]] = {}
    for seeds, weights in weighted_specs:
        pred, label, rows = _weighted_rank_ensemble(pred_by_seed, label_by_seed, seeds, weights)
        ic_row = _ic_prefilter_row(seeds, weights, pred, label, rows)
        ic_rows.append(ic_row)
        signal_by_key[ic_row["weight_key"]] = (pred, label, rows, seeds, weights)
    ic_df = pd.DataFrame(ic_rows).sort_values(
        ["ic_prefilter_score", "rankic_year_min", "rankic_year_mean", "RankIC_valid_all"],
        ascending=[False, False, False, False],
    )
    ic_df.to_csv(ic_grid_path, index=False)
    selected_keys = ic_df.head(int(args.max_ic_candidates))["weight_key"].tolist()

    cv_rows = []
    if cv_grid_path.exists() and not args.force_recompute:
        cv_rows = pd.read_csv(cv_grid_path).to_dict(orient="records")
        done = {str(row["selector_key"]) for row in cv_rows}
    else:
        done = set()

    strategies = _strategy_by_name(_split_csv(args.strategy_profiles))
    trial_names = args.trial_name or DEFAULT_TRIAL_NAMES
    tra_validation = {
        "window_key": "w1",
        "train": ["2012-01-01", "2019-12-31"],
        "valid": ["2020-01-01", "2022-12-31"],
        "test": ["2023-01-01", "2026-04-01"],
        "handler_kwargs_extra": stage2.TRA_RANK_ENSEMBLE_HANDLER_KWARGS,
    }

    for weight_key in selected_keys:
        pred, label, rows, seeds, weights = signal_by_key[weight_key]
        candidates = _candidate_by_name(pred, label, args.signal_profile)
        active_trials = (
            [
                f"{signal_name}__{strategy_name}"
                for signal_name in candidates
                for strategy_name in strategies
            ]
            if args.all_profile_trials
            else trial_names
        )
        for trial_name in active_trials:
            signal_name, strategy_name = trial_name.split("__", 1)
            selector_key = f"{weight_key}::{trial_name}"
            if selector_key in done:
                continue
            if signal_name not in candidates or strategy_name not in strategies:
                continue
            row = cv._evaluate_cv_candidate(tra_validation, candidates[signal_name], strategies[strategy_name])
            row.update(
                {
                    "selector_key": selector_key,
                    "weight_key": weight_key,
                    "seeds": ",".join(str(seed) for seed in seeds),
                    "weights": ",".join(f"{weight:.4f}" for weight in weights),
                    "valid_common_rows": rows,
                    "frozen_gate_pass": _frozen_gate_pass(row),
                    "strategy_profile": strategies[strategy_name]["strategy_profile"],
                }
            )
            cv_rows.append(row)
            done.add(selector_key)
            _rank_cv(pd.DataFrame(cv_rows)).to_csv(cv_grid_path, index=False)
            print(
                f"weight={weight_key} trial={trial_name} "
                f"full_ann={row['full_ann']:.6f} year_min={row['year_ann_min']:.6f} "
                f"year_std={row['year_ann_std']:.6f} frozen_gate_pass={row['frozen_gate_pass']}"
            )

    cv_df = _rank_cv(pd.DataFrame(cv_rows))
    cv_df.to_csv(cv_grid_path, index=False)
    summary = {
        "run_mode": "h200_weighted_validation_selector",
        "selection_scope": "validation_only_no_test_loaded",
        "test_usage_policy": "no_test_access_before_candidate_lock",
        "valid_summary_paths": [str(path) for path in args.valid_summary_path],
        "seed_pool": seed_pool,
        "weighted_spec_count": len(weighted_specs),
        "max_ic_candidates": int(args.max_ic_candidates),
        "signal_profile": args.signal_profile,
        "strategy_profiles": _split_csv(args.strategy_profiles),
        "frozen_gate": FROZEN_GATE,
        "ic_prefilter_path": str(ic_grid_path),
        "cv_grid_path": str(cv_grid_path),
        "cv_row_count": int(len(cv_df)),
        "frozen_gate_pass_count": int(cv_df["frozen_gate_pass"].sum()) if not cv_df.empty else 0,
        "best_validation_selected": cv_df.iloc[0].to_dict() if not cv_df.empty else None,
        "top20_validation": cv_df.head(20).to_dict(orient="records") if not cv_df.empty else [],
    }
    summary_path.write_text(json.dumps(fast._jsonable(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    print(ic_grid_path)
    print(cv_grid_path)
    print(summary_path)


if __name__ == "__main__":
    main()
