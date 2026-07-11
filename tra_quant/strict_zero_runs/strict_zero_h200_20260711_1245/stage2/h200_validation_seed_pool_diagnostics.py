from __future__ import annotations

import argparse
import ast
import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd
from qlib.contrib.eva.alpha import calc_ic

import run_rank_ensemble_tra_alpha360_once as rank_runner
import run_stage2_signal_search_fast as fast
from tra_local_helpers import localize_value


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp/h200_validation_seed_pool_diagnostics_20260706_1"
SEGMENTS = {
    "valid_all": ("2020-01-01", "2022-12-31"),
    "Y2020": ("2020-01-01", "2020-12-31"),
    "Y2021": ("2021-01-01", "2021-12-31"),
    "Y2022": ("2022-01-01", "2022-12-31"),
    "Q1_2020": ("2020-01-01", "2020-03-31"),
    "Q2_2020": ("2020-04-01", "2020-06-30"),
    "Q3_2020": ("2020-07-01", "2020-09-30"),
    "Q4_2020": ("2020-10-01", "2020-12-31"),
    "Q1_2021": ("2021-01-01", "2021-03-31"),
    "Q2_2021": ("2021-04-01", "2021-06-30"),
    "Q3_2021": ("2021-07-01", "2021-09-30"),
    "Q4_2021": ("2021-10-01", "2021-12-31"),
    "Q1_2022": ("2022-01-01", "2022-03-31"),
    "Q2_2022": ("2022-04-01", "2022-06-30"),
    "Q3_2022": ("2022-07-01", "2022-09-30"),
    "Q4_2022": ("2022-10-01", "2022-12-31"),
}


def _jsonable(value: Any) -> Any:
    return fast._jsonable(value)


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
        raise RuntimeError(f"test seed runs are not allowed in seed diagnostics: {path} keys={forbidden}")
    return data


def _seed_runs_from_summary(path: Path) -> dict[int, dict[str, Any]]:
    summary = _parse_summary(path)
    runs = summary.get("valid_base_runs")
    if runs is None:
        runs = summary.get("seed_runs_valid")
    if isinstance(runs, list):
        return {int(row["seed"]): dict(row, seed=int(row["seed"]), summary_path=str(path)) for row in runs}
    if isinstance(runs, dict):
        return {
            int(seed): dict(row, seed=int(seed), summary_path=str(path))
            for seed, row in runs.items()
        }
    raise TypeError(f"unsupported validation seed runs in {path}: {type(runs).__name__}")


def _load_seed_runs(paths: list[Path]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in paths:
        out.update(_seed_runs_from_summary(path))
    return out


def _date_index(df: pd.DataFrame) -> pd.Index:
    level = "datetime" if "datetime" in df.index.names else 0
    return df.index.get_level_values(level)


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = _date_index(df)
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return df.loc[mask].sort_index()


def _segment_ic(pred: pd.DataFrame, label: pd.DataFrame, start: str, end: str) -> dict[str, Any]:
    pred_s = _slice(pred, start, end).iloc[:, 0]
    label_s = _slice(label, start, end).iloc[:, 0]
    common = pred_s.index.intersection(label_s.index).sort_values()
    if len(common) == 0:
        return {"rows": 0, "IC": float("nan"), "Rank IC": float("nan")}
    ic, rank_ic = calc_ic(pred_s.loc[common], label_s.loc[common])
    return {"rows": int(len(common)), "IC": float(ic.mean()), "Rank IC": float(rank_ic.mean())}


def _seed_diagnostics(seed: int, run: dict[str, Any], pred: pd.DataFrame, label: pd.DataFrame) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details = []
    for segment_name, (start, end) in SEGMENTS.items():
        metrics = _segment_ic(pred, label, start, end)
        group = "full" if segment_name == "valid_all" else ("year" if segment_name.startswith("Y") else "quarter")
        details.append(
            {
                "seed": seed,
                "segment_name": segment_name,
                "segment_group": group,
                "start": start,
                "end": end,
                **metrics,
            }
        )
    detail_df = pd.DataFrame(details)
    full = detail_df[detail_df["segment_group"] == "full"].iloc[0]
    years = detail_df[detail_df["segment_group"] == "year"]["Rank IC"].astype(float)
    quarters = detail_df[detail_df["segment_group"] == "quarter"]["Rank IC"].astype(float)
    row = {
        "seed": seed,
        "summary_path": run.get("summary_path"),
        "cache_path": run.get("cache_path"),
        "full_IC": float(full["IC"]),
        "full_RankIC": float(full["Rank IC"]),
        "year_RankIC_min": float(years.min()),
        "year_RankIC_mean": float(years.mean()),
        "year_RankIC_std": float(years.std(ddof=0)),
        "year_RankIC_spread": float(years.max() - years.min()),
        "quarter_RankIC_min": float(quarters.min()),
        "quarter_RankIC_mean": float(quarters.mean()),
        "quarter_RankIC_std": float(quarters.std(ddof=0)),
        "quarter_RankIC_spread": float(quarters.max() - quarters.min()),
        "quarter_RankIC_positive_ratio": float((quarters > 0).mean()),
    }
    row["seed_stability_score"] = (
        row["quarter_RankIC_min"] * 1000.0
        + row["year_RankIC_min"] * 500.0
        + row["full_RankIC"] * 200.0
        - row["quarter_RankIC_spread"] * 300.0
    )
    row["seed_stability_gate"] = (
        row["full_RankIC"] >= 0.04
        and row["year_RankIC_min"] >= 0.03
        and row["quarter_RankIC_min"] >= 0.015
        and row["quarter_RankIC_positive_ratio"] >= 0.90
    )
    return row, details


def _ranked_daily_signal(pred: pd.DataFrame) -> pd.Series:
    score = pred.iloc[:, 0].sort_index()
    return score.groupby(level=0, group_keys=False).rank(method="average", pct=True)


def _pair_corr(a: pd.Series, b: pd.Series) -> float:
    common = a.index.intersection(b.index).sort_values()
    if len(common) == 0:
        return float("nan")
    return float(a.loc[common].corr(b.loc[common]))


def _combo_rows(seed_df: pd.DataFrame, pair_df: pd.DataFrame, sizes: list[int], limit: int) -> pd.DataFrame:
    eligible = seed_df.sort_values(
        ["seed_stability_gate", "seed_stability_score", "quarter_RankIC_min", "full_RankIC"],
        ascending=[False, False, False, False],
    )
    pool = eligible.head(max(limit, max(sizes) * 4))["seed"].astype(int).tolist()
    pair_lookup = {}
    for row in pair_df.to_dict(orient="records"):
        pair_lookup[tuple(sorted((int(row["seed_a"]), int(row["seed_b"]))))] = float(row["rank_signal_corr"])
    rows = []
    seed_metrics = {int(row["seed"]): row for row in seed_df.to_dict(orient="records")}
    for size in sizes:
        for combo in itertools.combinations(pool, size):
            pair_corrs = [
                pair_lookup.get(tuple(sorted(pair)), 1.0)
                for pair in itertools.combinations(combo, 2)
            ]
            metrics = [seed_metrics[int(seed)] for seed in combo]
            min_quarter = min(float(row["quarter_RankIC_min"]) for row in metrics)
            min_year = min(float(row["year_RankIC_min"]) for row in metrics)
            mean_full = sum(float(row["full_RankIC"]) for row in metrics) / len(metrics)
            mean_score = sum(float(row["seed_stability_score"]) for row in metrics) / len(metrics)
            mean_corr = sum(pair_corrs) / len(pair_corrs)
            max_corr = max(pair_corrs)
            gate_count = sum(bool(row["seed_stability_gate"]) for row in metrics)
            combo_score = mean_score + min_quarter * 600.0 + min_year * 300.0 - mean_corr * 25.0 - max_corr * 25.0
            rows.append(
                {
                    "combo": ",".join(str(seed) for seed in combo),
                    "combo_size": size,
                    "gate_seed_count": gate_count,
                    "seed_score_mean": mean_score,
                    "full_RankIC_mean": mean_full,
                    "year_RankIC_min_of_seeds": min_year,
                    "quarter_RankIC_min_of_seeds": min_quarter,
                    "rank_signal_corr_mean": mean_corr,
                    "rank_signal_corr_max": max_corr,
                    "combo_prefilter_score": combo_score,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["combo_prefilter_score", "gate_seed_count", "quarter_RankIC_min_of_seeds", "full_RankIC_mean"],
        ascending=[False, False, False, False],
    ).head(limit)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only H200 seed pool diagnostics.")
    parser.add_argument("--valid-summary-path", action="append", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--top-seeds", type=int, default=28)
    parser.add_argument("--combo-sizes", default="3,4")
    parser.add_argument("--combo-limit", type=int, default=80)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    seed_path = args.output_prefix.with_name(f"{args.output_prefix.name}_seed_metrics.csv")
    detail_path = args.output_prefix.with_name(f"{args.output_prefix.name}_segment_detail.csv")
    pair_path = args.output_prefix.with_name(f"{args.output_prefix.name}_pair_corr.csv")
    combo_path = args.output_prefix.with_name(f"{args.output_prefix.name}_combo_proposals.csv")
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.json")
    outputs = [seed_path, detail_path, pair_path, combo_path, summary_path]
    if not args.force_recompute and any(path.exists() for path in outputs):
        raise FileExistsError("diagnostic outputs already exist; use --force-recompute")

    seed_runs = _load_seed_runs(args.valid_summary_path)
    mod = rank_runner.load_module()
    pred_by_seed: dict[int, pd.DataFrame] = {}
    rank_signal_by_seed: dict[int, pd.Series] = {}
    seed_rows = []
    detail_rows = []
    for seed, run in sorted(seed_runs.items()):
        pred, label = mod._load_signal(Path(run["cache_path"]))
        pred_by_seed[seed] = pred
        seed_row, details = _seed_diagnostics(seed, run, pred, label)
        seed_rows.append(seed_row)
        detail_rows.extend(details)
        print(
            f"seed={seed} full_rankic={seed_row['full_RankIC']:.6f} "
            f"year_min={seed_row['year_RankIC_min']:.6f} quarter_min={seed_row['quarter_RankIC_min']:.6f} "
            f"gate={seed_row['seed_stability_gate']}"
        )

    seed_df = pd.DataFrame(seed_rows).sort_values(
        ["seed_stability_gate", "seed_stability_score", "quarter_RankIC_min", "full_RankIC"],
        ascending=[False, False, False, False],
    )
    detail_df = pd.DataFrame(detail_rows)
    seed_df.to_csv(seed_path, index=False)
    detail_df.to_csv(detail_path, index=False)

    top_seed_ids = seed_df.head(int(args.top_seeds))["seed"].astype(int).tolist()
    for seed in top_seed_ids:
        rank_signal_by_seed[seed] = _ranked_daily_signal(pred_by_seed[seed])
    pair_rows = []
    for seed_a, seed_b in itertools.combinations(top_seed_ids, 2):
        pair_rows.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "rank_signal_corr": _pair_corr(rank_signal_by_seed[seed_a], rank_signal_by_seed[seed_b]),
            }
        )
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(pair_path, index=False)

    combo_sizes = [int(item) for item in _split_csv(args.combo_sizes)]
    combo_df = _combo_rows(seed_df, pair_df, combo_sizes, int(args.combo_limit))
    combo_df.to_csv(combo_path, index=False)

    summary = {
        "run_mode": "h200_validation_seed_pool_diagnostics",
        "selection_scope": "validation_only_no_test_loaded",
        "test_usage_policy": "test_not_loaded_not_used_for_selection",
        "valid_summary_paths": [str(path) for path in args.valid_summary_path],
        "seed_count": int(len(seed_df)),
        "seed_stability_gate_count": int(seed_df["seed_stability_gate"].sum()),
        "top_seed_count_for_pair_corr": int(len(top_seed_ids)),
        "combo_sizes": combo_sizes,
        "combo_limit": int(args.combo_limit),
        "seed_metrics_path": str(seed_path),
        "segment_detail_path": str(detail_path),
        "pair_corr_path": str(pair_path),
        "combo_proposals_path": str(combo_path),
        "top_seeds": seed_df.head(20).to_dict(orient="records"),
        "top_combos": combo_df.head(20).to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    print(seed_path)
    print(detail_path)
    print(pair_path)
    print(combo_path)
    print(summary_path)


if __name__ == "__main__":
    main()
