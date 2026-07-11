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
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp/h200_quarter_ic_ensemble_screen_20260705_1"
DEFAULT_GATE = {
    "gate_version": "quarter_rankic_stability_v1_20260705",
    "rankic_valid_all_min": 0.055,
    "rankic_year_min_min": 0.045,
    "rankic_quarter_min_min": 0.025,
    "rankic_quarter_positive_ratio_min": 1.0,
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
        raise RuntimeError(f"test seed runs are not allowed in quarterly IC screen: {path} keys={forbidden}")
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


def _date_segments() -> dict[str, tuple[str, str]]:
    segments: dict[str, tuple[str, str]] = {"valid_all": ("2020-01-01", "2022-12-31")}
    for year in (2020, 2021, 2022):
        segments[f"Y{year}"] = (f"{year}-01-01", f"{year}-12-31")
    for year in (2020, 2021, 2022):
        for quarter, start, end in (
            (1, "01-01", "03-31"),
            (2, "04-01", "06-30"),
            (3, "07-01", "09-30"),
            (4, "10-01", "12-31"),
        ):
            segments[f"Q{quarter}_{year}"] = (f"{year}-{start}", f"{year}-{end}")
    return segments


def _slice_signal(df: pd.DataFrame, segment: tuple[str, str]) -> pd.DataFrame:
    dates = df.index.get_level_values("datetime" if "datetime" in df.index.names else 0)
    mask = (dates >= pd.Timestamp(segment[0])) & (dates <= pd.Timestamp(segment[1]))
    return df.loc[mask].sort_index()


def _segment_rankic(pred: pd.DataFrame, label: pd.DataFrame, segment: tuple[str, str]) -> tuple[float, float, int]:
    pred_s = _slice_signal(pred, segment).iloc[:, 0]
    label_s = _slice_signal(label, segment).iloc[:, 0]
    common = pred_s.index.intersection(label_s.index).sort_values()
    ic, rank_ic = calc_ic(pred_s.loc[common], label_s.loc[common])
    return float(ic.mean()), float(rank_ic.mean()), int(len(common))


def _summarize_rankic(prefix: str, values: list[float]) -> dict[str, Any]:
    series = pd.Series(values, dtype="float64")
    return {
        f"{prefix}_min": float(series.min()),
        f"{prefix}_mean": float(series.mean()),
        f"{prefix}_median": float(series.median()),
        f"{prefix}_std": float(series.std(ddof=0)),
        f"{prefix}_positive_ratio": float((series > 0).mean()),
    }


def _metric_row(key: str, pred: pd.DataFrame, label: pd.DataFrame, segments: dict[str, tuple[str, str]]) -> dict[str, Any]:
    row: dict[str, Any] = {"key": key, "rows": int(len(pred))}
    year_rankics: list[float] = []
    quarter_rankics: list[float] = []
    for name, segment in segments.items():
        ic, rank_ic, rows = _segment_rankic(pred, label, segment)
        row[f"IC_{name}"] = ic
        row[f"RankIC_{name}"] = rank_ic
        row[f"rows_{name}"] = rows
        if name.startswith("Y"):
            year_rankics.append(rank_ic)
        elif name.startswith("Q"):
            quarter_rankics.append(rank_ic)
    row.update(_summarize_rankic("rankic_year", year_rankics))
    row.update(_summarize_rankic("rankic_quarter", quarter_rankics))
    row["rankic_valid_all"] = row["RankIC_valid_all"]
    row["quarter_ic_score"] = (
        row["rankic_quarter_min"] * 1000
        + row["rankic_quarter_mean"] * 250
        + row["rankic_year_min"] * 400
        + row["rankic_valid_all"] * 100
        - row["rankic_quarter_std"] * 300
    )
    row["quarter_ic_gate_pass"] = (
        row["rankic_valid_all"] >= DEFAULT_GATE["rankic_valid_all_min"]
        and row["rankic_year_min"] >= DEFAULT_GATE["rankic_year_min_min"]
        and row["rankic_quarter_min"] >= DEFAULT_GATE["rankic_quarter_min_min"]
        and row["rankic_quarter_positive_ratio"] >= DEFAULT_GATE["rankic_quarter_positive_ratio_min"]
    )
    weak = [
        name
        for name in segments
        if name.startswith("Q") and row[f"RankIC_{name}"] < DEFAULT_GATE["rankic_quarter_min_min"]
    ]
    row["weak_quarters"] = ",".join(weak)
    return row


def _rank_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.sort_values(
        ["quarter_ic_gate_pass", "quarter_ic_score", "rankic_quarter_min", "rankic_year_min", "rankic_valid_all"],
        ascending=[False, False, False, False, False],
        na_position="last",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only quarterly RankIC screen for H200 seed ensembles.")
    parser.add_argument("--valid-summary-path", action="append", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--seed-pool", default="")
    parser.add_argument("--top-seeds", type=int, default=12)
    parser.add_argument("--combo-sizes", default="3,4")
    parser.add_argument("--max-combos", type=int, default=1000)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    seed_path = args.output_prefix.with_name(f"{args.output_prefix.name}_seed_rankic.csv")
    combo_path = args.output_prefix.with_name(f"{args.output_prefix.name}_combo_rankic.csv")
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.json")
    if not args.force_recompute and (seed_path.exists() or combo_path.exists() or summary_path.exists()):
        raise FileExistsError("quarter IC screen output already exists; use --force-recompute to overwrite")

    seed_runs = _load_valid_seed_runs(args.valid_summary_path)
    if args.seed_pool:
        seed_pool = [int(seed) for seed in _split_csv(args.seed_pool)]
    else:
        seed_pool = sorted(seed_runs)
    missing = [seed for seed in seed_pool if seed not in seed_runs]
    if missing:
        raise RuntimeError(f"seed pool missing validation runs: {missing}")

    mod = rank_runner.load_module()
    pred_by_seed: dict[int, pd.DataFrame] = {}
    label_by_seed: dict[int, pd.DataFrame] = {}
    common = None
    for seed in seed_pool:
        pred, label = mod._load_signal(Path(seed_runs[seed]["cache_path"]))
        pred_by_seed[seed] = pred.sort_index()
        label_by_seed[seed] = label.sort_index()
        common = pred.index if common is None else common.intersection(pred.index)
    if common is None:
        raise RuntimeError("no seeds loaded")
    common = common.sort_values()

    ranked_pred_by_seed: dict[int, pd.DataFrame] = {}
    label = label_by_seed[seed_pool[0]].loc[common].sort_index()
    segments = _date_segments()
    seed_rows = []
    for seed in seed_pool:
        pred = pred_by_seed[seed].loc[common].sort_index()
        ranked = pred.groupby(level=0, group_keys=False).rank(method="average", pct=True)
        ranked_pred_by_seed[seed] = ranked
        row = _metric_row(str(seed), ranked, label, segments)
        row["seed"] = seed
        seed_rows.append(row)
        print(
            f"seed={seed} rankic_valid={row['rankic_valid_all']:.6f} "
            f"quarter_min={row['rankic_quarter_min']:.6f} gate={row['quarter_ic_gate_pass']}"
        )
    seed_df = _rank_df(pd.DataFrame(seed_rows))
    seed_df.to_csv(seed_path, index=False)

    selected_seeds = seed_df.head(int(args.top_seeds))["seed"].astype(int).tolist()
    combo_sizes = [int(item) for item in _split_csv(args.combo_sizes)]
    combos = [
        tuple(combo)
        for size in combo_sizes
        for combo in itertools.combinations(selected_seeds, size)
    ]
    combo_rows = []
    for combo in combos[: int(args.max_combos)]:
        pred = None
        for seed in combo:
            item = ranked_pred_by_seed[seed]
            pred = item.copy() if pred is None else pred.add(item, fill_value=0.0)
        pred = pred / float(len(combo))
        row = _metric_row(",".join(str(seed) for seed in combo), pred, label, segments)
        row["combo"] = ",".join(str(seed) for seed in combo)
        row["combo_size"] = len(combo)
        combo_rows.append(row)
        if len(combo_rows) % 50 == 0:
            _rank_df(pd.DataFrame(combo_rows)).to_csv(combo_path, index=False)
            print(f"evaluated_combos={len(combo_rows)}")
    combo_df = _rank_df(pd.DataFrame(combo_rows))
    combo_df.to_csv(combo_path, index=False)

    summary = {
        "run_mode": "h200_quarter_ic_ensemble_screen",
        "selection_scope": "validation_only_no_test_loaded",
        "test_usage_policy": "consumed_test_not_loaded_not_used_for_selection",
        "valid_summary_paths": [str(path) for path in args.valid_summary_path],
        "seed_pool": seed_pool,
        "selected_seed_pool": selected_seeds,
        "combo_sizes": combo_sizes,
        "max_combos": int(args.max_combos),
        "gate": DEFAULT_GATE,
        "common_rows": int(len(common)),
        "seed_count": int(len(seed_df)),
        "seed_gate_pass_count": int(seed_df["quarter_ic_gate_pass"].sum()),
        "combo_count": int(len(combo_df)),
        "combo_gate_pass_count": int(combo_df["quarter_ic_gate_pass"].sum()) if not combo_df.empty else 0,
        "seed_rankic_path": str(seed_path),
        "combo_rankic_path": str(combo_path),
        "top_seed_rankic": seed_df.head(20).to_dict(orient="records"),
        "top_combo_rankic": combo_df.head(20).to_dict(orient="records") if not combo_df.empty else [],
    }
    summary_path.write_text(json.dumps(fast._jsonable(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    print(seed_path)
    print(combo_path)
    print(summary_path)


if __name__ == "__main__":
    main()
