from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

from h200_strict_validation_selector import DEFAULT_VALID_SUMMARIES
from tra_local_helpers import localize_value


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_GRID = BASE_DIR / "tmp/h200_strict_validation_selector_old_explicit_20260705_2_grid.csv"
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp/h200_strict_validation_selector_old_explicit_20260705_2_balanced"
BALANCED_SELECTION_RULE = {
    "full_ann_min": 0.20,
    "year_ann_min_min": 0.15,
    "year_ann_std_max": 0.10,
    "full_mdd_min": -0.25,
    "sort": [
        "balanced_pass",
        "robust_ann_min",
        "year_ann_min",
        "full_ann",
        "stable_score",
        "full_ir",
    ],
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


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
    return data


def _seed_runs_from_base_summary(path: Path) -> dict[int, dict[str, Any]]:
    summary = _parse_summary(path)
    runs = summary.get("valid_base_runs", {})
    if isinstance(runs, list):
        return {int(row["seed"]): row for row in runs}
    if isinstance(runs, dict):
        return {int(seed): dict(row, seed=int(seed)) for seed, row in runs.items()}
    raise TypeError(f"unsupported valid_base_runs in {path}: {type(runs).__name__}")


def _load_valid_seed_runs(paths: list[Path]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for path in paths:
        out.update(_seed_runs_from_base_summary(path))
    return out


def _write_validation_summary(seed_runs: dict[int, dict[str, Any]], combo: tuple[int, ...], path: Path) -> None:
    rows = []
    for seed in combo:
        row = dict(seed_runs[seed])
        row["seed"] = int(seed)
        rows.append(row)
    lines = [
        "window_key=w1",
        "train=['2012-01-01', '2019-12-31']",
        "valid=['2020-01-01', '2022-12-31']",
        "test=['2023-01-01', '2026-04-01']",
        f"seed_runs_valid={rows!r}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _add_balanced_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in ["full_ann", "year_ann_min", "year_ann_std", "full_mdd"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["robust_ann_min"] = df[["full_ann", "year_ann_min"]].min(axis=1)
    df["balanced_pass"] = (
        (df["full_ann"] >= BALANCED_SELECTION_RULE["full_ann_min"])
        & (df["year_ann_min"] >= BALANCED_SELECTION_RULE["year_ann_min_min"])
        & (df["year_ann_std"] <= BALANCED_SELECTION_RULE["year_ann_std_max"])
        & (df["full_mdd"] >= BALANCED_SELECTION_RULE["full_mdd_min"])
    )
    return df


def _rank_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = _add_balanced_columns(df)
    return df.sort_values(
        BALANCED_SELECTION_RULE["sort"],
        ascending=[False, False, False, False, False, False],
        na_position="last",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock one strict H200 validation candidate from completed validation grids.")
    parser.add_argument("--grid-path", action="append", type=Path, default=None)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--valid-summary-path", action="append", type=Path, default=None)
    parser.add_argument("--signal-profile", default="cash-quality-quick")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid_paths = args.grid_path or [DEFAULT_GRID]
    valid_summary_paths = args.valid_summary_path or DEFAULT_VALID_SUMMARIES
    frames = [pd.read_csv(path) for path in grid_paths]
    df = pd.concat(frames, ignore_index=True)
    if "selector_key" in df.columns:
        df = df.drop_duplicates("selector_key", keep="first")
    df = _rank_rows(df)
    selected = df.iloc[0].to_dict()
    selected_combo = tuple(int(seed) for seed in str(selected["combo"]).split(","))

    output_prefix = args.output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    grid_path = output_prefix.with_name(f"{output_prefix.name}_grid.csv")
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.json")
    locked_path = output_prefix.with_name(f"{output_prefix.name}_locked_manifest.json")
    selected_row_path = output_prefix.with_name(f"{output_prefix.name}_locked_cv_row.csv")
    validation_wrapper_path = output_prefix.with_name(f"{output_prefix.name}_locked_validation_summary.txt")

    df.to_csv(grid_path, index=False)
    pd.DataFrame([selected]).to_csv(selected_row_path, index=False)
    seed_runs = _load_valid_seed_runs(valid_summary_paths)
    missing = [seed for seed in selected_combo if seed not in seed_runs]
    if missing:
        raise RuntimeError(f"selected combo missing validation seed runs: {missing}")
    _write_validation_summary(seed_runs, selected_combo, validation_wrapper_path)

    summary = {
        "run_mode": "h200_strict_validation_only_balanced_lock",
        "selection_scope": "validation_only_no_test_summary_loaded",
        "test_usage_policy": "no_test_access_before_locked_manifest",
        "source_grid_paths": [str(path) for path in grid_paths],
        "valid_summary_paths": [str(path) for path in valid_summary_paths],
        "selection_rule": BALANCED_SELECTION_RULE,
        "row_count": int(len(df)),
        "balanced_pass_count": int(df["balanced_pass"].sum()),
        "selected_row": selected,
        "grid_path": str(grid_path),
        "locked_cv_row_path": str(selected_row_path),
        "locked_manifest_path": str(locked_path),
        "locked_validation_summary_path": str(validation_wrapper_path),
    }
    locked = {
        "locked_at": pd.Timestamp.utcnow().isoformat(),
        "run_mode": "strict_validation_locked_candidate",
        "selection_scope": "validation_only_no_test_summary_loaded",
        "test_usage_policy": "test_may_be_run_once_after_this_manifest",
        "selected_row": selected,
        "cv_grid_path": str(grid_path),
        "locked_cv_row_path": str(selected_row_path),
        "validation_summary_path": str(validation_wrapper_path),
        "final_test_summary_path": None,
        "signal_profile": args.signal_profile,
        "strategy_profile": selected.get("strategy_profile", "stable-core"),
        "target_test_ann": 0.2804402793535583,
    }
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    locked_path.write_text(json.dumps(_jsonable(locked), indent=2, ensure_ascii=True), encoding="utf-8")
    print(grid_path)
    print(summary_path)
    print(locked_path)
    print(validation_wrapper_path)
    print(selected_row_path)
    print(f"locked_combo={selected['combo']} locked_trial={selected['trial_name']}")


if __name__ == "__main__":
    main()
