from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

from h200_strict_validation_selector import DEFAULT_VALID_SUMMARIES, STRICT_SELECTION_RULE
from tra_local_helpers import localize_value


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SHARD_PREFIX = BASE_DIR / "tmp/h200_strict_validation_selector_cashq_top8_20260705_1_shard"
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp/h200_strict_validation_selector_cashq_top8_20260705_1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
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


def _rank_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        STRICT_SELECTION_RULE["sort"],
        ascending=[False, False, False, False, False],
        na_position="last",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge strict H200 validation-only selector shards.")
    parser.add_argument("--shard-prefix", type=Path, default=DEFAULT_SHARD_PREFIX)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--valid-summary-path", action="append", type=Path, default=None)
    parser.add_argument("--signal-profile", default="cash-quality-quick")
    parser.add_argument("--strategy-profiles", default="stable-core,stable-cash-quick")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    shard_grid_paths = [
        args.shard_prefix.with_name(f"{args.shard_prefix.name}{index:02d}of{args.num_shards:02d}_grid.csv")
        for index in range(args.num_shards)
    ]
    shard_summary_paths = [
        args.shard_prefix.with_name(f"{args.shard_prefix.name}{index:02d}of{args.num_shards:02d}_summary.json")
        for index in range(args.num_shards)
    ]
    missing = [str(path) for path in [*shard_grid_paths, *shard_summary_paths] if not path.exists()]
    if missing:
        raise FileNotFoundError("strict selector shards are not complete: " + ", ".join(missing))

    frames = [pd.read_csv(path) for path in shard_grid_paths]
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise RuntimeError("no validation rows found in shard grids")
    if "selector_key" in df.columns:
        before = len(df)
        df = df.drop_duplicates("selector_key", keep="first")
        if len(df) != before:
            print(f"dropped_duplicate_selector_rows={before - len(df)}")

    df = _rank_rows(df)
    selected = df.iloc[0].to_dict()
    selected_combo = tuple(int(seed) for seed in str(selected["combo"]).split(","))

    grid_path = args.output_prefix.with_name(f"{args.output_prefix.name}_grid.csv")
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.json")
    locked_path = args.output_prefix.with_name(f"{args.output_prefix.name}_locked_manifest.json")
    validation_wrapper_path = args.output_prefix.with_name(f"{args.output_prefix.name}_locked_validation_summary.txt")
    selected_row_path = args.output_prefix.with_name(f"{args.output_prefix.name}_locked_cv_row.csv")

    df.to_csv(grid_path, index=False)
    pd.DataFrame([selected]).to_csv(selected_row_path, index=False)

    valid_summary_paths = args.valid_summary_path or DEFAULT_VALID_SUMMARIES
    seed_runs = _load_valid_seed_runs(valid_summary_paths)
    missing_seeds = [seed for seed in selected_combo if seed not in seed_runs]
    if missing_seeds:
        raise RuntimeError(f"selected combo missing validation seed runs: {missing_seeds}")
    _write_validation_summary(seed_runs, selected_combo, validation_wrapper_path)

    summary = {
        "run_mode": "h200_strict_validation_only_selector_merged",
        "selection_scope": "validation_only_no_test_summary_loaded",
        "test_usage_policy": "no_test_access_before_locked_manifest",
        "source_shard_grid_paths": [str(path) for path in shard_grid_paths],
        "source_shard_summary_paths": [str(path) for path in shard_summary_paths],
        "valid_summary_paths": [str(path) for path in valid_summary_paths],
        "signal_profile": args.signal_profile,
        "strategy_profiles": [item.strip() for item in str(args.strategy_profiles).split(",") if item.strip()],
        "strict_selection_rule": STRICT_SELECTION_RULE,
        "row_count": int(len(df)),
        "strict_pass_count": int(df["strict_pass"].sum()) if "strict_pass" in df else None,
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
