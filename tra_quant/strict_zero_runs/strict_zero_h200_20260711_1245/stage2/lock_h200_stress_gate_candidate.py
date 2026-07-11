from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import h200_validation_stress_audit as stress_audit
import run_stage2_signal_search_fast as fast


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp/h200_stress_gate_locked_20260706_1"
SORT_COLUMNS = [
    "stress_score",
    "year_ann_min",
    "full_ann",
    "half_ann_min",
    "quarter_ann_min",
    "quarter_rankic_min",
]


def _has_text(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    text = str(value).strip()
    return bool(text) and text.lower() != "nan"


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_combo(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in _split_csv(str(value)))


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


def _rank_passes(df: pd.DataFrame) -> pd.DataFrame:
    for column in SORT_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    sort_columns = [column for column in SORT_COLUMNS if column in df.columns]
    return df.sort_values(sort_columns, ascending=[False] * len(sort_columns), na_position="last")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lock one H200 candidate only after validation stress gate passes.")
    parser.add_argument("--candidate-summary-path", action="append", type=Path, required=True)
    parser.add_argument("--valid-summary-path", action="append", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--signal-profile", default="robust-small")
    parser.add_argument("--target-test-ann", type=float, default=0.2643467300531994)
    parser.add_argument("--allow-weighted", action="store_true")
    parser.add_argument(
        "--require-fused-stage2",
        action="store_true",
        help="Refuse baseline_seq and lock only a candidate with a Stage2 signal overlay.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_prefix = args.output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    ranked_path = output_prefix.with_name(f"{output_prefix.name}_passed_ranked.csv")
    summary_path = output_prefix.with_name(f"{output_prefix.name}_summary.json")
    locked_path = output_prefix.with_name(f"{output_prefix.name}_locked_manifest.json")
    selected_row_path = output_prefix.with_name(f"{output_prefix.name}_locked_cv_row.csv")
    validation_wrapper_path = output_prefix.with_name(f"{output_prefix.name}_locked_validation_summary.txt")
    if not args.force and any(path.exists() for path in (ranked_path, summary_path, locked_path, selected_row_path, validation_wrapper_path)):
        raise FileExistsError("lock output already exists; use --force to overwrite")

    df = pd.concat([pd.read_csv(path) for path in args.candidate_summary_path], ignore_index=True)
    if "selector_key" in df.columns:
        df = df.drop_duplicates("selector_key", keep="last")
    if "stress_gate_pass" not in df.columns:
        raise RuntimeError("candidate summary has no stress_gate_pass column")
    passed = df[df["stress_gate_pass"].fillna(False).astype(bool)].copy()
    if args.require_fused_stage2:
        if "signal_name" not in passed.columns:
            raise RuntimeError("candidate summary has no signal_name column")
        passed = passed[passed["signal_name"].astype(str) != "baseline_seq"].copy()
    if passed.empty:
        reason = " fused Stage2" if args.require_fused_stage2 else ""
        raise RuntimeError(f"no{reason} validation stress-gate-passing candidates; refusing to lock")
    passed = _rank_passes(passed)
    selected = passed.iloc[0].to_dict()
    if _has_text(selected.get("weights")) and not args.allow_weighted:
        raise RuntimeError("selected candidate is weighted; final_once does not reproduce weighted stage1 ensembles")

    combo = _parse_combo(selected["combo"])
    seed_runs = stress_audit._load_valid_seed_runs(args.valid_summary_path)
    missing = [seed for seed in combo if seed not in seed_runs]
    if missing:
        raise RuntimeError(f"selected combo missing validation seed runs: {missing}")
    _write_validation_summary(seed_runs, combo, validation_wrapper_path)
    passed.to_csv(ranked_path, index=False)
    pd.DataFrame([selected]).to_csv(selected_row_path, index=False)

    locked = {
        "locked_at": pd.Timestamp.utcnow().isoformat(),
        "run_mode": "strict_validation_locked_candidate",
        "selection_scope": "validation_only_stress_gate_passed_before_test",
        "test_usage_policy": "test_may_be_run_once_after_this_manifest",
        "selected_row": selected,
        "cv_grid_paths": [str(path.expanduser().resolve()) for path in args.candidate_summary_path],
        "locked_cv_row_path": str(selected_row_path),
        "validation_summary_path": str(validation_wrapper_path),
        "final_test_summary_path": None,
        "signal_profile": args.signal_profile,
        "strategy_profile": selected.get("strategy_profile", "stable-core"),
        "target_test_ann": float(args.target_test_ann),
        "hard_constraints": {"account": 150000, "topk": 5, "universe": "csi300"},
        "requires_fused_stage2": bool(args.require_fused_stage2),
    }
    summary = {
        "run_mode": "h200_stress_gate_lock",
        "selection_scope": "validation_only_stress_gate_passed_before_test",
        "test_usage_policy": "no_test_access_before_locked_manifest",
        "candidate_summary_paths": [str(path.expanduser().resolve()) for path in args.candidate_summary_path],
        "valid_summary_paths": [str(path) for path in args.valid_summary_path],
        "passed_count": int(len(passed)),
        "requires_fused_stage2": bool(args.require_fused_stage2),
        "selected_row": selected,
        "passed_ranked_path": str(ranked_path),
        "locked_manifest_path": str(locked_path),
        "locked_cv_row_path": str(selected_row_path),
        "locked_validation_summary_path": str(validation_wrapper_path),
    }
    summary_path.write_text(json.dumps(fast._jsonable(summary), indent=2, ensure_ascii=True), encoding="utf-8")
    locked_path.write_text(json.dumps(fast._jsonable(locked), indent=2, ensure_ascii=True), encoding="utf-8")
    print(locked_path)
    print(validation_wrapper_path)
    print(selected_row_path)
    print(f"locked_combo={selected['combo']} locked_trial={selected['trial_name']}")


if __name__ == "__main__":
    main()
