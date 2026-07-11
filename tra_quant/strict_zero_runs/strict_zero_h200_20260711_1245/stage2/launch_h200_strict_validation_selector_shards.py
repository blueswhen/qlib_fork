from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = BASE_DIR.parent.parent
DEFAULT_OUTPUT_PREFIX = BASE_DIR / "tmp/h200_strict_validation_selector_expanded_top12_20260705_1"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch strict validation-only selector shards with log redirection.")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--valid-summary-path", action="append", type=Path, default=None)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--top-valid-seeds", type=int, default=12)
    parser.add_argument("--combo-sizes", default="3,4")
    parser.add_argument("--combo", action="append", default=None)
    parser.add_argument("--signal-profile", default="cash-quality-quick")
    parser.add_argument("--strategy-profiles", default="stable-core,stable-cash-quick")
    parser.add_argument("--trial-name", action="append", default=None)
    parser.add_argument("--all-profile-trials", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--log-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")

    output_prefix = args.output_prefix.expanduser().resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    shard_prefix = output_prefix.with_name(f"{output_prefix.name}_shard")
    log_dir = (args.log_dir or BASE_DIR / "logs" / output_prefix.name).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    env["PYTHONPATH"] = f"{BASE_DIR}:{WORKSPACE}:{env.get('PYTHONPATH', '')}"
    env.setdefault("QLIB_PROVIDER_URI", str(WORKSPACE / "training_data/cn_data_latest"))

    launched = []
    for index in range(args.num_shards):
        shard_output_prefix = shard_prefix.with_name(f"{shard_prefix.name}{index:02d}of{args.num_shards:02d}")
        command = [
            args.python,
            "h200_strict_validation_selector.py",
            "--output-prefix",
            str(shard_output_prefix),
            "--top-valid-seeds",
            str(args.top_valid_seeds),
            "--combo-sizes",
            args.combo_sizes,
            "--signal-profile",
            args.signal_profile,
            "--strategy-profiles",
            args.strategy_profiles,
            "--num-shards",
            str(args.num_shards),
            "--shard-index",
            str(index),
        ]
        if args.force_recompute:
            command.append("--force-recompute")
        if args.all_profile_trials:
            command.append("--all-profile-trials")
        for trial_name in args.trial_name or []:
            command.extend(["--trial-name", trial_name])
        for combo in args.combo or []:
            command.extend(["--combo", combo])
        for path in args.valid_summary_path or []:
            command.extend(["--valid-summary-path", str(path.expanduser().resolve())])

        log_path = log_dir / f"shard{index:02d}of{args.num_shards:02d}.log"
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=BASE_DIR,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        launched.append(
            {
                "shard_index": index,
                "pid": process.pid,
                "command": command,
                "log_path": log_path,
                "output_prefix": shard_output_prefix,
            }
        )

    manifest = {
        "run_mode": "strict_validation_only_selector_shard_launcher",
        "selection_scope": "validation_only_no_test_summary_loaded",
        "test_usage_policy": "no_test_access_before_locked_manifest",
        "output_prefix": output_prefix,
        "shard_prefix": shard_prefix,
        "num_shards": args.num_shards,
        "top_valid_seeds": args.top_valid_seeds,
        "combo_sizes": args.combo_sizes,
        "explicit_combos": args.combo or None,
        "signal_profile": args.signal_profile,
        "strategy_profiles": args.strategy_profiles,
        "trial_names": args.trial_name or None,
        "all_profile_trials": bool(args.all_profile_trials),
        "valid_summary_paths": [path.expanduser().resolve() for path in args.valid_summary_path or []],
        "launched_at": pd.Timestamp.utcnow().isoformat(),
        "launched": launched,
    }
    manifest_path = output_prefix.with_name(f"{output_prefix.name}_launcher_manifest.json")
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, ensure_ascii=True), encoding="utf-8")
    print(manifest_path)
    for item in launched:
        print(f"pid={item['pid']} log={item['log_path']} output_prefix={item['output_prefix']}")


if __name__ == "__main__":
    main()
