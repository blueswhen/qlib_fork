from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


TRA_QUANT_DIR = Path(__file__).resolve().parent
QLIB_ROOT = TRA_QUANT_DIR.parent
RUNS_DIR = TRA_QUANT_DIR / "runs"
SOURCE_STAGE1 = TRA_QUANT_DIR / "stage1"
SOURCE_STAGE2 = TRA_QUANT_DIR / "stage2"
DEFAULT_PROVIDER_URI = QLIB_ROOT / "training_data" / "cn_data_latest"

ARTIFACT_SUFFIXES = {".pkl", ".csv", ".json", ".txt", ".log"}
COPY_SUFFIXES = {".py", ".sh", ".md"}


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_kv_summary(path: Path) -> dict[str, object]:
    data: dict[str, object] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            data[key] = ast.literal_eval(value)
        except Exception:
            data[key] = value
    return data


def suffix_path(path: Path, suffix: str) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def copy_source_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=False)
    for item in src.iterdir():
        if item.is_file() and item.suffix in COPY_SUFFIXES:
            shutil.copy2(item, dst / item.name)
        elif item.is_dir() and item.name == "docs":
            shutil.copytree(
                item,
                dst / item.name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pkl", "*.csv", "*.json", "*.txt", "*.log"),
            )


def assert_no_intermediate_artifacts(path: Path) -> None:
    offenders = [
        item
        for item in path.rglob("*")
        if item.is_file()
        and item.suffix in ARTIFACT_SUFFIXES
        and "docs" not in item.relative_to(path).parts
    ]
    if offenders:
        formatted = "\n".join(str(item) for item in offenders[:50])
        raise RuntimeError(f"fresh run tree contains forbidden intermediate artifacts:\n{formatted}")


def assert_under_run(path: Path, run_dir: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    run_dir_resolved = run_dir.resolve()
    if run_dir_resolved not in [resolved, *resolved.parents]:
        raise RuntimeError(f"{label} escapes fresh run directory: {resolved}")
    return resolved


def ensure_training_data_link(provider_uri: Path) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    link = RUNS_DIR / "training_data"
    target = QLIB_ROOT / "training_data"
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != target.resolve():
            raise RuntimeError(f"{link} exists but does not point to {target}")
    else:
        link.symlink_to(target, target_is_directory=True)
    if not provider_uri.exists():
        raise FileNotFoundError(f"provider_uri does not exist: {provider_uri}")
    return link


def command_env(stage_dir: Path, provider_uri: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["QLIB_PROVIDER_URI"] = str(provider_uri.resolve())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(stage_dir), str(QLIB_ROOT), existing]) if existing else os.pathsep.join(
        [str(stage_dir), str(QLIB_ROOT)]
    )
    return env


def run_step(name: str, cmd: list[str], *, cwd: Path, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        header = f"[{datetime.now():%F %T}] START {name}\ncmd={cmd}\ncwd={cwd}\n\n"
        print(header, end="")
        log.write(header)
        log.flush()
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        rc = proc.wait()
        footer = f"\n[{datetime.now():%F %T}] END {name} status={rc}\n"
        print(footer, end="")
        log.write(footer)
    if rc != 0:
        raise RuntimeError(f"step failed: {name}; see {log_path}")


def seed42_result(stage1_dir: Path, suffix: str, segment: str) -> dict[str, object]:
    matches = sorted(stage1_dir.glob(f"rolling_result_tra_*_{suffix}_{segment}.txt"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one seed42 {segment} result for suffix {suffix}, found {matches}")
    data = read_kv_summary(matches[0])
    return {
        "seed": 42,
        "cache_path": str(Path(str(data["cache_path"])).resolve()),
        "result_path": str(matches[0].resolve()),
        "recorder_id": str(data.get("recorder_id", "None")),
    }


def dual_seed_runs(summary_path: Path, key: str, wanted_seeds: tuple[int, ...]) -> list[dict[str, object]]:
    summary = read_kv_summary(summary_path)
    raw = summary.get(key)
    if not isinstance(raw, dict):
        raise RuntimeError(f"{key} missing from {summary_path}")
    runs = []
    for seed in wanted_seeds:
        item = raw.get(seed) or raw.get(str(seed))
        if not isinstance(item, dict):
            raise RuntimeError(f"seed {seed} missing from {key} in {summary_path}")
        cache_path = Path(str(item["cache_path"])).resolve()
        if not cache_path.exists():
            raise FileNotFoundError(f"generated cache missing: {cache_path}")
        runs.append(
            {
                "seed": seed,
                "cache_path": str(cache_path),
                "result_path": str(Path(str(item["result_path"])).resolve()),
                "recorder_id": str(item.get("recorder_id", "None")),
            }
        )
    return runs


def write_stage1_wrapper_summaries(
    *,
    stage1_dir: Path,
    run_id: str,
    seed42_suffix: str,
    dual_suffix: str,
) -> tuple[Path, Path]:
    summary_path = suffix_path(stage1_dir / "tra_global_dual_seed_pattern_summary.txt", dual_suffix)
    valid_runs = [
        seed42_result(stage1_dir, seed42_suffix, "valid"),
        *dual_seed_runs(summary_path, "valid_base_runs", (2026, 3407)),
    ]
    test_runs = [
        seed42_result(stage1_dir, seed42_suffix, "test"),
        *dual_seed_runs(summary_path, "test_base_runs", (2026, 3407)),
    ]
    valid_summary = stage1_dir / "tmp" / f"rank_ensemble_old6_validation_summary_{run_id}.txt"
    test_summary = stage1_dir / "tmp" / f"rank_ensemble_old6_final_test_summary_{run_id}.txt"
    valid_summary.parent.mkdir(parents=True, exist_ok=True)
    common = [
        "window_key=w1",
        "train=['2012-01-01', '2019-12-31']",
        "valid=['2020-01-01', '2022-12-31']",
        "test=['2023-01-01', '2026-04-01']",
    ]
    valid_summary.write_text("\n".join(common + [f"seed_runs_valid={valid_runs}"]) + "\n", encoding="utf-8")
    test_summary.write_text("\n".join(common + [f"seed_runs_test={test_runs}"]) + "\n", encoding="utf-8")
    return valid_summary, test_summary


def ensure_paths_inside_run(paths: list[Path], run_dir: Path) -> None:
    for path in paths:
        assert_under_run(path, run_dir, label="path")


def cache_paths_from_summary(summary_path: Path, key: str) -> list[Path]:
    summary = read_kv_summary(summary_path)
    raw = summary.get(key)
    if not isinstance(raw, list):
        raise RuntimeError(f"{key} missing from {summary_path}")
    paths: list[Path] = []
    for item in raw:
        if not isinstance(item, dict) or "cache_path" not in item:
            raise RuntimeError(f"bad {key} item in {summary_path}: {item!r}")
        path = Path(str(item["cache_path"])).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"summary cache path does not exist: {path}")
        paths.append(path)
    return paths


def assert_summary_cache_paths_inside_run(summary_path: Path, key: str, run_dir: Path) -> list[Path]:
    assert_under_run(summary_path, run_dir, label=f"{key} summary")
    paths = cache_paths_from_summary(summary_path, key)
    for path in paths:
        assert_under_run(path, run_dir, label=f"{key} cache")
    return paths


def assert_planned_commands_from_zero(
    planned_commands: dict[str, list[str]],
    *,
    run_dir: Path,
    stage1_valid_summary: Path,
    stage1_test_summary: Path,
    stage1_repro_json: Path,
    stage2_cv_grid: Path,
) -> None:
    run_dir_resolved = run_dir.resolve()

    def require_flag(step: str, flag: str, expected: Path) -> None:
        cmd = planned_commands[step]
        if flag not in cmd:
            raise RuntimeError(f"{step} missing required from-zero flag {flag}")
        value = Path(cmd[cmd.index(flag) + 1]).expanduser().resolve()
        if value != expected.expanduser().resolve():
            raise RuntimeError(f"{step} {flag} points to {value}, expected {expected}")
        assert_under_run(value, run_dir_resolved, label=f"{step} {flag}")

    require_flag("stage1_rank_ensemble_repro", "--validation-summary-path", stage1_valid_summary)
    require_flag("stage1_rank_ensemble_repro", "--final-test-summary-path", stage1_test_summary)
    require_flag("stage1_rank_ensemble_repro", "--output-path", stage1_repro_json)
    require_flag("stage2_cv_search", "--stage1-validation-summary-path", stage1_valid_summary)
    require_flag("stage2_cv_search", "--stage1-repro-json-path", stage1_repro_json)
    require_flag("stage2_final_test_once", "--stage1-validation-summary-path", stage1_valid_summary)
    require_flag("stage2_final_test_once", "--stage1-final-test-summary-path", stage1_test_summary)
    require_flag("stage2_final_test_once", "--cv-grid-path", stage2_cv_grid)
    if "--force-recompute" not in planned_commands["stage2_cv_search"]:
        raise RuntimeError("stage2_cv_search must include --force-recompute in from-zero mode")

    for step, cmd in planned_commands.items():
        for token in cmd:
            if not isinstance(token, str):
                continue
            if str(TRA_QUANT_DIR / "stage1" / "tmp") in token or str(TRA_QUANT_DIR / "stage2" / "tmp") in token:
                raise RuntimeError(f"{step} command references source-tree tmp artifact path: {token}")
            if str(TRA_QUANT_DIR / "stage1" / "tra_cache") in token or str(TRA_QUANT_DIR / "stage2" / "fusion_cache") in token:
                raise RuntimeError(f"{step} command references source-tree cache artifact path: {token}")
            if token.endswith((".pkl", ".csv", ".json", ".txt", ".log")) and str(run_dir_resolved) not in token:
                raise RuntimeError(f"{step} command references an intermediate artifact outside this run: {token}")


def build_manifest(run_dir: Path, payload: dict[str, object]) -> None:
    manifest = run_dir / "from_zero_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-key from-zero retrain for the current TRA stage1 + stage2 best baseline.")
    parser.add_argument("--run-id", default=f"best_baseline_from_zero_{now_tag()}")
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--gpu-slots", default="0,1")
    parser.add_argument("--stage2-num-shards", type=int, default=8)
    parser.add_argument("--stage2-log-dir-name", default="stage2_signal_cv")
    parser.add_argument("--skip-execute", action="store_true", help="create the isolated run tree and command manifest only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = RUNS_DIR / args.run_id
    if run_dir.exists():
        raise FileExistsError(
            f"run directory already exists, refusing to reuse any intermediate artifacts: {run_dir}"
        )
    provider_uri = args.provider_uri.expanduser().resolve()
    ensure_training_data_link(provider_uri)

    stage1_dir = run_dir / "stage1"
    stage2_dir = run_dir / "stage2"
    log_dir = run_dir / "logs"
    copy_source_tree(SOURCE_STAGE1, stage1_dir)
    copy_source_tree(SOURCE_STAGE2, stage2_dir)
    assert_no_intermediate_artifacts(stage1_dir)
    assert_no_intermediate_artifacts(stage2_dir)
    for subdir in ("tmp", "tra_cache", "mlruns"):
        (stage1_dir / subdir).mkdir(parents=True, exist_ok=True)
    for subdir in ("tmp", "fusion_cache", "logs", "mlruns"):
        (stage2_dir / subdir).mkdir(parents=True, exist_ok=True)

    dual_suffix = f"{args.run_id}_dualseed_b12000"
    seed42_suffix = f"{args.run_id}_seed42_refcheck_b16384"
    stage2_cv_suffix = f"{args.run_id}_stage2_cv"
    stage2_cv_prefix = stage2_dir / "tmp" / f"stage2_cv_robust_small_stable_core_{args.run_id}"
    stage2_cv_grid = stage2_cv_prefix.with_name(f"{stage2_cv_prefix.name}_cv_grid.csv")
    stage2_final_suffix = f"{args.run_id}_stage2_final"
    stage2_final_prefix = stage2_dir / "tmp" / f"stage2_cv_cash_quality_z001_hold7r085_final_{args.run_id}"
    stage2_final_summary = stage2_final_prefix.with_name(f"{stage2_final_prefix.name}_summary.json")
    stage1_valid_summary = stage1_dir / "tmp" / f"rank_ensemble_old6_validation_summary_{args.run_id}.txt"
    stage1_test_summary = stage1_dir / "tmp" / f"rank_ensemble_old6_final_test_summary_{args.run_id}.txt"
    stage1_repro_json = stage1_dir / "tmp" / f"repro_rank_ensemble_single_strategy_{args.run_id}.json"

    planned_commands = {
        "stage1_dual_seed": [
            sys.executable,
            "tune_tra_alpha360_global.py",
            "--dual-seed-study",
            "--dual-seed-base-only",
            "--model-family",
            "tra",
            "--result-suffix",
            dual_suffix,
            "--batch-size",
            "12000",
            "--gpu-slots",
            args.gpu_slots,
            "--max-concurrent-seeds",
            "2",
            "--min-available-ram-gb",
            "6",
            "--max-extra-swap-gb",
            "48",
            "--resource-poll-seconds",
            "2",
        ],
        "stage1_seed42_refcheck": [
            sys.executable,
            "retrain_seed42_refcheck.py",
            "--suffix",
            seed42_suffix,
            "--gpu-slots",
            args.gpu_slots,
            "--provider-uri",
            str(provider_uri),
        ],
        "stage1_rank_ensemble_repro": [
            sys.executable,
            "repro_rank_ensemble_single_strategy.py",
            "--strategy-trial",
            "prac_m000_hold3_r085",
            "--validation-summary-path",
            str(stage1_valid_summary),
            "--final-test-summary-path",
            str(stage1_test_summary),
            "--output-path",
            str(stage1_repro_json),
        ],
        "stage2_cv_search": [
            sys.executable,
            "run_stage2_signal_cv_search.py",
            "--output-prefix",
            str(stage2_cv_prefix),
            "--result-suffix",
            stage2_cv_suffix,
            "--signal-profile",
            "robust-small",
            "--strategy-profile",
            "stable-core",
            "--num-shards",
            str(args.stage2_num_shards),
            "--launch-shards",
            "--force-recompute",
            "--log-dir",
            str(stage2_dir / "logs" / args.stage2_log_dir_name),
            "--stage1-validation-summary-path",
            str(stage1_valid_summary),
            "--stage1-repro-json-path",
            str(stage1_repro_json),
        ],
        "stage2_final_test_once": [
            sys.executable,
            "evaluate_stage2_cv_selected_final.py",
            "--cv-grid-path",
            str(stage2_cv_grid),
            "--output-prefix",
            str(stage2_final_prefix),
            "--result-suffix",
            stage2_final_suffix,
            "--signal-profile",
            "robust-small",
            "--stage1-validation-summary-path",
            str(stage1_valid_summary),
            "--stage1-final-test-summary-path",
            str(stage1_test_summary),
        ],
    }
    build_manifest(
        run_dir,
        {
            "run_id": args.run_id,
            "created_at": f"{datetime.now():%F %T}",
            "policy": "from zero: fresh run directory; no existing pkl/csv/json/txt/log artifacts copied; stage2 uses --force-recompute; all stage1 paths injected from this run",
            "provider_uri": str(provider_uri),
            "training_data_link": str((RUNS_DIR / "training_data").resolve()),
            "stage1_dir": str(stage1_dir),
            "stage2_dir": str(stage2_dir),
            "planned_outputs": {
                "stage1_validation_summary": str(stage1_valid_summary),
                "stage1_final_test_summary": str(stage1_test_summary),
                "stage1_repro_json": str(stage1_repro_json),
                "stage2_cv_grid": str(stage2_cv_grid),
                "stage2_final_summary": str(stage2_final_summary),
            },
            "commands": planned_commands,
        },
    )
    assert_planned_commands_from_zero(
        planned_commands,
        run_dir=run_dir,
        stage1_valid_summary=stage1_valid_summary,
        stage1_test_summary=stage1_test_summary,
        stage1_repro_json=stage1_repro_json,
        stage2_cv_grid=stage2_cv_grid,
    )
    print(f"fresh run tree ready: {run_dir}")
    if args.skip_execute:
        print(f"skip-execute set; manifest written to {run_dir / 'from_zero_manifest.json'}")
        return 0

    stage1_env = command_env(stage1_dir, provider_uri)
    stage2_env = command_env(stage2_dir, provider_uri)
    run_step(
        "stage1_dual_seed",
        planned_commands["stage1_dual_seed"],
        cwd=stage1_dir,
        log_path=log_dir / "01_stage1_dual_seed.log",
        env=stage1_env,
    )
    run_step(
        "stage1_seed42_refcheck",
        planned_commands["stage1_seed42_refcheck"],
        cwd=stage1_dir,
        log_path=log_dir / "02_stage1_seed42_refcheck.log",
        env=stage1_env,
    )
    generated_valid_summary, generated_test_summary = write_stage1_wrapper_summaries(
        stage1_dir=stage1_dir,
        run_id=args.run_id,
        seed42_suffix=seed42_suffix,
        dual_suffix=dual_suffix,
    )
    if generated_valid_summary != stage1_valid_summary or generated_test_summary != stage1_test_summary:
        raise RuntimeError("internal summary path mismatch")
    valid_cache_paths = assert_summary_cache_paths_inside_run(stage1_valid_summary, "seed_runs_valid", run_dir)
    test_cache_paths = assert_summary_cache_paths_inside_run(stage1_test_summary, "seed_runs_test", run_dir)
    if len(valid_cache_paths) != 3 or len(test_cache_paths) != 3:
        raise RuntimeError(
            f"stage1 wrapper summaries must contain exactly 3 validation and 3 test caches; "
            f"got {len(valid_cache_paths)} valid, {len(test_cache_paths)} test"
        )
    run_step(
        "stage1_rank_ensemble_repro",
        planned_commands["stage1_rank_ensemble_repro"],
        cwd=stage1_dir,
        log_path=log_dir / "03_stage1_rank_ensemble_repro.log",
        env=stage1_env,
    )
    ensure_paths_inside_run([stage1_repro_json], run_dir)
    if not stage1_repro_json.exists():
        raise FileNotFoundError(f"stage1 repro JSON was not generated: {stage1_repro_json}")
    run_step(
        "stage2_cv_search",
        planned_commands["stage2_cv_search"],
        cwd=stage2_dir,
        log_path=log_dir / "04_stage2_cv_search.log",
        env=stage2_env,
    )
    ensure_paths_inside_run([stage2_cv_grid], run_dir)
    if not stage2_cv_grid.exists():
        raise FileNotFoundError(f"stage2 CV grid was not generated: {stage2_cv_grid}")
    run_step(
        "stage2_final_test_once",
        planned_commands["stage2_final_test_once"],
        cwd=stage2_dir,
        log_path=log_dir / "05_stage2_final_test_once.log",
        env=stage2_env,
    )
    ensure_paths_inside_run([stage2_final_summary], run_dir)
    if not stage2_final_summary.exists():
        raise FileNotFoundError(f"stage2 final summary was not generated: {stage2_final_summary}")
    print(f"from-zero best baseline retrain completed: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
