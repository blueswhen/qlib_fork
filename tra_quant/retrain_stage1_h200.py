from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


TRA_QUANT_DIR = Path(__file__).resolve().parent
QLIB_ROOT = TRA_QUANT_DIR.parent
RUNS_DIR = TRA_QUANT_DIR / "runs"
SOURCE_STAGE1 = TRA_QUANT_DIR / "stage1"
DEFAULT_PROVIDER_URI = QLIB_ROOT / "training_data" / "cn_data_latest"
ARTIFACT_SUFFIXES = {".pkl", ".csv", ".json", ".txt", ".log"}
COPY_SUFFIXES = {".py", ".sh", ".md"}

BASELINE = {
    "strategy_trial": "prac_m000_hold3_r085",
    "validation_score": 266.0988237852937,
    "validation_ann": 0.2194366933519968,
    "validation_ir": 1.1263637944285017,
    "validation_mdd": -0.2085999506707754,
    "test_ann": 0.16268627636270552,
    "test_ir": 1.0552511615852953,
    "test_mdd": -0.23068313292472836,
}


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def copy_source_tree(src: Path, dst: Path) -> None:
    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored = {"__pycache__", ".pytest_cache", "mlruns", "tra_cache", "tmp"}
        for name in names:
            path = Path(_dir) / name
            if path.is_file() and path.suffix not in COPY_SUFFIXES:
                ignored.add(name)
        return ignored

    shutil.copytree(src, dst, ignore=ignore)


def assert_no_intermediate_artifacts(stage_dir: Path) -> None:
    offenders = [
        path
        for path in stage_dir.rglob("*")
        if path.is_file() and path.suffix in ARTIFACT_SUFFIXES and path.name != "README.md"
    ]
    if offenders:
        joined = "\n".join(str(path) for path in offenders[:20])
        raise RuntimeError(f"copied intermediate artifacts into fresh run tree:\n{joined}")


def command_env(stage_dir: Path, provider_uri: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["QLIB_PROVIDER_URI"] = str(provider_uri.resolve())
    env.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    existing = env.get("PYTHONPATH", "")
    prefix = os.pathsep.join([str(stage_dir), str(QLIB_ROOT)])
    env["PYTHONPATH"] = os.pathsep.join([prefix, existing]) if existing else prefix
    return env


def run_step(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str],
    gpu_log_path: Path | None = None,
    monitor_gpu_slots: str | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    monitor_thread = None
    if gpu_log_path is not None and monitor_gpu_slots is not None:
        monitor_thread = threading.Thread(
            target=monitor_gpu_utilization,
            args=(gpu_log_path, monitor_gpu_slots, stop_event),
            daemon=True,
        )
        monitor_thread.start()

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"[{datetime.now():%F %T}] START {name}\n")
        log.write("command=" + " ".join(command) + "\n")
        log.flush()
        status = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        log.write(f"\n[{datetime.now():%F %T}] END {name} status={status}\n")

    stop_event.set()
    if monitor_thread is not None:
        monitor_thread.join(timeout=10)
    if status != 0:
        raise RuntimeError(f"{name} failed with status {status}; see {log_path}")


def run_parallel_base_training(
    *,
    stage1_dir: Path,
    log_dir: Path,
    env: dict[str, str],
    seeds: str,
    batch_size: int,
    valid_gpu_slots: str,
    test_gpu_slots: str,
    max_concurrent_seeds: int,
    valid_suffix: str,
    test_suffix: str,
    gpu_log_path: Path,
) -> list[list[str]]:
    log_dir.mkdir(parents=True, exist_ok=True)
    orchestrator_log = log_dir / "01_train_multiseed.log"
    all_gpu_slots = ",".join(
        slot
        for slots in (valid_gpu_slots, test_gpu_slots)
        for slot in (part.strip() for part in slots.split(","))
        if slot
    )
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_gpu_utilization,
        args=(gpu_log_path, all_gpu_slots, stop_event),
        daemon=True,
    )
    monitor_thread.start()

    def command(segment: str, gpu_slots: str, suffix: str) -> list[str]:
        return [
            sys.executable,
            "tune_tra_alpha360_global.py",
            "--dual-seed-study",
            "--dual-seed-base-only",
            "--dual-seed-segment",
            segment,
            "--model-family",
            "tra",
            "--seeds",
            seeds,
            "--result-suffix",
            suffix,
            "--batch-size",
            str(batch_size),
            "--gpu-slots",
            gpu_slots,
            "--max-concurrent-seeds",
            str(max_concurrent_seeds),
            "--min-available-ram-gb",
            "12",
            "--max-extra-swap-gb",
            "128",
            "--resource-poll-seconds",
            "5",
        ]

    jobs = [
        ("valid", command("valid", valid_gpu_slots, valid_suffix), log_dir / "01_train_valid_base.log"),
        ("test", command("test", test_gpu_slots, test_suffix), log_dir / "01_train_test_base.log"),
    ]
    processes: list[tuple[str, subprocess.Popen, object]] = []
    commands = [job[1] for job in jobs]
    try:
        with orchestrator_log.open("w", encoding="utf-8") as log:
            log.write(f"[{datetime.now():%F %T}] START stage1_h200_parallel_base_train\n")
            log.write(f"valid_gpu_slots={valid_gpu_slots}\n")
            log.write(f"test_gpu_slots={test_gpu_slots}\n")
            for segment, cmd, segment_log_path in jobs:
                segment_log = segment_log_path.open("w", encoding="utf-8")
                segment_log.write(f"[{datetime.now():%F %T}] START {segment} base training\n")
                segment_log.write("command=" + " ".join(cmd) + "\n")
                segment_log.flush()
                log.write(f"{segment}_command=" + " ".join(cmd) + "\n")
                processes.append(
                    (
                        segment,
                        subprocess.Popen(cmd, cwd=stage1_dir, env=env, stdout=segment_log, stderr=subprocess.STDOUT),
                        segment_log,
                    )
                )
            log.flush()

            failed = []
            for segment, process, segment_log in processes:
                status = process.wait()
                segment_log.write(f"\n[{datetime.now():%F %T}] END {segment} base training status={status}\n")
                segment_log.close()
                log.write(f"{segment}_status={status}\n")
                log.flush()
                if status != 0:
                    failed.append((segment, status))
                    for other_segment, other_process, _other_log in processes:
                        if other_segment != segment and other_process.poll() is None:
                            other_process.terminate()
            if failed:
                raise RuntimeError(f"parallel base training failed: {failed}; see {orchestrator_log}")
            log.write(f"[{datetime.now():%F %T}] END stage1_h200_parallel_base_train status=0\n")
    finally:
        stop_event.set()
        monitor_thread.join(timeout=10)
        for _segment, process, segment_log in processes:
            if process.poll() is None:
                process.terminate()
            if not segment_log.closed:
                segment_log.close()

    return commands


def run_parallel_search(
    *,
    stage1_dir: Path,
    log_dir: Path,
    env: dict[str, str],
    strategy_profile: str,
    valid_summary: Path,
    test_summary: Path,
    search_prefix: Path,
    num_shards: int,
    reuse_grid_path: Path | None = None,
) -> None:
    orchestrator_log = log_dir / "03_strategy_search.log"
    shard_logs = [
        log_dir / f"03_strategy_search_shard{shard_index:02d}of{num_shards:02d}.log"
        for shard_index in range(num_shards)
    ]
    reuse_args = []
    if reuse_grid_path is not None and reuse_grid_path.exists():
        reuse_args = ["--reuse-grid-path", str(reuse_grid_path)]

    processes: list[tuple[int, subprocess.Popen, object]] = []
    with orchestrator_log.open("w", encoding="utf-8") as log:
        log.write(f"[{datetime.now():%F %T}] START stage1_h200_strategy_search_parallel\n")
        log.write(f"num_shards={num_shards}\n")
        for shard_index, shard_log_path in enumerate(shard_logs):
            shard_prefix = search_prefix.with_name(f"{search_prefix.name}_shard{shard_index:02d}of{num_shards:02d}")
            for suffix in ("_validation_grid.csv", "_summary.json"):
                path = shard_prefix.with_name(f"{shard_prefix.name}{suffix}")
                if path.exists():
                    path.unlink()
            command = [
                sys.executable,
                "run_rank_ensemble_tra_alpha360_once.py",
                "--validation-only",
                "--strategy-profile",
                strategy_profile,
                "--validation-summary-path",
                str(valid_summary),
                "--final-test-summary-path",
                str(test_summary),
                "--output-prefix",
                str(search_prefix),
                "--num-shards",
                str(num_shards),
                "--shard-index",
                str(shard_index),
                *reuse_args,
            ]
            log.write(f"shard{shard_index:02d}_command=" + " ".join(command) + "\n")
            shard_log = shard_log_path.open("w", encoding="utf-8")
            shard_log.write(f"[{datetime.now():%F %T}] START shard {shard_index}/{num_shards}\n")
            shard_log.write("command=" + " ".join(command) + "\n")
            shard_log.flush()
            processes.append(
                (
                    shard_index,
                    subprocess.Popen(command, cwd=stage1_dir, env=env, stdout=shard_log, stderr=subprocess.STDOUT),
                    shard_log,
                )
            )
        log.flush()

        failed = []
        try:
            for shard_index, process, shard_log in processes:
                status = process.wait()
                shard_log.write(f"\n[{datetime.now():%F %T}] END shard {shard_index}/{num_shards} status={status}\n")
                shard_log.close()
                if status != 0:
                    failed.append((shard_index, status))
        finally:
            for _shard_index, process, shard_log in processes:
                if process.poll() is None:
                    process.terminate()
                if not shard_log.closed:
                    shard_log.close()
        if failed:
            log.write(f"failed_shards={failed}\n")
            raise RuntimeError(f"parallel strategy search failed; see {orchestrator_log}")

        merge_command = [
            sys.executable,
            "run_rank_ensemble_tra_alpha360_once.py",
            "--merge-shards",
            "--strategy-profile",
            strategy_profile,
            "--output-prefix",
            str(search_prefix),
            "--num-shards",
            str(num_shards),
        ]
        log.write("merge_command=" + " ".join(merge_command) + "\n")
        log.flush()
        merge_status = subprocess.run(merge_command, cwd=stage1_dir, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if merge_status != 0:
            raise RuntimeError(f"strategy search shard merge failed with status {merge_status}; see {orchestrator_log}")

        final_command = [
            sys.executable,
            "run_rank_ensemble_tra_alpha360_once.py",
            "--strategy-profile",
            strategy_profile,
            "--validation-summary-path",
            str(valid_summary),
            "--final-test-summary-path",
            str(test_summary),
            "--output-prefix",
            str(search_prefix),
        ]
        log.write("final_test_command=" + " ".join(final_command) + "\n")
        log.flush()
        final_status = subprocess.run(final_command, cwd=stage1_dir, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        log.write(f"\n[{datetime.now():%F %T}] END stage1_h200_strategy_search_parallel status={final_status}\n")
        if final_status != 0:
            raise RuntimeError(f"strategy search final test failed with status {final_status}; see {orchestrator_log}")


def monitor_gpu_utilization(path: Path, gpu_slots: str, stop_event: threading.Event) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wanted = {slot.strip() for slot in gpu_slots.split(",") if slot.strip()}
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "gpu", "utilization_gpu", "memory_used_mb", "memory_total_mb"])
        while not stop_event.is_set():
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                timestamp = f"{time.time():.3f}"
                for line in completed.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) != 4 or (wanted and parts[0] not in wanted):
                        continue
                    writer.writerow([timestamp, *parts])
                file.flush()
            stop_event.wait(5.0)


def suffix_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}") if suffix else path


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


def write_seed_summaries(stage1_dir: Path, valid_dual_suffix: str, test_dual_suffix: str, run_id: str) -> tuple[Path, Path]:
    valid_dual_summary = read_kv_summary(
        suffix_path(stage1_dir / "tra_global_dual_seed_pattern_summary.txt", valid_dual_suffix)
    )
    test_dual_summary = read_kv_summary(
        suffix_path(stage1_dir / "tra_global_dual_seed_pattern_summary.txt", test_dual_suffix)
    )
    valid_runs = [
        {"seed": seed, **run}
        for seed, run in sorted(valid_dual_summary["valid_base_runs"].items())
    ]
    test_runs = [
        {"seed": seed, **run}
        for seed, run in sorted(test_dual_summary["test_base_runs"].items())
    ]
    valid_summary = stage1_dir / "tmp" / f"rank_ensemble_h200_validation_summary_{run_id}.txt"
    test_summary = stage1_dir / "tmp" / f"rank_ensemble_h200_final_test_summary_{run_id}.txt"
    common_lines = [
        "window_key=w1",
        "train=['2012-01-01', '2019-12-31']",
        "valid=['2020-01-01', '2022-12-31']",
        "test=['2023-01-01', '2026-04-01']",
    ]
    valid_summary.write_text(
        "\n".join([*common_lines, f"seed_runs_valid={valid_runs}"]) + "\n",
        encoding="utf-8",
    )
    test_summary.write_text(
        "\n".join([*common_lines, f"seed_runs_test={test_runs}"]) + "\n",
        encoding="utf-8",
    )
    return valid_summary, test_summary


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _util_stats(values: list[int]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "mean": sum(values) / len(values) if values else 0.0,
        "max": max(values) if values else 0,
        "share_ge_80": sum(1 for value in values if value >= 80) / len(values) if values else 0.0,
    }


def gpu_utilization_summary(path: Path) -> dict[str, object]:
    rows: dict[str, list[int]] = {}
    active_rows: dict[str, list[int]] = {}
    if not path.exists():
        return {
            "sample_count": 0,
            "active_sample_count": 0,
            "per_gpu": {},
            "all_gpus_mean_at_least_80": False,
            "all_gpus_active_mean_at_least_80": False,
        }
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            gpu = row["gpu"]
            utilization = int(float(row["utilization_gpu"]))
            memory_used_mb = int(float(row["memory_used_mb"]))
            rows.setdefault(gpu, []).append(utilization)
            if memory_used_mb > 1000:
                active_rows.setdefault(gpu, []).append(utilization)
    per_gpu = {
        gpu: {
            "overall": _util_stats(values),
            "active": _util_stats(active_rows.get(gpu, [])),
        }
        for gpu, values in sorted(rows.items(), key=lambda item: int(item[0]))
    }
    return {
        "sample_count": sum(item["overall"]["samples"] for item in per_gpu.values()),
        "active_sample_count": sum(item["active"]["samples"] for item in per_gpu.values()),
        "per_gpu": per_gpu,
        "all_gpus_mean_at_least_80": bool(per_gpu) and all(
            item["overall"]["mean"] >= 80.0 for item in per_gpu.values()
        ),
        "all_gpus_active_mean_at_least_80": bool(per_gpu) and all(
            item["active"]["mean"] >= 80.0 for item in per_gpu.values()
        ),
    }


def baseline_pass(result: dict[str, object]) -> bool:
    validation = result["validation"]
    test = result["test"]
    return (
        validation["score"] >= BASELINE["validation_score"]
        and validation["ann"] >= BASELINE["validation_ann"]
        and validation["ir"] >= BASELINE["validation_ir"]
        and validation["mdd"] >= BASELINE["validation_mdd"]
        and test["ann"] >= BASELINE["test_ann"]
        and test["ir"] >= BASELINE["test_ir"]
        and test["mdd"] >= BASELINE["test_mdd"]
    )


def search_baseline_pass(summary: dict[str, object]) -> bool:
    selected = summary.get("best_validation_selected") or {}
    test = summary.get("selected_test_result") or {}
    return (
        selected.get("validation_score", float("-inf")) >= BASELINE["validation_score"]
        and selected.get("validation_with_cost_ann_return", float("-inf")) >= BASELINE["validation_ann"]
        and selected.get("validation_with_cost_ir", float("-inf")) >= BASELINE["validation_ir"]
        and selected.get("validation_with_cost_mdd", float("-inf")) >= BASELINE["validation_mdd"]
        and test.get("test_with_cost_ann_return", float("-inf")) >= BASELINE["test_ann"]
        and test.get("test_with_cost_ir", float("-inf")) >= BASELINE["test_ir"]
        and test.get("test_with_cost_mdd", float("-inf")) >= BASELINE["test_mdd"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an H200 multi-seed TRA stage1 baseline and evaluate rank ensemble.")
    parser.add_argument("--run-id", default=f"stage1_h200_{now_tag()}")
    parser.add_argument("--seeds", default="42,2026,3407,7,13,99,1234,5678")
    parser.add_argument("--gpu-slots", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--valid-gpu-slots", default="0,1,2,3")
    parser.add_argument("--test-gpu-slots", default="4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--max-concurrent-seeds", type=int, default=1)
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--strategy-trial", default=BASELINE["strategy_trial"])
    parser.add_argument("--strategy-profile", choices=["tra-best-72", "300"], default="tra-best-72")
    parser.add_argument("--run-search", action="store_true")
    parser.add_argument("--search-num-shards", type=int, default=16)
    args = parser.parse_args()

    run_dir = RUNS_DIR / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    provider_uri = args.provider_uri.expanduser().resolve()
    if not provider_uri.exists():
        raise FileNotFoundError(provider_uri)

    stage1_dir = run_dir / "stage1"
    log_dir = run_dir / "logs"
    copy_source_tree(SOURCE_STAGE1, stage1_dir)
    assert_no_intermediate_artifacts(stage1_dir)
    for subdir in ("tmp", "tra_cache", "mlruns"):
        (stage1_dir / subdir).mkdir(parents=True, exist_ok=True)

    env = command_env(stage1_dir, provider_uri)
    dual_suffix = f"{args.run_id}_b{args.batch_size}"
    valid_dual_suffix = f"{dual_suffix}_valid"
    test_dual_suffix = f"{dual_suffix}_test"
    gpu_log = log_dir / "01_gpu_utilization.csv"

    manifest = {
        "run_id": args.run_id,
        "created_at": f"{datetime.now():%F %T}",
        "baseline": BASELINE,
        "provider_uri": str(provider_uri),
        "stage1_dir": str(stage1_dir),
        "seeds": [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()],
        "gpu_slots": args.gpu_slots,
        "valid_gpu_slots": args.valid_gpu_slots,
        "test_gpu_slots": args.test_gpu_slots,
        "batch_size": args.batch_size,
        "valid_dual_suffix": valid_dual_suffix,
        "test_dual_suffix": test_dual_suffix,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "h200_stage1_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train_commands = run_parallel_base_training(
        stage1_dir=stage1_dir,
        log_dir=log_dir,
        env=env,
        seeds=args.seeds,
        batch_size=args.batch_size,
        valid_gpu_slots=args.valid_gpu_slots,
        test_gpu_slots=args.test_gpu_slots,
        max_concurrent_seeds=args.max_concurrent_seeds,
        valid_suffix=valid_dual_suffix,
        test_suffix=test_dual_suffix,
        gpu_log_path=gpu_log,
    )
    manifest["train_commands"] = train_commands
    (run_dir / "h200_stage1_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    valid_summary, test_summary = write_seed_summaries(stage1_dir, valid_dual_suffix, test_dual_suffix, args.run_id)
    repro_json = stage1_dir / "tmp" / f"repro_rank_ensemble_h200_{args.run_id}.json"
    run_step(
        "stage1_h200_fixed_strategy_eval",
        [
            sys.executable,
            "repro_rank_ensemble_single_strategy.py",
            "--strategy-trial",
            args.strategy_trial,
            "--validation-summary-path",
            str(valid_summary),
            "--final-test-summary-path",
            str(test_summary),
            "--output-path",
            str(repro_json),
        ],
        cwd=stage1_dir,
        log_path=log_dir / "02_fixed_strategy_eval.log",
        env=env,
    )

    fixed_result = read_json(repro_json)
    report = {
        "baseline": BASELINE,
        "fixed_strategy_result": fixed_result,
        "fixed_strategy_pass": baseline_pass(fixed_result),
        "validation_summary": str(valid_summary),
        "final_test_summary": str(test_summary),
        "gpu_utilization": gpu_utilization_summary(gpu_log),
    }

    if args.run_search and not report["fixed_strategy_pass"]:
        search_prefix = stage1_dir / "tmp" / f"rank_ensemble_h200_search_{args.run_id}"
        if args.search_num_shards > 1:
            run_parallel_search(
                stage1_dir=stage1_dir,
                log_dir=log_dir,
                env=env,
                strategy_profile=args.strategy_profile,
                valid_summary=valid_summary,
                test_summary=test_summary,
                search_prefix=search_prefix,
                num_shards=args.search_num_shards,
            )
        else:
            run_step(
                "stage1_h200_strategy_search",
                [
                    sys.executable,
                    "run_rank_ensemble_tra_alpha360_once.py",
                    "--force-recompute",
                    "--strategy-profile",
                    args.strategy_profile,
                    "--validation-summary-path",
                    str(valid_summary),
                    "--final-test-summary-path",
                    str(test_summary),
                    "--output-prefix",
                    str(search_prefix),
                ],
                cwd=stage1_dir,
                log_path=log_dir / "03_strategy_search.log",
                env=env,
            )
        search_summary = search_prefix.with_name(f"{search_prefix.name}_summary.json")
        report["search_summary"] = str(search_summary)
        report["search_result"] = read_json(search_summary)
        report["search_pass"] = search_baseline_pass(report["search_result"])
    else:
        report["search_pass"] = False

    report_path = run_dir / "h200_stage1_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(report_path)
    return 0 if report["fixed_strategy_pass"] or report["search_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
