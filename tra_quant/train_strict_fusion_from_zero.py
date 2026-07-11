from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRA_QUANT_DIR = Path(__file__).resolve().parent
QLIB_ROOT = TRA_QUANT_DIR.parent
DEFAULT_RUN_ROOT = TRA_QUANT_DIR / "strict_zero_runs"
SOURCE_STAGE1 = TRA_QUANT_DIR / "stage1"
SOURCE_STAGE2 = TRA_QUANT_DIR / "stage2"
DEFAULT_PROVIDER_URI = QLIB_ROOT / "training_data" / "cn_data_latest"

SEEDS = (5678, 2050, 2044)
BATCH_SIZE = 12000
ROLLING_TASKS_PER_SEED = 4
TARGET_TEST_ANN = 0.2804402793535583
TRIAL_NAMES = (
    "cash_quality_z_tw001__prac_m000_hold5_r08",
    "cash_quality_z_tw001__prac_m000_hold5_r085",
    "cash_quality_z_tw001__prac_m000_hold5_r09",
    "cash_quality_z_tw001__prac_m000_hold5_r095",
    "cash_quality_z_tw001__prac_m000_hold7_r08",
    "cash_quality_z_tw001__prac_m000_hold7_r085",
    "cash_quality_z_tw001__prac_m000_hold7_r09",
    "cash_quality_z_tw001__prac_m000_hold7_r095",
    "cash_quality_z_tw001__prac_m000_hold10_r08",
    "cash_quality_z_tw001__prac_m000_hold10_r085",
    "cash_quality_z_tw001__prac_m000_hold10_r09",
    "cash_quality_z_tw001__prac_m000_hold10_r095",
)
STRESS_GATE = {
    "full_ann_min": 0.24,
    "year_ann_min_min": 0.18,
    "half_ann_min_min": 0.03,
    "quarter_ann_min_min": -0.25,
    "quarter_positive_ratio_min": 0.75,
    "quarter_rankic_min": 0.02,
    "quarter_rankic_positive_ratio_min": 1.0,
}
COPY_SUFFIXES = {".py", ".sh", ".md"}
ARTIFACT_SUFFIXES = {".pkl", ".csv", ".json", ".txt", ".log"}


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    memory_total_mib: int
    memory_free_mib: int


@dataclass(frozen=True)
class ResourcePlan:
    gpu_slots: str
    gpu_count: int
    jobs_per_gpu: int
    max_parallel_stage1_tasks: int
    stage2_workers: int
    threads_per_worker: int
    cpu_count: int
    ram_total_gib: float
    ram_available_gib: float
    min_available_ram_gib: float


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, set)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def read_kv_summary(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            data[key] = ast.literal_eval(value)
        except Exception:
            data[key] = value
    return data


def parse_seed_map(summary_path: Path, key: str) -> dict[int, dict[str, Any]]:
    raw = read_kv_summary(summary_path).get(key)
    if not isinstance(raw, dict):
        raise RuntimeError(f"{key} missing from {summary_path}")
    return {int(seed): dict(item) for seed, item in raw.items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_metadata_fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    total_size = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        stat = item.stat()
        record = f"{item.relative_to(path)}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
        digest.update(record.encode("utf-8"))
        count += 1
        total_size += stat.st_size
    return {"path": str(path), "file_count": count, "total_size": total_size, "metadata_sha256": digest.hexdigest()}


def source_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def detect_gpus() -> list[GPUInfo]:
    output = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        raise RuntimeError("no NVIDIA GPUs detected; this training flow requires CUDA GPUs")
    gpus: list[GPUInfo] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise RuntimeError(f"unexpected nvidia-smi row: {line}")
        gpus.append(GPUInfo(int(parts[0]), parts[1], int(parts[2]), int(parts[3])))
    return gpus


def visible_gpu_indices(gpus: list[GPUInfo], requested: str) -> list[int]:
    available = {gpu.index for gpu in gpus}
    if requested != "auto":
        selected = [int(item.strip()) for item in requested.split(",") if item.strip()]
    else:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cuda_visible and all(part.strip().isdigit() for part in cuda_visible.split(",")):
            selected = [int(part.strip()) for part in cuda_visible.split(",")]
        else:
            selected = sorted(available)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError(f"invalid GPU selection: {selected}")
    missing = [index for index in selected if index not in available]
    if missing:
        raise ValueError(f"requested GPU indices are unavailable: {missing}; available={sorted(available)}")
    return selected


def memory_info() -> tuple[float, float]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = int(raw.strip().split()[0])
    return values["MemTotal"] / 1024**2, values["MemAvailable"] / 1024**2


def build_resource_plan(
    gpus: list[GPUInfo],
    selected_indices: list[int],
    seed_count: int,
    jobs_per_gpu_override: int,
    stage2_workers_override: int,
) -> ResourcePlan:
    selected = [gpu for gpu in gpus if gpu.index in selected_indices]
    cpu_count = os.cpu_count() or 1
    ram_total_gib, ram_available_gib = memory_info()
    total_tasks = seed_count * ROLLING_TASKS_PER_SEED
    min_free_gib = min(gpu.memory_free_mib for gpu in selected) / 1024

    # Keep batch size fixed across machines. Extra VRAM is used by concurrent
    # independent rolling tasks, which does not change model semantics.
    vram_jobs = max(1, min(4, int(max(0.0, min_free_gib - 4.0) // 24.0)))
    ram_global_jobs = max(1, int(max(0.0, ram_available_gib - 16.0) // 18.0))
    useful_jobs_per_gpu = max(1, math.ceil(total_tasks / len(selected)))
    auto_jobs = max(1, min(vram_jobs, useful_jobs_per_gpu, max(1, ram_global_jobs // len(selected))))
    jobs_per_gpu = jobs_per_gpu_override or auto_jobs
    if jobs_per_gpu < 1:
        raise ValueError("jobs per GPU must be positive")

    max_parallel = min(total_tasks, len(selected) * jobs_per_gpu, ram_global_jobs)
    threads_per_worker = max(1, min(8, cpu_count // max(1, max_parallel)))
    auto_stage2_workers = max(
        1,
        min(
            len(TRIAL_NAMES),
            max(1, cpu_count // 12),
            max(1, int(max(0.0, ram_available_gib - 16.0) // 32.0)),
        ),
    )
    stage2_workers = stage2_workers_override or auto_stage2_workers
    min_available_ram_gib = min(32.0, max(6.0, ram_total_gib * 0.02))
    return ResourcePlan(
        gpu_slots=",".join(str(index) for index in selected_indices),
        gpu_count=len(selected),
        jobs_per_gpu=jobs_per_gpu,
        max_parallel_stage1_tasks=max_parallel,
        stage2_workers=max(1, min(stage2_workers, len(TRIAL_NAMES))),
        threads_per_worker=threads_per_worker,
        cpu_count=cpu_count,
        ram_total_gib=ram_total_gib,
        ram_available_gib=ram_available_gib,
        min_available_ram_gib=min_available_ram_gib,
    )


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


def assert_clean_source_snapshot(path: Path) -> None:
    offenders = [
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix in ARTIFACT_SUFFIXES and "docs" not in item.relative_to(path).parts
    ]
    if offenders:
        raise RuntimeError(f"source snapshot contains training artifacts: {offenders[:20]}")


def assert_under_run(path: Path, run_dir: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    root = run_dir.resolve()
    if root not in [resolved, *resolved.parents]:
        raise RuntimeError(f"{label} escapes fresh run directory: {resolved}")
    return resolved


def ensure_training_data_link(run_root: Path, provider_uri: Path) -> Path:
    run_root.mkdir(parents=True, exist_ok=True)
    source = QLIB_ROOT / "training_data"
    link = run_root / "training_data"
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != source.resolve():
            raise RuntimeError(f"{link} must be a symlink to {source}")
    else:
        link.symlink_to(source, target_is_directory=True)
    if not provider_uri.exists():
        raise FileNotFoundError(f"Qlib provider does not exist: {provider_uri}")
    return link


def environment_snapshot(gpus: list[GPUInfo], resource_plan: ResourcePlan) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("torch", "numpy", "pandas", "pyqlib", "mlflow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": command_output(["uname", "-a"]),
        "git_commit": command_output(["git", "-C", str(QLIB_ROOT), "rev-parse", "HEAD"]),
        "git_tracked_status": command_output(
            ["git", "-C", str(QLIB_ROOT), "status", "--short", "--untracked-files=no"]
        ),
        "nvidia_driver": command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "gpus": [asdict(gpu) for gpu in gpus],
        "resource_plan": asdict(resource_plan),
        "packages": packages,
    }


def command_env(stage_dir: Path, provider_uri: Path, threads_per_worker: int) -> dict[str, str]:
    env = os.environ.copy()
    env["QLIB_PROVIDER_URI"] = str(provider_uri.resolve())
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    env["OMP_NUM_THREADS"] = str(threads_per_worker)
    env["MKL_NUM_THREADS"] = str(threads_per_worker)
    env["NUMEXPR_MAX_THREADS"] = str(threads_per_worker)
    existing = env.get("PYTHONPATH", "")
    paths = [str(stage_dir), str(QLIB_ROOT)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def run_step(name: str, command: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x", encoding="utf-8") as log:
        header = f"[{datetime.now():%F %T}] START {name}\ncmd={command!r}\ncwd={cwd}\n\n"
        print(header, end="", flush=True)
        log.write(header)
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{name}] {line}", end="", flush=True)
            log.write(line)
        return_code = process.wait()
        footer = f"\n[{datetime.now():%F %T}] END {name} status={return_code}\n"
        print(footer, end="", flush=True)
        log.write(footer)
    if return_code != 0:
        raise RuntimeError(f"step failed: {name}; see {log_path}")


def run_parallel_steps(
    steps: list[tuple[str, list[str], Path]],
    cwd: Path,
    log_dir: Path,
    env: dict[str, str],
    workers: int,
) -> None:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_step, name, command, cwd, log_dir / f"{order:02d}_{name}.log", env): name
            for order, (name, command, _) in enumerate(steps, start=1)
        }
        for future in as_completed(futures):
            future.result()


def validate_stage1_summary(summary_path: Path, expected_key: str, forbidden_key: str, run_dir: Path) -> None:
    data = read_kv_summary(summary_path)
    runs = parse_seed_map(summary_path, expected_key)
    if set(runs) != set(SEEDS):
        raise RuntimeError(f"{summary_path} has seeds {sorted(runs)}, expected {list(SEEDS)}")
    if data.get(forbidden_key):
        raise RuntimeError(f"{summary_path} unexpectedly contains {forbidden_key}")
    for seed, row in runs.items():
        cache = assert_under_run(Path(str(row["cache_path"])), run_dir, f"seed {seed} cache")
        result = assert_under_run(Path(str(row["result_path"])), run_dir, f"seed {seed} result")
        if not cache.exists() or not result.exists():
            raise FileNotFoundError(f"missing fresh Stage1 output for seed {seed}: {cache} / {result}")


def assert_no_prelock_test_artifacts(stage1_dir: Path, stage2_dir: Path) -> None:
    offenders = list((stage1_dir / "tra_cache").glob("*test*.pkl"))
    offenders += list(stage1_dir.glob("rolling_result_*test*.txt"))
    offenders += list(stage1_dir.glob("tra_global_*test*.txt"))
    offenders += list((stage2_dir / "fusion_cache").glob("*test*.pkl"))
    offenders += list((stage2_dir / "tmp").glob("*final*"))
    if offenders:
        raise RuntimeError(f"test artifacts exist before validation lock: {offenders[:20]}")


def freeze_pretest_lock(locked_manifest: Path, immutable_path: Path) -> str:
    payload = json.loads(locked_manifest.read_text(encoding="utf-8"))
    if payload.get("final_test_summary_path") is not None:
        raise RuntimeError("locked manifest already contains a final result before test")
    shutil.copy2(locked_manifest, immutable_path)
    immutable_path.chmod(0o444)
    digest = sha256_file(immutable_path)
    immutable_path.with_suffix(".sha256").write_text(f"{digest}  {immutable_path.name}\n", encoding="ascii")
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict one-key Stage1+Stage2 training from zero with validation-only selection and one final test."
    )
    parser.add_argument("--run-id", default=f"strict_fusion_from_zero_{now_tag()}")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--provider-uri", type=Path, default=DEFAULT_PROVIDER_URI)
    parser.add_argument("--gpu-slots", default="auto", help="auto uses all visible GPUs; otherwise comma-separated indices")
    parser.add_argument("--jobs-per-gpu", type=int, default=0, help="0 selects from free VRAM and host RAM")
    parser.add_argument("--stage2-workers", type=int, default=0, help="0 selects from CPU and host RAM")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--target-test-ann", type=float, default=TARGET_TEST_ANN)
    parser.add_argument("--skip-execute", action="store_true", help="create and validate the isolated plan without training")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size != BATCH_SIZE:
        raise ValueError(
            f"formal cross-machine reproduction requires batch_size={BATCH_SIZE}; "
            "changing it creates a different experiment"
        )

    run_root = args.run_root.expanduser().resolve()
    run_dir = run_root / args.run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists; refusing artifact reuse: {run_dir}")
    provider_uri = args.provider_uri.expanduser().resolve()
    data_link = ensure_training_data_link(run_root, provider_uri)

    gpus = detect_gpus()
    selected_indices = visible_gpu_indices(gpus, args.gpu_slots)
    resource_plan = build_resource_plan(
        gpus,
        selected_indices,
        len(SEEDS),
        args.jobs_per_gpu,
        args.stage2_workers,
    )

    stage1_dir = run_dir / "stage1"
    stage2_dir = run_dir / "stage2"
    log_dir = run_dir / "logs"
    copy_source_tree(SOURCE_STAGE1, stage1_dir)
    copy_source_tree(SOURCE_STAGE2, stage2_dir)
    assert_clean_source_snapshot(stage1_dir)
    assert_clean_source_snapshot(stage2_dir)
    for directory in (
        stage1_dir / "tmp",
        stage1_dir / "tra_cache",
        stage1_dir / "mlruns",
        stage2_dir / "tmp",
        stage2_dir / "fusion_cache",
        stage2_dir / "logs",
        stage2_dir / "mlruns",
        log_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    valid_suffix = f"{args.run_id}_b{args.batch_size}_valid"
    test_suffix = f"{args.run_id}_b{args.batch_size}_test"
    valid_summary = stage1_dir / f"tra_global_dual_seed_pattern_summary_{valid_suffix}.txt"
    test_summary = stage1_dir / f"tra_global_dual_seed_pattern_summary_{test_suffix}.txt"
    selector_prefixes = [stage2_dir / "tmp" / f"validation_trial_{index:02d}" for index in range(len(TRIAL_NAMES))]
    selector_grids = [prefix.with_name(f"{prefix.name}_grid.csv") for prefix in selector_prefixes]
    stress_prefix = stage2_dir / "tmp" / "validation_stress"
    candidate_summary = stress_prefix.with_name(f"{stress_prefix.name}_candidate_summary.csv")
    lock_prefix = stage2_dir / "tmp" / "locked_candidate"
    locked_manifest = lock_prefix.with_name(f"{lock_prefix.name}_locked_manifest.json")
    immutable_lock = stage2_dir / "tmp" / "locked_candidate_pretest_immutable.json"
    final_prefix = stage2_dir / "tmp" / "final_test_once"
    final_summary = final_prefix.with_name(f"{final_prefix.name}_summary.json")

    common_stage1 = [
        sys.executable,
        "tune_tra_alpha360_global.py",
        "--dual-seed-study",
        "--dual-seed-base-only",
        "--model-family",
        "tra",
        "--seeds",
        ",".join(str(seed) for seed in SEEDS),
        "--batch-size",
        str(args.batch_size),
        "--gpu-slots",
        resource_plan.gpu_slots,
        "--max-concurrent-seeds",
        str(resource_plan.jobs_per_gpu),
        "--min-available-ram-gb",
        str(resource_plan.min_available_ram_gib),
        "--max-extra-swap-gb",
        "48",
        "--resource-poll-seconds",
        "2",
    ]
    validation_command = common_stage1 + [
        "--dual-seed-segment",
        "valid",
        "--result-suffix",
        valid_suffix,
    ]
    test_command = common_stage1 + [
        "--dual-seed-segment",
        "test",
        "--result-suffix",
        test_suffix,
    ]

    selector_commands: list[list[str]] = []
    for prefix, trial_name in zip(selector_prefixes, TRIAL_NAMES):
        selector_commands.append(
            [
                sys.executable,
                "h200_quarter_stability_selector.py",
                "--valid-summary-path",
                str(valid_summary),
                "--output-prefix",
                str(prefix),
                "--combo",
                ",".join(str(seed) for seed in SEEDS),
                "--signal-profile",
                "cash-quality-quick",
                "--strategy-profiles",
                "stable-cash-quick",
                "--segment-levels",
                "year,half,quarter",
                "--trial-name",
                trial_name,
                "--force-recompute",
            ]
        )

    stress_command = [
        sys.executable,
        "h200_validation_stress_audit.py",
        "--valid-summary-path",
        str(valid_summary),
        "--output-prefix",
        str(stress_prefix),
        "--signal-profile",
        "cash-quality-quick",
        "--strategy-profiles",
        "stable-cash-quick",
        "--segment-levels",
        "year,half,quarter",
        "--max-candidates",
        str(len(TRIAL_NAMES)),
        "--prefilter-full-ann-min",
        str(STRESS_GATE["full_ann_min"]),
        "--prefilter-year-ann-min",
        str(STRESS_GATE["year_ann_min_min"]),
        "--gate-full-ann-min",
        str(STRESS_GATE["full_ann_min"]),
        "--gate-year-ann-min",
        str(STRESS_GATE["year_ann_min_min"]),
        "--gate-half-ann-min",
        str(STRESS_GATE["half_ann_min_min"]),
        "--gate-quarter-ann-min",
        str(STRESS_GATE["quarter_ann_min_min"]),
        "--gate-quarter-positive-ratio-min",
        str(STRESS_GATE["quarter_positive_ratio_min"]),
        "--gate-quarter-rankic-min",
        str(STRESS_GATE["quarter_rankic_min"]),
        "--gate-quarter-rankic-positive-ratio-min",
        str(STRESS_GATE["quarter_rankic_positive_ratio_min"]),
        "--force-recompute",
    ]
    for grid in selector_grids:
        stress_command.extend(["--cv-grid-path", str(grid)])

    lock_command = [
        sys.executable,
        "lock_h200_stress_gate_candidate.py",
        "--candidate-summary-path",
        str(candidate_summary),
        "--valid-summary-path",
        str(valid_summary),
        "--output-prefix",
        str(lock_prefix),
        "--signal-profile",
        "cash-quality-quick",
        "--target-test-ann",
        str(args.target_test_ann),
        "--require-fused-stage2",
    ]
    final_command = [
        sys.executable,
        "h200_strict_final_test_once.py",
        "--locked-manifest",
        str(locked_manifest),
        "--test-summary-path",
        str(test_summary),
        "--output-prefix",
        str(final_prefix),
        "--result-suffix",
        f"{args.run_id}_final_once",
    ]

    preregistration = {
        "experiment_id": args.run_id,
        "created_at_utc": utc_now(),
        "objective": {"metric": "test_with_cost_ann_return", "target_ge": args.target_test_ann},
        "protocol": {
            "from_zero": True,
            "selection_scope": "validation_only_before_immutable_lock",
            "test_usage_policy": "test training and fused final evaluation begin only after immutable lock",
            "train": ["2012-01-01", "2019-12-31"],
            "valid": ["2020-01-01", "2022-12-31"],
            "test": ["2023-01-01", "2026-04-01"],
            "account": 150000,
            "topk": 5,
            "universe": "csi300",
        },
        "stage1": {"seeds": SEEDS, "batch_size": args.batch_size, "rolling_tasks_per_seed": ROLLING_TASKS_PER_SEED},
        "stage2": {
            "signal_profile": "cash-quality-quick",
            "strategy_profile": "stable-cash-quick",
            "trial_names": TRIAL_NAMES,
            "stress_gate": STRESS_GATE,
            "sort": ["stress_score", "year_ann_min", "full_ann", "half_ann_min", "quarter_ann_min"],
        },
    }
    write_json(run_dir / "preregistered_protocol.json", preregistration)

    manifest = {
        "run_id": args.run_id,
        "created_at_utc": utc_now(),
        "policy": "fresh isolated run; source code only; no existing model, cache, grid, lock, or result artifact",
        "run_dir": run_dir,
        "provider_uri": provider_uri,
        "training_data_link": data_link,
        "environment": environment_snapshot(gpus, resource_plan),
        "source_snapshot_sha256": {
            "stage1": source_fingerprint(stage1_dir),
            "stage2": source_fingerprint(stage2_dir),
            "qlib_custom_strategy": sha256_file(QLIB_ROOT / "qlib/contrib/strategy/custom_signal_strategy.py"),
        },
        "provider_metadata_fingerprint": tree_metadata_fingerprint(provider_uri),
        "feature_file_sha256": {
            "csi300_daily_fundamental_features_rankpct_qlib.parquet": sha256_file(
                QLIB_ROOT
                / "training_data/processed/training/csi300_daily_fundamental_features_rankpct_qlib.parquet"
            )
        },
        "planned_outputs": {
            "valid_summary": valid_summary,
            "test_summary": test_summary,
            "candidate_summary": candidate_summary,
            "locked_manifest": locked_manifest,
            "immutable_pretest_lock": immutable_lock,
            "final_summary": final_summary,
        },
        "commands": {
            "stage1_validation_from_zero": validation_command,
            "stage2_validation_shards": selector_commands,
            "stage2_validation_stress": stress_command,
            "validation_lock": lock_command,
            "stage1_test_from_zero_after_lock": test_command,
            "final_test_once": final_command,
        },
    }
    write_json(run_dir / "from_zero_manifest.json", manifest)
    for path in (valid_summary, test_summary, candidate_summary, locked_manifest, immutable_lock, final_summary):
        assert_under_run(path, run_dir, "planned output")

    print(json.dumps(asdict(resource_plan), indent=2), flush=True)
    print(f"fresh strict run ready: {run_dir}", flush=True)
    if args.skip_execute:
        print(f"plan only; manifest={run_dir / 'from_zero_manifest.json'}", flush=True)
        return 0

    stage1_env = command_env(stage1_dir, provider_uri, resource_plan.threads_per_worker)
    stage2_env = command_env(stage2_dir, provider_uri, resource_plan.threads_per_worker)

    run_step(
        "01_stage1_validation_from_zero",
        validation_command,
        stage1_dir,
        log_dir / "01_stage1_validation_from_zero.log",
        stage1_env,
    )
    validate_stage1_summary(valid_summary, "valid_base_runs", "test_base_runs", run_dir)
    assert_no_prelock_test_artifacts(stage1_dir, stage2_dir)

    shard_steps = [
        (f"02_stage2_validation_{index:02d}", command, grid)
        for index, (command, grid) in enumerate(zip(selector_commands, selector_grids))
    ]
    run_parallel_steps(
        shard_steps,
        stage2_dir,
        log_dir / "stage2_validation_shards",
        stage2_env,
        resource_plan.stage2_workers,
    )
    missing_grids = [grid for grid in selector_grids if not grid.exists()]
    if missing_grids:
        raise FileNotFoundError(f"missing Stage2 validation grids: {missing_grids}")
    assert_no_prelock_test_artifacts(stage1_dir, stage2_dir)

    run_step(
        "03_stage2_validation_stress",
        stress_command,
        stage2_dir,
        log_dir / "03_stage2_validation_stress.log",
        stage2_env,
    )
    run_step(
        "04_validation_lock",
        lock_command,
        stage2_dir,
        log_dir / "04_validation_lock.log",
        stage2_env,
    )
    assert_no_prelock_test_artifacts(stage1_dir, stage2_dir)
    immutable_lock_sha256 = freeze_pretest_lock(locked_manifest, immutable_lock)
    write_json(
        run_dir / "pretest_checkpoint.json",
        {
            "created_at_utc": utc_now(),
            "immutable_lock": immutable_lock,
            "immutable_lock_sha256": immutable_lock_sha256,
            "assertion": "no test cache or final output existed when this checkpoint was written",
        },
    )

    run_step(
        "05_stage1_test_from_zero_after_lock",
        test_command,
        stage1_dir,
        log_dir / "05_stage1_test_from_zero_after_lock.log",
        stage1_env,
    )
    validate_stage1_summary(test_summary, "test_base_runs", "valid_base_runs", run_dir)
    run_step(
        "06_final_test_once",
        final_command,
        stage2_dir,
        log_dir / "06_final_test_once.log",
        stage2_env,
    )
    if not final_summary.exists():
        raise FileNotFoundError(f"final summary was not generated: {final_summary}")

    final_payload = json.loads(final_summary.read_text(encoding="utf-8"))
    final_ann = float(final_payload.get("test_with_cost_ann_return", float("-inf")))
    target_ann = float(args.target_test_ann)
    completion = {
        "completed_at_utc": utc_now(),
        "run_id": args.run_id,
        "immutable_pretest_lock_sha256": immutable_lock_sha256,
        "selected_trial": final_payload.get("strategy_trial"),
        "selected_signal": final_payload.get("signal_name"),
        "test_with_cost_ann_return": final_ann,
        "test_with_cost_ir": final_payload.get("test_with_cost_ir"),
        "test_with_cost_mdd": final_payload.get("test_with_cost_mdd"),
        "target_test_ann": target_ann,
        "target_met_or_exceeded": bool(final_ann >= target_ann - 1e-12),
        "target_exceeded": bool(final_ann > target_ann + 1e-12),
        "final_summary": final_summary,
    }
    write_json(run_dir / "completion_audit.json", completion)
    print(json.dumps(jsonable(completion), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
