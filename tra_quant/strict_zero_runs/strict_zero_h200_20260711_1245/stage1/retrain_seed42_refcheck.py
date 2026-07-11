from __future__ import annotations

import argparse
import copy
import importlib.util
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent.parent
DEFAULT_PROVIDER_URI = ROOT_DIR / "training_data" / "cn_data_latest"
DEFAULT_SUFFIX = "seed42_refcheck_retrain_20260620_1"


def load_tuner():
    path = BASE_DIR / "tune_tra_alpha360_global.py"
    spec = importlib.util.spec_from_file_location("tune_tra_alpha360_global_seed42_retrain", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load tuner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_seed42_refcheck(*, suffix: str, gpu_slots: str, provider_uri: Path) -> dict[str, dict]:
    os.environ["QLIB_PROVIDER_URI"] = str(provider_uri.resolve())
    tuner = load_tuner()
    model_trial = copy.deepcopy(tuner.MODEL_TRIALS[0])
    common = dict(
        model_family="tra",
        model_trial=model_trial,
        seed=42,
        gpu_slots=gpu_slots,
        deterministic_runtime=True,
        window_key=tuner.WINDOW_KEY,
        n_drop=1,
        strategy_class=None,
        strategy_module_path=None,
        strategy_kwargs_extra=None,
        dataset_kwargs_extra={"batch_size": 16384},
        rolling_task_limit=None,
        parallel_mode="fork_readonly",
    )
    valid = tuner._run_family_trial(
        result_suffix=f"{suffix}_valid",
        risk_degree=0.70,
        eval_segment="valid",
        backtest_segment="valid",
        **common,
    )
    print("seed42_valid_result=", valid)
    test = tuner._run_family_trial(
        result_suffix=f"{suffix}_test",
        risk_degree=0.85,
        eval_segment="test",
        backtest_segment="test",
        **common,
    )
    print("seed42_test_result=", test)
    return {"valid": valid, "test": test}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain the authoritative seed42 TRA stage1 cache with the same "
            "global/refcheck configuration used by the frozen baseline."
        )
    )
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--gpu-slots", default="0,1")
    parser.add_argument("--provider-uri", default=str(DEFAULT_PROVIDER_URI))
    args = parser.parse_args()

    provider_uri = Path(args.provider_uri).expanduser().resolve()
    if not provider_uri.exists():
        raise FileNotFoundError(f"QLib provider_uri does not exist: {provider_uri}")

    run_seed42_refcheck(
        suffix=args.suffix,
        gpu_slots=args.gpu_slots,
        provider_uri=provider_uri,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
