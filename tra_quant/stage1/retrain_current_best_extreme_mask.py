from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from extreme_event_mask import DEFAULT_MASK_END, DEFAULT_MASK_START


BASE_DIR = Path(__file__).resolve().parent

STAGE1_SCRIPT = BASE_DIR / "tune_tra_alpha360_strict.py"
STAGE2_SCRIPT = BASE_DIR / "tune_alpha360_tra_fundamental_xgb_weight_refine.py"


def _run_python(script: Path, extra_args: list[str] | None = None):
    cmd = [sys.executable, str(script)]
    if extra_args:
        cmd.extend(extra_args)
    subprocess.run(cmd, cwd=BASE_DIR, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Retrain the current best baseline while masking train labels in the 2015-06 to 2016-01 extreme-event window."
    )
    parser.add_argument("--window-key", default="w1", choices=["w1", "w2", "w3"])
    parser.add_argument("--result-suffix", default="sw0", help="suffix appended to all generated outputs")
    parser.add_argument("--train-label-mask-start", default=DEFAULT_MASK_START)
    parser.add_argument("--train-label-mask-end", default=DEFAULT_MASK_END)
    parser.add_argument("--force-stage1-final", action="store_true", help="rerun stage1 final test even if summary exists")
    args = parser.parse_args()

    common_args = [
        "--window-key",
        args.window_key,
        "--result-suffix",
        args.result_suffix,
        "--train-label-mask-start",
        args.train_label_mask_start,
        "--train-label-mask-end",
        args.train_label_mask_end,
    ]

    stage1_args = list(common_args)
    if args.force_stage1_final:
        stage1_args.append("--force-final")

    _run_python(STAGE1_SCRIPT, stage1_args)
    _run_python(STAGE2_SCRIPT, common_args)


if __name__ == "__main__":
    main()