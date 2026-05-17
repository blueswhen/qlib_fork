#!/usr/bin/env bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/blueswhen/miniconda3/envs/py310/bin/python}"
STEP1_PID="${STEP1_PID:-}"
STEP1_SUFFIX="${STEP1_SUFFIX:-tra18_residual_step1_20260506_1}"
STEP2_SUFFIX="${STEP2_SUFFIX:-tra18_residual_full_20260506_1}"

STEP1_VALIDATION_PATH="$BASE_DIR/tra_global_validation_best_${STEP1_SUFFIX}.txt"
STEP1_FINAL_PATH="$BASE_DIR/tra_global_final_test_${STEP1_SUFFIX}.txt"
STEP2_LOG_PATH="$BASE_DIR/${STEP2_SUFFIX}.log"

echo "[queue] base_dir=$BASE_DIR"
echo "[queue] step1_suffix=$STEP1_SUFFIX"
echo "[queue] step2_suffix=$STEP2_SUFFIX"

if [[ -n "$STEP1_PID" ]] && kill -0 "$STEP1_PID" 2>/dev/null; then
    echo "[queue] waiting for step1 pid $STEP1_PID to exit"
    tail --pid="$STEP1_PID" -f /dev/null
    echo "[queue] step1 pid $STEP1_PID exited"
fi

if [[ ! -f "$STEP1_VALIDATION_PATH" ]]; then
    echo "[queue] missing step1 validation summary: $STEP1_VALIDATION_PATH" >&2
    exit 1
fi

if [[ ! -f "$STEP1_FINAL_PATH" ]]; then
    echo "[queue] missing step1 final test summary: $STEP1_FINAL_PATH" >&2
    exit 1
fi

echo "[queue] launching residual stage2"
cd "$BASE_DIR"
PYTHONUNBUFFERED=1 "$PYTHON_BIN" tune_alpha360_tra_residual_fundamental_xgb.py \
    --window-key w1 \
    --result-suffix "$STEP2_SUFFIX" \
    --tra-validation-best-path "$STEP1_VALIDATION_PATH" \
    --tra-final-test-path "$STEP1_FINAL_PATH" \
    2>&1 | tee "$STEP2_LOG_PATH"

echo "[queue] pipeline finished"
echo "[queue] compare file: $BASE_DIR/alpha360_tra_residual_fundamental_xgb_final_test_${STEP2_SUFFIX}.txt"