#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$SCRIPT_DIR"
exec "$PYTHON_BIN" train_strict_fusion_from_zero.py "$@"
