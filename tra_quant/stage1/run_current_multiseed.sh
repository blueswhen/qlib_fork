#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QLIB_PROVIDER_URI="${QLIB_PROVIDER_URI:-$SCRIPT_DIR/../../training_data/cn_data_latest}"

cd "$SCRIPT_DIR"
python run_current_best_multiseed_ensemble.py --materialize-test-summary "$@"
