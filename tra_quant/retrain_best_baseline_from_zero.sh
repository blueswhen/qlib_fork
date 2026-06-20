#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python retrain_best_baseline_from_zero.py "$@"
