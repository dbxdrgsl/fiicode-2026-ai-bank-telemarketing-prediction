#!/bin/bash
# Overnight batch runner for CPU experiments
# Usage: bash run_overnight_cpu.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/admin/.venvs/fiicode/bin/python}"

echo "========================================"
echo "OVERNIGHT CPU BATCH - $(date -u)"
echo "Root:   $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "========================================"

"$PYTHON_BIN" -c "import sys; print('Python OK:', sys.version.split()[0])"

echo ""
echo "[1/3] exp033_target_encoding"
"$PYTHON_BIN" -m src.train --config experiments/exp033_target_encoding.yaml
echo "exp033 complete at $(date -u)"

echo ""
echo "[2/3] exp034_lightgbm_target_enc"
"$PYTHON_BIN" -m src.train --config experiments/exp034_lightgbm_target_enc.yaml
echo "exp034 complete at $(date -u)"

echo ""
echo "[3/3] exp035_xgboost_target_enc"
"$PYTHON_BIN" -m src.train --config experiments/exp035_xgboost_target_enc.yaml
echo "exp035 complete at $(date -u)"

echo ""
echo "========================================"
echo "ALL CPU EXPERIMENTS COMPLETE - $(date -u)"
echo "========================================"
