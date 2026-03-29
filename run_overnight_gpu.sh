#!/bin/bash
# Overnight batch runner for GPU experiments
# Usage: bash run_overnight_gpu.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/admin/.venvs/fiicode/bin/python}"

echo "========================================"
echo "OVERNIGHT GPU BATCH - $(date -u)"
echo "Root:   $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "========================================"

"$PYTHON_BIN" -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "[1/4] exp036_gpu_attention_target_enc"
"$PYTHON_BIN" gpu_train.py --config experiments/exp036_gpu_attention_target_enc.yaml
echo "exp036 complete at $(date -u)"

echo ""
echo "[2/4] exp037_gpu_fttransformer_deep"
"$PYTHON_BIN" gpu_train.py --config experiments/exp037_gpu_fttransformer_deep.yaml
echo "exp037 complete at $(date -u)"

echo ""
echo "[3/4] exp038_gpu_tabresnet_target_enc"
"$PYTHON_BIN" gpu_train.py --config experiments/exp038_gpu_tabresnet_target_enc.yaml
echo "exp038 complete at $(date -u)"

echo ""
echo "[4/4] exp039_gpu_mega_blend"
"$PYTHON_BIN" gpu_train.py --config experiments/exp039_gpu_mega_blend.yaml
echo "exp039 complete at $(date -u)"

echo ""
echo "========================================"
echo "ALL GPU EXPERIMENTS COMPLETE - $(date -u)"
echo "========================================"
