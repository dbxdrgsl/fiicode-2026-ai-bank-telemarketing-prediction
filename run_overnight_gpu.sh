#!/bin/bash
# Overnight batch runner for GPU experiments
# Run from your GPU machine in the synced folder
# Usage: bash run_overnight_gpu.sh

set -e

echo "========================================"
echo "OVERNIGHT GPU BATCH - $(date)"
echo "========================================"

source .venv/bin/activate

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# exp036: Attention + target encoding (~30 min)
echo ""
echo "[1/4] Starting exp036_gpu_attention_target_enc..."
python gpu_train.py --config experiments/exp036_gpu_attention_target_enc.yaml
echo "exp036 complete at $(date)"

# exp037: Deep FT-Transformer (~45 min)
echo ""
echo "[2/4] Starting exp037_gpu_fttransformer_deep..."
python gpu_train.py --config experiments/exp037_gpu_fttransformer_deep.yaml
echo "exp037 complete at $(date)"

# exp038: TabResNet + target encoding (~30 min)
echo ""
echo "[3/4] Starting exp038_gpu_tabresnet_target_enc..."
python gpu_train.py --config experiments/exp038_gpu_tabresnet_target_enc.yaml
echo "exp038 complete at $(date)"

# exp039: Mega blend of all architectures (~1 hour)
echo ""
echo "[4/4] Starting exp039_gpu_mega_blend..."
python gpu_train.py --config experiments/exp039_gpu_mega_blend.yaml
echo "exp039 complete at $(date)"

echo ""
echo "========================================"
echo "ALL GPU EXPERIMENTS COMPLETE - $(date)"
echo "========================================"
echo ""
echo "Results synced via Dropbox to local machine."
echo "Check outputs/oof/ and outputs/submissions/ for predictions."
