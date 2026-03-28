#!/bin/bash
# Overnight batch runner for CPU experiments
# Run from: /mnt/c/Users/dbxdr_iytiz92/Dropbox/fiicode
# Usage: bash run_overnight_cpu.sh

set -e

echo "========================================"
echo "OVERNIGHT CPU BATCH - $(date)"
echo "========================================"

source .venv/bin/activate

# exp033: CatBoost + target encoding (~2 hours)
echo ""
echo "[1/3] Starting exp033_target_encoding..."
python -m src.train --config experiments/exp033_target_encoding.yaml
echo "exp033 complete at $(date)"

# exp034: LightGBM + target encoding (~1.5 hours)
echo ""
echo "[2/3] Starting exp034_lightgbm_target_enc..."
python -m src.train --config experiments/exp034_lightgbm_target_enc.yaml
echo "exp034 complete at $(date)"

# exp035: XGBoost + target encoding (~1.5 hours)
echo ""
echo "[3/3] Starting exp035_xgboost_target_enc..."
python -m src.train --config experiments/exp035_xgboost_target_enc.yaml
echo "exp035 complete at $(date)"

echo ""
echo "========================================"
echo "ALL CPU EXPERIMENTS COMPLETE - $(date)"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Check outputs/logs/ for results"
echo "2. Run: python -m src.blend --name exp040_overnight_blend --summaries outputs/logs/exp033_target_encoding/best_run_summary.json outputs/logs/exp034_lightgbm_target_enc/best_run_summary.json outputs/logs/exp035_xgboost_target_enc/best_run_summary.json"
