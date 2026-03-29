# GPU Machine Instructions - Orthogonal Experiments

## Prerequisites
```bash
# Sync repo
cd ~/fiicode-2026-ai-bank-telemarketing-prediction  # or wherever you synced
git pull

# Activate venv
source venv/bin/activate  # or your venv path

# Install required packages
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install pytorch-tabnet
```

## Experiment 1: GNN (Graph Neural Network)

**What it does:** Builds customer similarity graph to capture relationships CatBoost misses.

**Run command:**
```bash
# Wait for the gnn-customer-graph agent to finish creating gpu_train_gnn.py
# Then run:
python gpu_train_gnn.py --config experiments/exp044_gnn_graphsage.yaml

# If you want to try GAT variant too:
python gpu_train_gnn.py --config experiments/exp045_gnn_gat.yaml
```

**Expected runtime:** ~2 hours per experiment

**What gets created:**
- `outputs/submissions/exp044_gnn_graphsage/submission.csv`
- `outputs/oof/exp044_gnn_graphsage/oof_predictions.csv`
- `outputs/logs/exp044_gnn_graphsage/best_run_summary.json`

**After completion:**
```bash
git add outputs/logs/exp044* outputs/submissions/exp044* outputs/oof/exp044* outputs/models/exp044*
git commit -m "exp044 GNN GraphSAGE results from GPU"
git push
```

---

## Experiment 2: TabNet (Instance-wise Attention)

**What it does:** Learns which features matter for EACH prediction dynamically via attention.

**Run command:**
```bash
# Wait for the tabnet-attention agent to finish creating gpu_train_tabnet.py
# Then run:
python gpu_train_tabnet.py --config experiments/exp046_tabnet.yaml

# If you want deeper version:
python gpu_train_tabnet.py --config experiments/exp047_tabnet_deep.yaml
```

**Expected runtime:** ~1.5 hours per experiment

**What gets created:**
- `outputs/submissions/exp046_tabnet/submission.csv`
- `outputs/oof/exp046_tabnet/oof_predictions.csv`
- `outputs/logs/exp046_tabnet/best_run_summary.json`
- `outputs/logs/exp046_tabnet/feature_importance_comparison.csv` (TabNet vs CatBoost)

**After completion:**
```bash
git add outputs/logs/exp046* outputs/submissions/exp046* outputs/oof/exp046* outputs/models/exp046*
git commit -m "exp046 TabNet attention results from GPU"
git push
```

---

## Quick Check GPU Usage

```bash
# Verify GPU is active
nvidia-smi

# Monitor during training
watch -n 1 nvidia-smi
```

---

## Troubleshooting

**If torch-geometric install fails:**
```bash
# Try CPU-only version for graph construction (slower but works)
pip install torch-geometric

# Then move the graph to GPU machine separately
```

**If out of memory:**
```bash
# Reduce batch size in the YAML config
# Or run with smaller hidden dims
```

**Check progress:**
```bash
# Watch the log files
tail -f outputs/logs/exp044_gnn_graphsage/optuna_trials.csv
tail -f outputs/logs/exp046_tabnet/optuna_trials.csv
```

---

## Expected Results

- **GNN (exp044/045):** Should get CV ~0.932-0.935
- **TabNet (exp046/047):** Should get CV ~0.933-0.936

The key is whether they're **orthogonal** to exp012 (CatBoost), not just higher CV.
We'll blend them to see if they capture different patterns.

---

## Queue Multiple Experiments

If you can queue jobs:
```bash
# Run all 4 experiments sequentially
python gpu_train_gnn.py --config experiments/exp044_gnn_graphsage.yaml && \
python gpu_train_gnn.py --config experiments/exp045_gnn_gat.yaml && \
python gpu_train_tabnet.py --config experiments/exp046_tabnet.yaml && \
python gpu_train_tabnet.py --config experiments/exp047_tabnet_deep.yaml && \
git add outputs/ experiments/ && \
git commit -m "All GPU orthogonal experiments complete" && \
git push
```

Total runtime: ~7 hours for all 4 experiments
