# GPU Lunch Break Tasks (~1-2 hours)

## For GPU Machine - Queue During Lunch

### Task 1: Neural Stacking (BEST - GPU ACCELERATED) 🔥
```bash
cd ~/fiicode-2026-ai-bank-telemarketing-prediction
git pull
source venv/bin/activate
python gpu_train_stacking.py
```

**What it does:** Trains **neural meta-model on GPU** using ALL OOF predictions. Learns optimal blending automatically with non-linear combinations.
**Expected time:** ~30-45 minutes (GPU accelerated)
**Expected gain:** +50-150 points if models are diverse
**GPU benefit:** 10x faster than CPU, can use deeper networks

---

### Task 2: Deep TabNet Ensemble (GPU ACCELERATED) 🔥
```bash
python gpu_train_tabnet.py --config experiments/exp051_tabnet_ensemble.yaml
```

**What it does:** Trains 3 diverse TabNet architectures simultaneously and blends them
**Expected time:** ~45 minutes (GPU parallel training)
**Expected gain:** +20-60 points from TabNet diversity
**GPU benefit:** Can train multiple models in parallel

---

### Task 3: Attention Network Sweep (GPU ONLY) 🔥
```bash
python gpu_train.py --config experiments/exp052_attention_sweep.yaml
```

**What it does:** Trains 5 attention networks with different architectures/seeds
**Expected time:** ~30 minutes (GPU parallel)
**Expected gain:** +30-80 points from diversity
**GPU benefit:** Trains all variants in parallel, impossible on CPU

---

## Priority Order (if time-limited):

1. **Stacking** (most promising, automatic optimization)
2. **Blend optimization** (fast, might squeeze last points)
3. **Adversarial** (lowest priority, unlikely to help)

---

## Commands to Queue All 3 (GPU-accelerated):

```bash
cd ~/fiicode-2026-ai-bank-telemarketing-prediction && \
git pull && \
source venv/bin/activate && \
python gpu_train_stacking.py && \
python gpu_train_tabnet.py --config experiments/exp051_tabnet_ensemble.yaml && \
python gpu_train.py --config experiments/exp052_attention_sweep.yaml && \
git add outputs/ && \
git commit -m "GPU lunch tasks: neural stacking + TabNet ensemble + attention sweep" && \
git push
```

**Total time:** ~1.5-2 hours (all GPU-accelerated)
**Perfect for lunch break** ☕

## Why These Are GPU Tasks:

- ✅ **Neural stacking**: Trains deep neural network (10x faster on GPU)
- ✅ **TabNet ensemble**: GPU-accelerated attention mechanism
- ✅ **Attention sweep**: Pure GPU neural networks
- ❌ **NOT included**: CatBoost/XGBoost (CPU-only, no GPU benefit)
