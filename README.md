# FiiCode 2026 Workspace

Lean local experiment repo for Kaggle competition `fiicode-2026-ai-competition`.
The repo now supports:

- CatBoost as the main tabular anchor
- LightGBM and XGBoost as diversity models
- OOF-based blending
- reproducible YAML experiments
- Kaggle CLI validation and submission journaling

## Competition

The task is binary classification: predict the probability that a client of a
Portuguese bank subscribes to a term deposit after a telemarketing campaign.

- Target: `Subscribed`
- Metric: ROC-AUC
- Submission format:

```csv
id,Subscribed
534,0.9
535,0.1
536,0.5
```

Constraints that matter:

- no external data
- no external pretrained models
- no AutoML
- max 5 submissions per day
- reproducibility may be requested by organizers

## Repo Shape

```text
fiicode/
|-- AGENTS.md
|-- CLAUDE.md
|-- README.md
|-- .claude/
|-- data/raw/{train.csv,test.csv}
|-- notebooks/improved.ipynb
|-- src/
|   |-- config.py
|   |-- features.py
|   |-- modeling.py
|   |-- tracking.py
|   |-- train.py
|   |-- infer.py
|   |-- cv.py
|   |-- blend.py
|   `-- submit.py
|-- experiments/
|   |-- exp001_baseline.yaml
|   |-- exp002_no_duration.yaml
|   |-- exp003_blend_features.yaml
|   |-- exp004_blend_no_duration.yaml
|   |-- exp005_blend_no_class_weight.yaml
|   |-- exp006_catboost_longsearch.yaml
|   |-- exp007_catboost_seedbag.yaml
|   |-- exp008_lightgbm_onehot.yaml
|   `-- exp009_xgboost_onehot.yaml
|-- outputs/{models,oof,submissions,logs}
|-- kernel-metadata.json.example
`-- requirements.txt
```

## Workflow

Activate the WSL venv:

```bash
cd /mnt/c/Users/dbxdr_iytiz92/Dropbox/fiicode
source .venv/bin/activate
```

Train an experiment:

```bash
python -m src.train --config experiments/exp003_blend_features.yaml
```

Run ablations:

```bash
python -m src.cv --config experiments/exp003_blend_features.yaml
python -m src.cv --config experiments/exp003_blend_features.yaml --quick
```

Rebuild a submission from saved params:

```bash
python -m src.infer --config experiments/exp003_blend_features.yaml
```

Audit whether the local data is a public subset of UCI Bank Marketing:

```bash
python -m src.audit_public_overlap
```

Validate or submit:

```bash
python -m src.submit --submission outputs/submissions/exp003_blend_features/submission.csv --message "exp003 blend"
python -m src.submit --submission outputs/submissions/exp003_blend_features/submission.csv --message "exp003 blend" --run-kaggle-cli
```

Build a blend from saved experiments:

```bash
python -m src.blend --name exp010_blend \
  --summaries \
  outputs/logs/exp003_blend_features/best_run_summary.json \
  outputs/logs/exp008_lightgbm_onehot/best_run_summary.json \
  outputs/logs/exp009_xgboost_onehot/best_run_summary.json
```

Train custom GPU neural trio (Attention + FT-style Transformer + TabResNet):

```bash
python gpu_train.py --config experiments/exp025_gpu_nn_trio.yaml
```

Validate GPU blend submission candidate:

```bash
python -m src.submit --submission outputs/submissions/exp025_gpu_nn_blend/submission.csv --message "exp025 gpu nn blend"
```

Blend GPU neural branch with current CatBoost incumbent:

```bash
python -m src.blend --name exp026_exp012_plus_gpu \
  --summaries \
  outputs/logs/exp012_blend_bucket_features_fixed/best_run_summary.json \
  outputs/logs/exp025_gpu_nn_blend/best_run_summary.json
```

## Experiment Ladder

- `exp003_blend_features`: current best CatBoost-style blend features
- `exp004_blend_no_duration`: check whether duration-derived features are hurting
- `exp005_blend_no_class_weight`: isolate class-weight impact
- `exp006_catboost_longsearch`: heavier CatBoost Optuna search
- `exp007_catboost_seedbag`: fixed-param CatBoost seed bagging
- `exp008_lightgbm_onehot`: diversity model with one-hot encoded categoricals
- `exp009_xgboost_onehot`: diversity model with one-hot encoded categoricals
- `exp010_blend_bucket_features`: add sign-aware balance and bucketed business-state features to the blend CatBoost path
- `exp012_blend_bucket_features_fixed`: fixed-param full evaluation of the best partial `exp010` search snapshot
- `exp017_blend_bucket_finance_interactions_screen`: cheap fixed-param screen for curated finance and balance interactions on the current best pruned CatBoost path
- `exp018_blend_bucket_contact_interactions_screen`: cheap fixed-param screen for curated contact-history interactions on the current best pruned CatBoost path
- `exp019_blend_bucket_state_crosses_screen`: cheap fixed-param screen for extra low-cardinality state crosses on the current best pruned CatBoost path
- `exp020_blend_bucket_adv_weighted_screen`: cheap screen for adversarial train weighting on the `exp012` bucket feature path
- `exp020_blend_bucket_adv_weighted_fixed`: fixed-param `exp012` rerun with adversarial train weighting toward test-like rows to address measured train/test drift
- `exp021_blend_bucket_seedbag5_fixed`: fixed-param `exp012` rerun with a wider five-seed CatBoost bag to reduce prediction variance on the public-best path
- `exp025_gpu_nn_trio`: GPU-only custom neural trio (attention, FT-style transformer, tabresnet) trained from scratch with multi-seed CV; exports model-level and blend artifacts

## Kaggle Ops

Preferred auth: `~/.kaggle/kaggle.json`.

Current working workaround in this repo:

- Kaggle auth is available from the interactive WSL `admin` `zsh` session via
  `KAGGLE_API_TOKEN` in `/home/admin/.zshrc`
- non-interactive shells may not see that token
- for agent-driven Kaggle actions, use the same shell path that works manually:

```bash
wsl.exe -u admin --cd /mnt/c/Users/dbxdr_iytiz92/Dropbox/fiicode \
  zsh -ic "source .venv/bin/activate && python -m src.submit --submission outputs/submissions/exp003_blend_features/submission.csv --message 'exp003 blend' --run-kaggle-cli"
```

If you want the workflow to be less shell-dependent, move the token to
`/home/admin/.kaggle/kaggle.json`.

Install deps:

```bash
python -m pip install -r requirements.txt
```

Download competition data:

```bash
kaggle competitions download -c fiicode-2026-ai-competition -p data
```

Submit predictions:

```bash
kaggle competitions submit -c fiicode-2026-ai-competition -f outputs/submissions/exp003_blend_features/submission.csv -m "exp003 blend"
```

Notebook sync:

```bash
kaggle kernels pull YOUR_USERNAME/YOUR_KERNEL_SLUG -p kaggle_kernel --metadata
kaggle kernels push -p kaggle_kernel
kaggle kernels output YOUR_USERNAME/YOUR_KERNEL_SLUG -p kaggle_outputs
```

## Tracking

- `outputs/logs/experiment_runs.csv`: local run registry with model family, feature set, local AUC, and OOF correlation to prior best
- `outputs/logs/leaderboard_journal.csv`: validated/submitted/scored Kaggle journal
- `outputs/logs/<experiment>/best_run_summary.json`: canonical experiment artifact

## Continuous Optimization Loop

If you want an agent to push continuously, give it an explicit submission budget
and the right shell context.

Recommended prompt pattern:

```text
Maximize Kaggle score for this repo autonomously.

You may:
- edit code
- run training and CV
- validate submissions
- submit to Kaggle from my WSL admin zsh context
- update experiment_runs.csv and leaderboard_journal.csv

Rules:
- keep 2 submissions in reserve each day
- only submit if the candidate is locally better than the current best, or if it adds strong model diversity for a blend
- prefer cheap experiments first, then promote winners to heavier runs
- after every completed run, decide the next best experiment without asking unless blocked
- after every Kaggle score, reassess the strategy against the current incumbent
```

Aggressive variant:

```text
Run a continuous Kaggle optimization loop.
Use up to 3 submissions today.
Keep pushing until you hit a blocker, quota limit, or a clearly better incumbent.
Prioritize leaderboard gains over elegance.
```

Use `AGENTS.md` for Codex-facing repo rules and `.claude/` for Claude-facing project memory.
