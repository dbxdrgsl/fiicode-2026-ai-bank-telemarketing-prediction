# FiiCode 2026 Workspace

Lean local experiment repo for Kaggle competition `fiicode-2026-ai-competition`.
The repo now supports:

- CatBoost as the main tabular anchor
- LightGBM and XGBoost as diversity models
- OOF-based blending
- reproducible YAML experiments
- Kaggle CLI validation and submission journaling

## Re-run 0.94939

Final best public score captured in this repo: `0.94939`

- Final submission: `exp064b_exp058_anti_light_gradient`
- Public LB recorded in `outputs/logs/leaderboard_journal.csv`
- Exact formula: `clip(1.11 * exp058 - 0.11 * exp_light_attention, 0, 1)`

### Related Docs

- [`docs/README.md`](docs/README.md)
  Index of the focused final-score docs.
- [`docs/final_094939_solution.md`](docs/final_094939_solution.md)
  Explains what the final solution does, why it starts from `exp058`, why
  `exp062` mattered, and why the final submission uses an anti-light
  extrapolation.
- [`docs/final_094939_rerun.md`](docs/final_094939_rerun.md)
  Gives the full detailed rerun path for regenerating the exact `exp064b`
  artifact bundle and submitting it again.

### 1. Activate the WSL environment

```bash
cd /mnt/c/Users/dbxdr_iytiz92/Dropbox/fiicode
source .venv/bin/activate
```

### 2. Generate the full `exp064b` artifact bundle

The experiment definition file is already versioned in the repo:

- `experiments/exp064b_exp058_anti_light_gradient.yaml`

The command below regenerates the four derived runtime artifacts:

- `outputs/submissions/exp064b_exp058_anti_light_gradient/submission.csv`
- `outputs/oof/exp064b_exp058_anti_light_gradient/oof_predictions.csv`
- `outputs/logs/exp064b_exp058_anti_light_gradient/best_run_summary.json`
- `outputs/models/exp064b_exp058_anti_light_gradient/best_params.json`

It uses the two saved source prediction files already in the repo:

- `outputs/submissions/exp058_blend_exp052_catneural/submission.csv`
- `outputs/submissions/exp_light_attention/submission.csv`

Run:

```bash
python - <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

repo = Path(".")
experiment = "exp064b_exp058_anti_light_gradient"
alpha = 0.11

train = pd.read_csv(repo / "data" / "raw" / "train.csv", usecols=["id", "Subscribed"])

exp058_oof = pd.read_csv(repo / "outputs" / "oof" / "exp058_blend_exp052_catneural" / "oof_predictions.csv")
light_oof_raw = pd.read_csv(repo / "outputs" / "oof" / "exp_light_attention" / "oof_predictions.csv")

exp058 = pd.read_csv(repo / "outputs" / "submissions" / "exp058_blend_exp052_catneural" / "submission.csv")
light = pd.read_csv(repo / "outputs" / "submissions" / "exp_light_attention" / "submission.csv")

exp058_oof = exp058_oof.sort_values("id").reset_index(drop=True)
exp058 = exp058.sort_values("id").reset_index(drop=True)
light = light.sort_values("id").reset_index(drop=True)
light_oof = (
    train.merge(
        light_oof_raw[["id", "Subscribed"]].rename(columns={"Subscribed": "model_pred"}),
        on="id",
        how="inner",
    )
    .sort_values("id")
    .reset_index(drop=True)
)

if not exp058["id"].equals(light["id"]):
    raise SystemExit("id mismatch between exp058 and exp_light_attention")
if not exp058_oof["id"].equals(light_oof["id"]):
    raise SystemExit("train id mismatch between exp058 OOF and exp_light_attention OOF")

submission = exp058.copy()
submission["Subscribed"] = np.clip(
    (1.0 + alpha) * exp058["Subscribed"].to_numpy()
    - alpha * light["Subscribed"].to_numpy(),
    0.0,
    1.0,
)

oof = exp058_oof[["id", "y_true"]].copy()
oof["oof_pred"] = np.clip(
    (1.0 + alpha) * exp058_oof["oof_pred"].to_numpy()
    - alpha * light_oof["model_pred"].to_numpy(),
    0.0,
    1.0,
)

submission_dir = repo / "outputs" / "submissions" / experiment
oof_dir = repo / "outputs" / "oof" / experiment
logs_dir = repo / "outputs" / "logs" / experiment
models_dir = repo / "outputs" / "models" / experiment
for path in [submission_dir, oof_dir, logs_dir, models_dir]:
    path.mkdir(parents=True, exist_ok=True)

submission_path = submission_dir / "submission.csv"
oof_path = oof_dir / "oof_predictions.csv"
summary_path = logs_dir / "best_run_summary.json"
best_params_path = models_dir / "best_params.json"

submission.to_csv(submission_path, index=False)
oof.to_csv(oof_path, index=False)

best_params = {
    "formula": "clip((1+alpha)*exp058 - alpha*exp_light_attention, 0, 1)",
    "alpha": alpha,
}
best_params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

base_auc = float(roc_auc_score(exp058_oof["y_true"], exp058_oof["oof_pred"]))
final_auc = float(roc_auc_score(oof["y_true"], oof["oof_pred"]))

summary = {
    "config_path": str((repo / "experiments" / "exp064b_exp058_anti_light_gradient.yaml").resolve()),
    "config": {
        "name": experiment,
        "model_family": "blend",
        "feature_set": "mixed",
        "with_class_weight": False,
        "blend_type": "manual_extrapolation",
        "alpha": alpha,
    },
    "data_dir": str((repo / "data" / "raw").resolve()),
    "study_storage": "manual-gradient-extrapolation",
    "existing_trials": 0,
    "requested_trials": 0,
    "best_trial_number": None,
    "search_auc": final_auc,
    "final_auc": final_auc,
    "best_params": {"alpha": alpha},
    "members": [
        {"name": "exp058_blend_exp052_catneural", "model_family": "blend", "weight": 1.11},
        {"name": "exp_light_attention", "model_family": "attention", "weight": -0.11},
    ],
    "best_single_model": "exp058_blend_exp052_catneural",
    "best_single_auc": base_auc,
    "improvement_over_best_single": final_auc - base_auc,
    "promoted": True,
    "paths": {
        "submission": str(submission_path.resolve()),
        "oof": str(oof_path.resolve()),
        "summary": str(summary_path.resolve()),
        "best_params": str(best_params_path.resolve()),
    },
}
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("Generated:")
print(submission_path)
print(oof_path)
print(summary_path)
print(best_params_path)
PY
```

Expected generated files:

- `outputs/submissions/exp064b_exp058_anti_light_gradient/submission.csv`
- `outputs/oof/exp064b_exp058_anti_light_gradient/oof_predictions.csv`
- `outputs/logs/exp064b_exp058_anti_light_gradient/best_run_summary.json`
- `outputs/models/exp064b_exp058_anti_light_gradient/best_params.json`

### 3. Validate the rebuilt file locally

```bash
python -m src.submit \
  --submission outputs/submissions/exp064b_exp058_anti_light_gradient/submission.csv \
  --message "exp064b anti-light gradient alpha 0.11"
```

### 4. Submit through the working Kaggle auth context

```bash
wsl.exe -u admin --cd /mnt/c/Users/dbxdr_iytiz92/Dropbox/fiicode \
  zsh -ic "source /home/admin/.venvs/fiicode/bin/activate && python -m src.submit --submission outputs/submissions/exp064b_exp058_anti_light_gradient/submission.csv --message 'exp064b anti-light gradient alpha 0.11' --run-kaggle-cli"
```

### 5. Record the resulting public score

For the final run in this repo, the recorded public LB was `0.94939`.

To log a scored rerun:

```bash
python -m src.submit \
  --submission outputs/submissions/exp064b_exp058_anti_light_gradient/submission.csv \
  --message "exp064b anti-light gradient alpha 0.11" \
  --public-lb 0.94939 \
  --notes "User-reported Kaggle public LB after final submission"
```

### Files That Define the Final 0.94939 Run

- `experiments/exp064b_exp058_anti_light_gradient.yaml`
- `outputs/submissions/exp064b_exp058_anti_light_gradient/submission.csv`
- `outputs/oof/exp064b_exp058_anti_light_gradient/oof_predictions.csv`
- `outputs/logs/exp064b_exp058_anti_light_gradient/best_run_summary.json`
- `outputs/models/exp064b_exp058_anti_light_gradient/best_params.json`

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

Score a submission against an allowed labeled reference and reconcile it against scored Kaggle history:

```bash
python -m src.score_submissions \
  --reference path/to/reference_labels.csv \
  --submission outputs/submissions/exp048_blend_exp012_exp046/submission.csv

python -m src.score_submissions \
  --reference path/to/reference_labels.csv \
  --history \
  --report-json outputs/logs/submission_score_audit.json
```

Notes:

- this checks whether a local reference reproduces the displayed Kaggle public score to the chosen decimal precision
- exact public-score reproduction is not identifiable from leaderboard history alone; you need an allowed labeled reference file
- do not use external or leakage-derived labels for competition decisions

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
