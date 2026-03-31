# Re-run 0.94939 In Detail

## Goal

Recreate the exact tracked artifact set for:

```text
exp064b_exp058_anti_light_gradient
```

with final public LB:

```text
0.94939
```

## Required Existing Inputs

These source files must already exist in the repo:

- `outputs/submissions/exp058_blend_exp052_catneural/submission.csv`
- `outputs/submissions/exp_light_attention/submission.csv`
- `outputs/oof/exp058_blend_exp052_catneural/oof_predictions.csv`
- `outputs/oof/exp_light_attention/oof_predictions.csv`
- `data/raw/train.csv`

## Tracked Experiment Definition

The versioned config file is:

```text
experiments/exp064b_exp058_anti_light_gradient.yaml
```

Key fields:

- `formula: clip((1 + alpha) * source - alpha * side, 0, 1)`
- `alpha: 0.11`

## Environment

Use the WSL repo environment:

```bash
cd /mnt/c/Users/dbxdr_iytiz92/Dropbox/fiicode
source .venv/bin/activate
```

## Generate The Full Artifact Bundle

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

## Expected Generated Files

- `experiments/exp064b_exp058_anti_light_gradient.yaml`
- `outputs/submissions/exp064b_exp058_anti_light_gradient/submission.csv`
- `outputs/oof/exp064b_exp058_anti_light_gradient/oof_predictions.csv`
- `outputs/logs/exp064b_exp058_anti_light_gradient/best_run_summary.json`
- `outputs/models/exp064b_exp058_anti_light_gradient/best_params.json`

## Validate Locally

```bash
python -m src.submit \
  --submission outputs/submissions/exp064b_exp058_anti_light_gradient/submission.csv \
  --message "exp064b anti-light gradient alpha 0.11"
```

## Submit Through The Working Kaggle Auth Context

```bash
wsl.exe -u admin --cd /mnt/c/Users/dbxdr_iytiz92/Dropbox/fiicode \
  zsh -ic "source /home/admin/.venvs/fiicode/bin/activate && python -m src.submit --submission outputs/submissions/exp064b_exp058_anti_light_gradient/submission.csv --message 'exp064b anti-light gradient alpha 0.11' --run-kaggle-cli"
```

## Record The Score

The tracked public score for the final run was:

```text
0.94939
```

To log that score:

```bash
python -m src.submit \
  --submission outputs/submissions/exp064b_exp058_anti_light_gradient/submission.csv \
  --message "exp064b anti-light gradient alpha 0.11" \
  --public-lb 0.94939 \
  --notes "User-reported Kaggle public LB after final submission"
```
