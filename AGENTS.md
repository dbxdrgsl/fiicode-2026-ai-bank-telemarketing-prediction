# AGENTS.md

## Goal

Compete on FiiCode 2026 with reproducible local experiments, disciplined
submission tracking, and score-focused iteration.

## Hard Rules

- Never modify files under `data/raw/`.
- Do not use external data, external pretrained models, or generic AutoML systems.
- Keep notebook-specific logic out of `src/`.
- Every experiment must be defined by a YAML file in `experiments/`.
- Every training run must save:
  - best params
  - OOF predictions
  - submission path
  - summary JSON
  - trials CSV
- Prefer one hypothesis per experiment.
- Kaggle submission actions must run from the working auth context:
  `wsl.exe -u admin --cd /mnt/c/Users/dbxdr_iytiz92/Dropbox/fiicode zsh -ic "source .venv/bin/activate && ..."`

## Repo Map

- `src/train.py`: main multi-model experiment entrypoint
- `src/infer.py`: rebuild a submission from saved summary/best params
- `src/cv.py`: ablations with progress output and quick mode
- `src/blend.py`: OOF-based blending and correlation analysis
- `src/submit.py`: submission validation, Kaggle CLI handoff, and journal updates
- `src/modeling.py`: CatBoost/LightGBM/XGBoost backends
- `src/tracking.py`: experiment and leaderboard CSV journals
- `src/features.py`: shared feature engineering and data loading
- `experiments/`: YAML configs
- `outputs/logs/`: experiment summaries, trials, journals, and blend diagnostics

## Required Commands

Train:

```bash
python -m src.train --config experiments/exp003_blend_features.yaml
```

Ablate:

```bash
python -m src.cv --config experiments/exp003_blend_features.yaml
python -m src.cv --config experiments/exp003_blend_features.yaml --quick
```

Rebuild:

```bash
python -m src.infer --config experiments/exp003_blend_features.yaml
```

Validate or submit:

```bash
python -m src.submit --submission outputs/submissions/exp003_blend_features/submission.csv --message "exp003 blend"
```

Agent-safe Kaggle submit path:

```bash
wsl.exe -u admin --cd /mnt/c/Users/dbxdr_iytiz92/Dropbox/fiicode zsh -ic "source .venv/bin/activate && python -m src.submit --submission outputs/submissions/exp003_blend_features/submission.csv --message 'exp003 blend' --run-kaggle-cli"
```

Blend:

```bash
python -m src.blend --name exp010_blend --summaries outputs/logs/exp003_blend_features/best_run_summary.json outputs/logs/exp008_lightgbm_onehot/best_run_summary.json outputs/logs/exp009_xgboost_onehot/best_run_summary.json
```

## Change Protocol

Before editing training logic, explain:

- what changed
- why it should help
- how to validate it
- the exact rerun command

After editing:

- keep train and inference on the same feature path
- preserve the artifact contract across model families
- update `README.md` instead of creating new top-level docs

## Agent Roles

- Research agent: summarize competition rules, metric, and candidate baselines
- Data agent: inspect imbalance, skew, and leakage risk
- Modeling agent: propose feature/model/CV changes
- Experiment agent: create the next YAMLs
- Review agent: audit reproducibility, submission validity, and regression risk

## Prompt Recipes

Experiment planner:

```text
Based on outputs/logs and current configs, propose the next 5 experiments.
Constraint:
- each experiment must isolate one hypothesis
- no duplicate ideas
- prefer cheap experiments first
Format as a table.
```

Blend auditor:

```text
Review these candidate experiment summaries and OOF files.
Return:
- which models are redundant
- which models are diverse enough to blend
- the most promising 2-model and 3-model blend ideas
```

Submission validator:

```text
Check whether the current pipeline will produce a valid Kaggle submission.
Validate:
- row count
- id column
- target column names
- ordering assumptions
- NaN/inf values
- dtype issues
Do not modify code until you explain the risks.
```

Continuous optimizer:

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
