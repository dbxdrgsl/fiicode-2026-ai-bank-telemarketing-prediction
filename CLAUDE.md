# Project Instructions

## Goal

- Maintain a minimal, reproducible Kaggle workspace for FiiCode 2026.
- Optimize ROC-AUC for the term-deposit prediction task without violating competition rules.

## Repo Shape

- Keep the repo config-driven and experiment-oriented.
- Raw competition files live in `data/raw/` and should be treated as immutable.
- Core code lives in `src/`.
- Notebook exports live in `notebooks/`.
- Generated outputs belong in `outputs/`.
- Do not reintroduce stale artifacts, caches, or duplicate docs unless explicitly requested.

## Primary Files

- `AGENTS.md`: cross-agent operating contract
- `README.md`: human-oriented project and competition guide
- `src/train.py`: main training and search entrypoint
- `src/cv.py`: ablations and CV helpers
- `src/infer.py`: submission rebuild
- `src/submit.py`: submission validation and Kaggle CLI handoff
- `src/features.py`: shared data prep and feature logic
- `experiments/*.yaml`: experiment definitions
- `notebooks/improved.ipynb`: Kaggle notebook export

## Competition Guardrails

- No external data.
- No external pretrained models.
- Do not introduce generic AutoML platforms or opaque automation-heavy pipelines.
- Submission files must use the schema `id,Subscribed` with probabilities in `[0, 1]`.
- Be careful with public leaderboard overfitting; prefer validation-backed changes.
- Keep changes reproducible and include exact rerun commands.

## Preferred Workflow

- Local training: `python -m src.train --config experiments/exp001_baseline.yaml`
- Ablation pass: `python -m src.cv --config experiments/exp001_baseline.yaml`
- Rebuild submission: `python -m src.infer --config experiments/exp001_baseline.yaml`
- Validate submission: `python -m src.submit --submission outputs/submissions/exp001_baseline/submission.csv`
- Kaggle submission: `kaggle competitions submit -c fiicode-2026-ai-competition -f outputs/submissions/exp001_baseline/submission.csv -m "exp001 baseline"`

## Kaggle Auth Note

- The reliable agent submission path is WSL `admin` interactive `zsh`
- `KAGGLE_API_TOKEN` currently lives in `/home/admin/.zshrc`
- direct non-interactive subprocess calls may miss that token
- when an agent needs to submit, prefer:
  `wsl.exe -u admin --cd /mnt/c/Users/dbxdr_iytiz92/Dropbox/fiicode zsh -ic "source .venv/bin/activate && python -m src.submit ... --run-kaggle-cli"`
- a future cleanup path is to move auth into `/home/admin/.kaggle/kaggle.json`

## Continuous Optimization Mode

If the user explicitly authorizes autonomous Kaggle iteration, the desired loop
is:

1. inspect current best local and leaderboard results
2. pick the cheapest high-signal next experiment
3. train locally
4. validate or quick-ablate
5. submit only if it is promotion-worthy
6. record the score in `outputs/logs/leaderboard_journal.csv`
7. immediately choose the next experiment

Default submission discipline:

- keep 2 daily submissions in reserve
- prefer local improvements and diversity gains over blind leaderboard fishing
- do not spam near-duplicate submissions

## Editing Guidance

- Prefer the focused CatBoost pipeline unless the user explicitly asks for broader experimentation.
- Keep train and inference on the same feature path.
- Keep notebook-compatible paths and avoid local-only assumptions.
- Prefer small, isolated experiment changes expressed in YAML configs.
- If docs need updates, consolidate them into `README.md` or `AGENTS.md` instead of adding new top-level docs.

## Claude Scaffold

- Project subagents live in `.claude/agents/`.
- Project skills live in `.claude/skills/`.
- Project settings live in `.claude/settings.json`.
- `AGENTS.md` is the shared operating contract across agent systems.
