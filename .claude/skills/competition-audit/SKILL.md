---
name: competition-audit
description: FiiCode 2026 competition guardrails and submission checklist. Use when changing training logic, data access, notebook packaging, or preparing a submission.
---
# FiiCode 2026 Competition Audit

Use this skill whenever work could affect compliance, reproducibility, or final
submission readiness.

## Hard Rules

- Do not introduce external data or external datasets.
- Do not introduce external pretrained models.
- Keep model search explicit and hand-authored; do not replace the workflow with
  generic AutoML systems.
- Keep competition data private and within the intended competition workflow.

## Metric And Submission

- Optimize for ROC-AUC.
- Final submission must be a CSV with header `id,Subscribed`.
- `Subscribed` must be a probability between 0 and 1.
- Preserve `test.csv` row order unless there is a very explicit reason not to.
- Check for missing ids, duplicate ids, and schema drift before recommending a
  submission.

## Leaderboard Discipline

- Treat public leaderboard gains as untrusted until local validation also
  improves.
- Avoid advice that assumes unlimited submissions; the competition has a daily
  submission cap.

## Reproducibility

- Provide exact rerun commands for local and Kaggle use when relevant.
- Keep paths valid both locally and on Kaggle.
- If the workflow changes, update `README.md` or `CLAUDE.md` so future agents
  inherit the new process.

## Final Checklist

1. No prohibited data, models, or tools introduced.
2. Validation logic still makes sense.
3. Submission schema and probability range are correct.
4. Commands for rerun, sync, or submit are explicit.
5. Repo structure remains minimal.
