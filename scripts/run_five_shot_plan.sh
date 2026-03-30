#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_five_shot_plan.sh submit exp058
#   bash scripts/run_five_shot_plan.sh validate exp058
#
# This script does not auto-submit multiple shots; it provides deterministic
# one-shot commands for the locked 5-submission playbook.

MODE=${1:-validate}
CAND=${2:-exp058}

submit_cmd() {
  local submission_path="$1"
  local message="$2"
  if [[ "$MODE" == "submit" ]]; then
    python -m src.submit --submission "$submission_path" --message "$message" --run-kaggle-cli
  else
    python -m src.submit --submission "$submission_path" --message "$message"
  fi
}

case "$CAND" in
  exp058)
    submit_cmd "outputs/submissions/exp058_blend_exp052_catneural/submission.csv" "exp058 cat+neural bestshot"
    ;;
  exp056)
    submit_cmd "outputs/submissions/exp056_blend_exp052_catlineage/submission.csv" "exp056 catlineage fallback"
    ;;
  exp057)
    submit_cmd "outputs/submissions/exp057_blend_exp052_allstrong/submission.csv" "exp057 allstrong fallback"
    ;;
  exp052)
    submit_cmd "outputs/submissions/exp052_blend_poststack/submission.csv" "exp052 stable anchor"
    ;;
  exp048)
    submit_cmd "outputs/submissions/exp048_blend_exp012_exp046/submission.csv" "exp048 incumbent anchor"
    ;;
  *)
    echo "Unknown candidate: $CAND"
    echo "Allowed: exp058 exp056 exp057 exp052 exp048"
    exit 1
    ;;
esac
