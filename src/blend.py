from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import repo_root
from src.features import ID_COL, TARGET
from src.tracking import append_experiment_run, load_oof_frame, load_submission_frame, load_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Blend saved experiment predictions with OOF-driven weights.")
    parser.add_argument("--name", type=str, required=True, help="Blend experiment name.")
    parser.add_argument(
        "--summaries",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to best_run_summary.json files for candidate experiments.",
    )
    parser.add_argument("--samples", type=int, default=5000, help="Random simplex samples for weight search.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for weight search.")
    parser.add_argument("--max-gap", type=float, default=0.0015, help="Keep models within this AUC gap to best.")
    parser.add_argument("--max-corr", type=float, default=0.995, help="Keep weaker models if corr-to-best is below this.")
    parser.add_argument("--max-models", type=int, default=5, help="Maximum number of models to include in the search.")
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.0003,
        help="Required OOF gain over the best single model to mark the blend as submit-worthy.",
    )
    return parser.parse_args()


def load_candidates(summary_paths: list[Path]) -> list[dict]:
    candidates = []
    for summary_path in summary_paths:
        summary = load_summary(summary_path)
        experiment = summary["config"]["name"]
        oof_path = Path(summary["paths"]["oof"])
        submission_path = Path(summary["paths"]["submission"])
        oof = load_oof_frame(oof_path)
        submission = load_submission_frame(submission_path)
        candidates.append(
            {
                "name": experiment,
                "model_family": summary["config"].get("model_family", "catboost"),
                "final_auc": float(summary["final_auc"]),
                "summary_path": summary_path,
                "oof": oof,
                "submission": submission,
            }
        )
    return candidates


def build_correlation_matrix(candidates: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame({candidate["name"]: candidate["oof"]["prediction"].values for candidate in candidates})
    return frame.corr()


def select_candidates(candidates: list[dict], max_gap: float, max_corr: float, max_models: int) -> list[dict]:
    best_candidate = max(candidates, key=lambda item: item["final_auc"])
    for candidate in candidates:
        candidate["corr_to_best"] = float(
            np.corrcoef(best_candidate["oof"]["prediction"], candidate["oof"]["prediction"])[0, 1]
        )

    eligible = [
        candidate
        for candidate in candidates
        if candidate["name"] == best_candidate["name"]
        or candidate["final_auc"] >= best_candidate["final_auc"] - max_gap
        or candidate["corr_to_best"] < max_corr
    ]

    selected = [best_candidate]
    remainder = [candidate for candidate in eligible if candidate["name"] != best_candidate["name"]]
    remainder.sort(key=lambda item: (item["corr_to_best"], -item["final_auc"]))
    for candidate in remainder:
        if len(selected) >= max_models:
            break
        selected.append(candidate)
    return selected


def search_best_weights(candidates: list[dict], samples: int, seed: int) -> tuple[np.ndarray, float]:
    oof_matrix = np.column_stack([candidate["oof"]["prediction"].values for candidate in candidates])
    y_true = candidates[0]["oof"]["y_true"].values
    rng = np.random.default_rng(seed)

    candidate_weights = [np.eye(len(candidates))[i] for i in range(len(candidates))]
    candidate_weights.extend(rng.dirichlet(np.ones(len(candidates)), size=samples))

    best_auc = -np.inf
    best_weights = candidate_weights[0]
    for weights in candidate_weights:
        predictions = oof_matrix @ weights
        auc = roc_auc_score(y_true, predictions)
        if auc > best_auc:
            best_auc = auc
            best_weights = weights
    return np.asarray(best_weights, dtype=float), float(best_auc)


def main():
    args = parse_args()
    summary_paths = [path if path.is_absolute() else (repo_root() / path).resolve() for path in args.summaries]
    candidates = load_candidates(summary_paths)
    if len(candidates) < 2:
        raise ValueError("Need at least two candidate summaries to blend.")

    selected = select_candidates(candidates, args.max_gap, args.max_corr, args.max_models)
    best_single = max(selected, key=lambda item: item["final_auc"])
    correlation = build_correlation_matrix(selected)
    best_weights, best_blend_auc = search_best_weights(selected, args.samples, args.seed)

    oof_ids = selected[0]["oof"][[ID_COL, "y_true"]].copy()
    oof_ids["oof_pred"] = np.column_stack([candidate["oof"]["prediction"].values for candidate in selected]) @ best_weights

    submission = selected[0]["submission"][[ID_COL]].copy()
    submission[TARGET] = np.column_stack([candidate["submission"][TARGET].values for candidate in selected]) @ best_weights

    root = repo_root()
    output_root = root / "outputs"
    models_dir = output_root / "models" / args.name
    oof_dir = output_root / "oof" / args.name
    submissions_dir = output_root / "submissions" / args.name
    logs_dir = output_root / "logs" / args.name
    for path in (models_dir, oof_dir, submissions_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    best_params_path = models_dir / "best_params.json"
    oof_path = oof_dir / "oof_predictions.csv"
    submission_path = submissions_dir / "submission.csv"
    summary_path = logs_dir / "best_run_summary.json"
    correlation_path = logs_dir / "oof_correlation.csv"

    best_params_path.write_text(
        json.dumps(
            {
                "weights": {candidate["name"]: float(weight) for candidate, weight in zip(selected, best_weights)},
                "samples": args.samples,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    oof_ids.to_csv(oof_path, index=False)
    submission.to_csv(submission_path, index=False)
    correlation.to_csv(correlation_path)

    improvement = best_blend_auc - best_single["final_auc"]
    corr_to_best = float(np.corrcoef(oof_ids["oof_pred"], best_single["oof"]["prediction"])[0, 1])
    metadata = {
        "config_path": None,
        "config": {
            "name": args.name,
            "model_family": "blend",
            "feature_set": "mixed",
            "with_class_weight": False,
        },
        "data_dir": None,
        "study_storage": "blend-search",
        "existing_trials": 0,
        "requested_trials": args.samples,
        "best_trial_number": None,
        "search_auc": best_blend_auc,
        "final_auc": best_blend_auc,
        "best_params": {"weights": {candidate["name"]: float(weight) for candidate, weight in zip(selected, best_weights)}},
        "members": [
            {
                "name": candidate["name"],
                "model_family": candidate["model_family"],
                "final_auc": candidate["final_auc"],
                "corr_to_best": candidate["corr_to_best"],
            }
            for candidate in selected
        ],
        "best_single_model": best_single["name"],
        "best_single_auc": best_single["final_auc"],
        "improvement_over_best_single": improvement,
        "promoted": improvement >= args.min_improvement,
        "paths": {
            "submission": str(submission_path),
            "oof": str(oof_path),
            "summary": str(summary_path),
            "best_params": str(best_params_path),
            "correlation": str(correlation_path),
        },
    }
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    append_experiment_run(
        output_root / "logs" / "experiment_runs.csv",
        experiment=args.name,
        model_family="blend",
        feature_set="mixed",
        with_class_weight=False,
        search_auc=best_blend_auc,
        final_auc=best_blend_auc,
        oof_corr_to_best=corr_to_best,
        submission_path=submission_path,
        summary_path=summary_path,
        notes=f"Blend of {', '.join(candidate['name'] for candidate in selected)}",
    )

    print("=" * 70)
    print("FiiCode Blend Search")
    print("=" * 70)
    print(f"Selected models: {', '.join(candidate['name'] for candidate in selected)}")
    print(f"Best single:     {best_single['name']} ({best_single['final_auc']:.6f})")
    print(f"Blend AUC:       {best_blend_auc:.6f}")
    print(f"Improvement:     {improvement:.6f}")
    print(f"Promoted:        {improvement >= args.min_improvement}")
    print(f"Saved weights:   {best_params_path}")
    print(f"Saved OOF:       {oof_path}")
    print(f"Saved submission:{submission_path}")
    print(f"Saved summary:   {summary_path}")
    print(f"Saved corr:      {correlation_path}")


if __name__ == "__main__":
    main()
