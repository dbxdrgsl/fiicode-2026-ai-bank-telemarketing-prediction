from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.modeling import normalize_model_family

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_COMPETITION_SLUG = "fiicode-2026-ai-competition"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class ExperimentConfig:
    name: str
    competition_slug: str = DEFAULT_COMPETITION_SLUG
    model_family: str = "catboost"
    feature_set: str = "focused"
    drop_columns: list[str] = field(default_factory=list)
    data_dir: str | None = None
    output_root: str = "outputs"
    n_trials: int = 30
    search_folds: int = 5
    search_seeds: list[int] = field(default_factory=lambda: [42])
    final_folds: int = 5
    final_seeds: list[int] = field(default_factory=lambda: [42, 2024, 3407])
    early_stop: int = 250
    study_name: str | None = None
    storage: str | None = None
    timeout: int | None = None
    thread_count: int = -1
    with_class_weight: bool = True
    fixed_params_path: str | None = None
    train_weight_mode: str | None = None
    train_weight_params: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.study_name:
            self.study_name = self.name
        self.model_family = normalize_model_family(self.model_family)
        if self.train_weight_mode:
            self.train_weight_mode = self.train_weight_mode.strip().lower()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentPaths:
    root: Path
    output_root: Path
    models_dir: Path
    oof_dir: Path
    submissions_dir: Path
    logs_dir: Path
    best_params_path: Path
    oof_path: Path
    submission_path: Path
    trials_path: Path
    summary_path: Path
    ablation_path: Path
    experiment_runs_path: Path
    leaderboard_journal_path: Path


def load_experiment_config(path: Path) -> ExperimentConfig:
    if yaml is None:
        raise ModuleNotFoundError("PyYAML not installed. Install with: pip install pyyaml")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw.setdefault("name", path.stem)
    raw.setdefault("study_name", raw["name"])
    return ExperimentConfig(**raw)


def resolve_experiment_paths(config: ExperimentConfig) -> ExperimentPaths:
    root = repo_root()
    output_root = (root / config.output_root).resolve()
    models_dir = output_root / "models" / config.name
    oof_dir = output_root / "oof" / config.name
    submissions_dir = output_root / "submissions" / config.name
    logs_dir = output_root / "logs" / config.name

    return ExperimentPaths(
        root=root,
        output_root=output_root,
        models_dir=models_dir,
        oof_dir=oof_dir,
        submissions_dir=submissions_dir,
        logs_dir=logs_dir,
        best_params_path=models_dir / "best_params.json",
        oof_path=oof_dir / "oof_predictions.csv",
        submission_path=submissions_dir / "submission.csv",
        trials_path=logs_dir / "optuna_trials.csv",
        summary_path=logs_dir / "best_run_summary.json",
        ablation_path=logs_dir / "ablation_results.csv",
        experiment_runs_path=output_root / "logs" / "experiment_runs.csv",
        leaderboard_journal_path=output_root / "logs" / "leaderboard_journal.csv",
    )


def ensure_output_dirs(paths: ExperimentPaths) -> None:
    for path in (
        paths.output_root / "models",
        paths.output_root / "oof",
        paths.output_root / "submissions",
        paths.output_root / "logs",
        paths.models_dir,
        paths.oof_dir,
        paths.submissions_dir,
        paths.logs_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
