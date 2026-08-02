"""Reproduce the benchmark, final model and submission from raw data."""

from pathlib import Path

import pandas as pd

from game_player_analysis.config import ARTIFACT_DIR, OUTPUT_DIR, TARGET
from game_player_analysis.data import load_train_test
from game_player_analysis.evaluation import build_submission, regression_metrics, snap_to_rank_grid
from game_player_analysis.features import build_train_test_features
from game_player_analysis.modeling import (
    build_model_candidates,
    compare_models,
    cross_validate_model,
    fit_final_model,
    save_model_bundle,
)
from game_player_analysis.validation import make_group_folds


def run() -> dict[str, Path]:
    """Execute the complete, leakage-safe modeling workflow."""
    train, test = load_train_test()
    X_train, X_test = build_train_test_features(train, test, include_kill_rank=True)
    behavior_train, _ = build_train_test_features(train, test, include_kill_rank=False)
    folds = make_group_folds(train)

    benchmark, oof_predictions, _ = compare_models(X_train, train[TARGET], folds)
    winner = benchmark.loc[benchmark["model"].ne("Median baseline"), "model"].iloc[0]
    winner_model = build_model_candidates()[winner]
    behavior_summary, behavior_oof, _ = cross_validate_model(
        winner,
        winner_model,
        behavior_train,
        train[TARGET],
        folds,
    )
    post_oof = oof_predictions[winner]
    post_metrics = regression_metrics(train[TARGET], post_oof)
    snapped_metrics = regression_metrics(
        train[TARGET], snap_to_rank_grid(post_oof, train["maxRank"])
    )
    ablation = pd.DataFrame(
        [
            {
                "scenario": "post_match_with_killRank",
                **post_metrics,
                "features": X_train.shape[1],
            },
            {
                "scenario": "behavior_without_killRank",
                "mae": behavior_summary["mae"],
                "rmse": behavior_summary["rmse"],
                "r2": behavior_summary["r2"],
                "features": behavior_train.shape[1],
            },
            {
                "scenario": "post_match_grid_snapped",
                **snapped_metrics,
                "features": X_train.shape[1],
            },
        ]
    )

    final_model = fit_final_model(winner, X_train, train[TARGET])
    paths = save_model_bundle(final_model, X_train.columns, benchmark)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ablation_path = ARTIFACT_DIR / "killrank_ablation.csv"
    ablation.to_csv(ablation_path, index=False)

    prediction = final_model.predict(X_test)
    submission = build_submission(test, prediction)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    return {**paths, "ablation": ablation_path, "submission": submission_path}


def main() -> None:
    """CLI entry point."""
    outputs = run()
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
