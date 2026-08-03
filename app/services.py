"""In-memory services behind the private Streamlit interface."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import clone

from game_player_analysis.cleaning import clean_ranking_sentinels, quality_issues
from game_player_analysis.config import ID_COLUMNS, RANDOM_STATE, TARGET
from game_player_analysis.data import DataValidationError, dataset_summary, validate_dataset
from game_player_analysis.evaluation import build_submission, regression_metrics
from game_player_analysis.features import build_model_features
from game_player_analysis.modeling import (
    cross_validate_model,
    load_model_bundle,
    randomized_model_search,
)
from game_player_analysis.validation import make_final_group_holdout, make_group_folds

CATBOOST_LIMITS: dict[str, tuple[int | float, ...]] = {
    "iterations": (400, 800, 1_200),
    "learning_rate": (0.03, 0.05, 0.08),
    "depth": (4, 6, 8),
    "l2_leaf_reg": (1.0, 3.0, 5.0, 10.0),
    "random_strength": (0.5, 1.0, 2.0),
    "border_count": (64, 128, 254),
}


def read_uploaded_dataset(content: bytes, *, require_target: bool) -> pd.DataFrame:
    """Read and validate one official-shaped CSV entirely in memory."""
    frame = pd.read_csv(
        BytesIO(content),
        sep=";",
        dtype={**{column: "string" for column in ID_COLUMNS}, "gameType": "string"},
        parse_dates=["date"],
    )
    validate_dataset(frame, require_target=require_target)
    return frame


def validate_uploaded_pair(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Reject an uploaded train/test pair sharing official game identifiers."""
    shared_games = set(train["gameId"]).intersection(test["gameId"])
    if shared_games:
        raise DataValidationError(f"Train and test share {len(shared_games)} gameId value(s)")


def private_data_overview(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the compact, non-persistent quality overview shown in the UI."""
    return dataset_summary(frame), quality_issues(frame)


def build_catboost(parameters: dict[str, int | float]) -> CatBoostRegressor:
    """Build the sole user-trainable model with a bounded parameter contract."""
    unsupported = set(parameters).difference(CATBOOST_LIMITS)
    if unsupported:
        raise ValueError(f"Unsupported CatBoost parameters: {sorted(unsupported)}")
    invalid = {
        name: value for name, value in parameters.items() if value not in CATBOOST_LIMITS[name]
    }
    if invalid:
        raise ValueError(f"Parameters outside the allowed UI ranges: {invalid}")
    return CatBoostRegressor(
        **parameters,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def evaluate_uploaded_catboost(
    train: pd.DataFrame,
    parameters: dict[str, int | float],
) -> dict[str, object]:
    """Evaluate one bounded configuration, then fit it on all uploaded rows."""
    cleaned = clean_ranking_sentinels(train)
    features = build_model_features(cleaned, include_kill_rank=True)
    target = cleaned[TARGET]
    development_index, holdout_index = make_final_group_holdout(cleaned)
    development = cleaned.iloc[development_index].reset_index(drop=True)
    development_features = features.iloc[development_index].reset_index(drop=True)
    development_target = target.iloc[development_index].reset_index(drop=True)
    folds = make_group_folds(development)
    candidate = build_catboost(parameters)
    summary, _, fold_details = cross_validate_model(
        "User CatBoost configuration",
        candidate,
        development_features,
        development_target,
        folds,
    )
    holdout_model = clone(candidate).fit(
        features.iloc[development_index],
        target.iloc[development_index],
    )
    holdout_prediction = np.clip(
        holdout_model.predict(features.iloc[holdout_index]),
        0.0,
        1.0,
    )
    holdout_metrics = regression_metrics(target.iloc[holdout_index], holdout_prediction)
    final_model = clone(candidate).fit(features, target)
    return {
        "model": final_model,
        "features": list(features.columns),
        "parameters": parameters,
        "cv_summary": summary,
        "fold_details": fold_details,
        "holdout_metrics": holdout_metrics,
        "development_rows": len(development_index),
        "holdout_rows": len(holdout_index),
    }


def evaluate_reference_on_uploaded_data(
    train: pd.DataFrame,
    model_directory: str | Path,
) -> dict[str, float]:
    """Score the frozen reference model on the candidate's exact grouped holdout.

    Comparing to the repository metric would be misleading when a visitor has
    uploaded another dataset. This function uses the same deterministic holdout
    as ``evaluate_uploaded_catboost`` so promotion is based on like-for-like
    predictions.
    """
    cleaned = clean_ranking_sentinels(train)
    model, manifest = load_model_bundle(model_directory)
    expected_features = list(manifest["features"])
    features = build_model_features(
        cleaned,
        include_kill_rank="killRank" in expected_features,
    )
    if list(features.columns) != expected_features:
        raise ValueError("The uploaded train file does not match the reference feature contract")
    _, holdout_index = make_final_group_holdout(cleaned)
    prediction = np.clip(model.predict(features.iloc[holdout_index]), 0.0, 1.0)
    return regression_metrics(cleaned[TARGET].iloc[holdout_index], prediction)


def candidate_beats_baseline(
    candidate_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> bool:
    """Return whether one candidate has a strictly lower MAE than its baseline."""
    return float(candidate_metrics["mae"]) < float(baseline_metrics["mae"])


def search_uploaded_catboost(
    train: pd.DataFrame,
    *,
    n_trials: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run a small, fixed search space on development folds only."""
    if n_trials not in {1, 2, 4}:
        raise ValueError("The interface only permits 1, 2 or 4 tuning trials")
    cleaned = clean_ranking_sentinels(train)
    features = build_model_features(cleaned, include_kill_rank=True)
    development_index, _ = make_final_group_holdout(cleaned)
    development = cleaned.iloc[development_index].reset_index(drop=True)
    development_features = features.iloc[development_index].reset_index(drop=True)
    development_target = development[TARGET]
    folds = make_group_folds(development)
    search, best_parameters = randomized_model_search(
        build_catboost({}),
        development_features,
        development_target,
        folds,
        CATBOOST_LIMITS,
        n_iter=n_trials,
    )
    return search, best_parameters


def predict_uploaded_test(
    test: pd.DataFrame,
    model: CatBoostRegressor,
    expected_features: list[str],
) -> pd.DataFrame:
    """Predict an uploaded official test file with an in-memory user model."""
    cleaned = clean_ranking_sentinels(test)
    features = build_model_features(cleaned, include_kill_rank=True)
    if list(features.columns) != expected_features:
        raise ValueError("The uploaded test file does not match the trained feature contract")
    return build_submission(test, np.asarray(model.predict(features), dtype=float))
