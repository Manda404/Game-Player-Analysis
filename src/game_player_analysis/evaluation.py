"""Shared regression metrics, post-processing and error diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from game_player_analysis.config import ID_COLUMNS, TARGET


def regression_metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    """Compute the single metric definition used throughout the project."""
    values = np.asarray(prediction, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, values)),
        "rmse": float(mean_squared_error(y_true, values) ** 0.5),
        "r2": float(r2_score(y_true, values)),
    }


def snap_to_rank_grid(
    prediction: np.ndarray | pd.Series,
    max_rank: pd.Series,
) -> np.ndarray:
    """Project bounded predictions onto each row's legal rank grid."""
    values = np.clip(np.asarray(prediction, dtype=float), 0.0, 1.0)
    ranks = np.asarray(max_rank, dtype=float)
    span = ranks - 1.0
    snapped = np.divide(
        np.rint(values * span),
        span,
        out=np.zeros_like(values),
        where=span > 0,
    )
    return np.clip(snapped, 0.0, 1.0)


def overfitting_comment(train_mae: float, validation_mae: float) -> str:
    """Turn the train/validation MAE gap into a readable diagnostic."""
    if validation_mae <= 0:
        return "non évalué"
    relative_gap = max(0.0, validation_mae - train_mae) / validation_mae
    if relative_gap < 0.05:
        return "faible écart"
    if relative_gap < 0.20:
        return "écart modéré"
    return "écart élevé"


def error_by_match_size(
    frame: pd.DataFrame,
    prediction: np.ndarray,
) -> pd.DataFrame:
    """Summarize absolute error by declared ranking-grid size."""
    diagnostic = pd.DataFrame(
        {
            "maxRank": frame["maxRank"].to_numpy(),
            "absolute_error": np.abs(frame[TARGET].to_numpy() - prediction),
        }
    )
    diagnostic["match_size"] = pd.cut(
        diagnostic["maxRank"],
        bins=[0, 5, 20, 80, np.inf],
        labels=["very_small", "small", "medium", "large"],
    )
    return diagnostic.groupby("match_size", observed=True)["absolute_error"].agg(
        rows="size", mae="mean"
    )


def build_submission(test: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    """Build an ordered submission with legal target values."""
    if len(test) != len(prediction):
        raise ValueError("Prediction length does not match the test dataset")
    submission = test.loc[:, list(ID_COLUMNS)].copy()
    submission[TARGET] = snap_to_rank_grid(prediction, test["maxRank"])
    return submission
