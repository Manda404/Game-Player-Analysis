"""Shared regression metrics, post-processing and error diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.base import RegressorMixin, clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from game_player_analysis.config import ID_COLUMNS, TARGET
from game_player_analysis.data import game_mode_family


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
        return "not assessed"
    relative_gap = max(0.0, validation_mae - train_mae) / validation_mae
    if relative_gap < 0.05:
        return "small gap"
    if relative_gap < 0.20:
        return "moderate gap"
    return "large gap"


def error_by_match_size(
    frame: pd.DataFrame,
    prediction: np.ndarray,
) -> pd.DataFrame:
    """Summarize absolute error by declared maxRank grid size.

    ``maxRank`` is not the observed or actual number of players in a match. The
    historical function name is retained for compatibility, while the output
    uses the more accurate ``rank_grid_size`` label.
    """
    diagnostic = pd.DataFrame(
        {
            "maxRank": frame["maxRank"].to_numpy(),
            "absolute_error": np.abs(frame[TARGET].to_numpy() - prediction),
        }
    )
    diagnostic["rank_grid_size"] = pd.cut(
        diagnostic["maxRank"],
        bins=[0, 5, 20, 80, np.inf],
        labels=["very_small", "small", "medium", "large"],
    )
    return diagnostic.groupby("rank_grid_size", observed=True)["absolute_error"].agg(
        rows="size", mae="mean"
    )


def subgroup_error_summary(
    frame: pd.DataFrame,
    prediction: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Return a long-form error audit across decision-relevant subgroups."""
    values = np.asarray(prediction, dtype=float)
    if len(frame) != len(values):
        raise ValueError("Prediction length does not match the error-audit frame")
    diagnostic = frame.loc[
        :, [TARGET, "gameType", "gameId", "teamId", "date", "maxRank", "kills"]
    ].copy()
    diagnostic["prediction"] = values
    diagnostic["residual"] = diagnostic["prediction"] - diagnostic[TARGET]
    diagnostic["absolute_error"] = diagnostic["residual"].abs()
    diagnostic["squared_error"] = diagnostic["residual"].pow(2)
    diagnostic["mode_family"] = game_mode_family(diagnostic["gameType"])
    diagnostic["observed_team_rows"] = frame.groupby(["gameId", "teamId"])["gameId"].transform(
        "size"
    )
    diagnostic["observed_team_rows"] = diagnostic["observed_team_rows"].clip(upper=3)
    diagnostic["rank_grid_size"] = pd.cut(
        diagnostic["maxRank"],
        bins=[0, 5, 20, 80, np.inf],
        labels=["≤5", "6–20", "21–80", ">80"],
    )
    diagnostic["target_band"] = pd.cut(
        diagnostic[TARGET],
        bins=[-np.inf, 0.1, 0.25, 0.5, 0.75, 0.9, np.inf],
        labels=["0–0.1", "0.1–0.25", "0.25–0.5", "0.5–0.75", "0.75–0.9", "0.9–1"],
    )
    diagnostic["kill_band"] = pd.cut(
        diagnostic["kills"],
        bins=[-np.inf, 0, 1, 3, np.inf],
        labels=["0", "1", "2–3", "4+"],
    )
    diagnostic["pseudo_month"] = pd.to_datetime(diagnostic["date"]).dt.to_period("M").astype(str)

    dimensions = {
        "mode_family": diagnostic["mode_family"],
        "gameType": diagnostic["gameType"],
        "observed_team_rows": diagnostic["observed_team_rows"].astype(str),
        "rank_grid_size": diagnostic["rank_grid_size"],
        "target_band": diagnostic["target_band"],
        "kill_band": diagnostic["kill_band"],
        "pseudo_month": diagnostic["pseudo_month"],
    }
    rows: list[pd.DataFrame] = []
    for dimension, labels in dimensions.items():
        summary = (
            diagnostic.assign(_subgroup=labels)
            .groupby("_subgroup", observed=True)
            .agg(
                rows=(TARGET, "size"),
                target_mean=(TARGET, "mean"),
                prediction_mean=("prediction", "mean"),
                bias=("residual", "mean"),
                mae=("absolute_error", "mean"),
                mse=("squared_error", "mean"),
            )
            .reset_index()
            .rename(columns={"_subgroup": "subgroup"})
        )
        summary["rmse"] = np.sqrt(summary.pop("mse"))
        summary.insert(0, "dimension", dimension)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def largest_error_cases(
    frame: pd.DataFrame,
    prediction: np.ndarray | pd.Series,
    *,
    n: int = 25,
) -> pd.DataFrame:
    """Return the anonymized rows with the largest absolute OOF errors."""
    values = np.asarray(prediction, dtype=float)
    if len(frame) != len(values):
        raise ValueError("Prediction length does not match the frame")
    columns = [
        *ID_COLUMNS,
        "gameType",
        "maxRank",
        "kills",
        "damages",
        "walkDist",
        "rideDist",
        "heals",
        "weapons",
        "gameTime",
        TARGET,
    ]
    result = frame.loc[:, columns].copy()
    result["prediction"] = values
    result["residual"] = result["prediction"] - result[TARGET]
    result["absolute_error"] = result["residual"].abs()
    return result.nlargest(n, "absolute_error").reset_index(drop=True)


def holdout_permutation_importance(
    estimator: RegressorMixin,
    X: pd.DataFrame,
    y: pd.Series,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    *,
    n_repeats: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit once and calculate clipped-MAE permutation importance on holdout."""
    model = clone(estimator)
    model.fit(X.iloc[train_index], y.iloc[train_index])
    prediction = np.clip(model.predict(X.iloc[validation_index]), 0.0, 1.0)

    def clipped_negative_mae(
        fitted: RegressorMixin,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> float:
        candidate = np.clip(fitted.predict(features), 0.0, 1.0)
        return -float(mean_absolute_error(target, candidate))

    result = permutation_importance(
        model,
        X.iloc[validation_index],
        y.iloc[validation_index],
        scoring=clipped_negative_mae,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "mae_increase_mean": result.importances_mean,
            "mae_increase_std": result.importances_std,
        }
    ).sort_values("mae_increase_mean", ascending=False)
    return importance.reset_index(drop=True), prediction


def catboost_holdout_shap_values(
    estimator: RegressorMixin,
    X: pd.DataFrame,
    y: pd.Series,
    validation_index: np.ndarray,
    prediction: np.ndarray,
    *,
    max_samples: int = 2_000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Explain a frozen CatBoost model on a deterministic grouped-holdout sample.

    CatBoost computes exact TreeSHAP values natively through
    ``get_feature_importance(type="ShapValues")``.  The last column is the
    expected value; it is separated from the feature contributions so the
    returned long table can be inspected or plotted without a separate SHAP
    dependency.  The additivity identity is checked against raw predictions.
    """
    if not isinstance(estimator, CatBoostRegressor):
        raise TypeError("SHAP explanations require a fitted CatBoostRegressor")
    if max_samples < 1:
        raise ValueError("max_samples must be at least one")

    holdout_rows = np.asarray(validation_index, dtype=int)
    clipped_prediction = np.asarray(prediction, dtype=float)
    if len(holdout_rows) != len(clipped_prediction):
        raise ValueError("Prediction length does not match the SHAP holdout")

    if len(holdout_rows) > max_samples:
        generator = np.random.default_rng(random_state)
        sample_positions = np.sort(
            generator.choice(len(holdout_rows), size=max_samples, replace=False)
        )
    else:
        sample_positions = np.arange(len(holdout_rows))

    sample_rows = holdout_rows[sample_positions]
    sample_features = X.iloc[sample_rows].copy()
    shap_matrix = np.asarray(
        estimator.get_feature_importance(Pool(sample_features), type="ShapValues"),
        dtype=float,
    )
    expected_shape = (len(sample_features), sample_features.shape[1] + 1)
    if shap_matrix.shape != expected_shape:
        raise RuntimeError(
            "Unexpected CatBoost SHAP shape: "
            f"expected {expected_shape}, received {shap_matrix.shape}"
        )

    feature_contributions = shap_matrix[:, :-1]
    expected_value = shap_matrix[:, -1]
    raw_prediction = np.asarray(estimator.predict(sample_features), dtype=float)
    if not np.allclose(
        feature_contributions.sum(axis=1) + expected_value,
        raw_prediction,
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError("CatBoost SHAP values do not reconstruct the model prediction")
    if not np.allclose(
        np.clip(raw_prediction, 0.0, 1.0),
        clipped_prediction[sample_positions],
        rtol=1e-6,
        atol=1e-7,
    ):
        raise RuntimeError("SHAP model predictions differ from the frozen holdout predictions")

    feature_names = sample_features.columns.to_numpy()
    global_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(feature_contributions).mean(axis=0),
            "mean_shap": feature_contributions.mean(axis=0),
            "positive_shap_share": (feature_contributions > 0).mean(axis=0),
            "explained_rows": len(sample_rows),
            "holdout_rows": len(holdout_rows),
        }
    ).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    global_importance.insert(0, "rank", np.arange(1, len(global_importance) + 1))

    sample = pd.DataFrame(
        {
            "source_row": sample_rows,
            TARGET: y.iloc[sample_rows].to_numpy(),
            "raw_prediction": raw_prediction,
            "prediction": np.clip(raw_prediction, 0.0, 1.0),
            "expected_value": expected_value,
        }
    )
    sample["residual"] = sample["prediction"] - sample[TARGET]
    sample["absolute_error"] = sample["residual"].abs()

    shap_values = pd.DataFrame(
        {
            "source_row": np.repeat(sample_rows, len(feature_names)),
            "feature": np.tile(feature_names, len(sample_rows)),
            "feature_value": sample_features.to_numpy(dtype=float).reshape(-1),
            "shap_value": feature_contributions.reshape(-1),
        }
    )
    shap_values["absolute_shap_value"] = shap_values["shap_value"].abs()
    return global_importance, sample, shap_values


def build_submission(test: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    """Build an ordered submission with legal target values."""
    if len(test) != len(prediction):
        raise ValueError("Prediction length does not match the test dataset")
    submission = test.loc[:, list(ID_COLUMNS)].copy()
    submission[TARGET] = snap_to_rank_grid(prediction, test["maxRank"])
    return submission
