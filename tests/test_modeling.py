"""Tests for evaluation, model comparison and artifact alignment."""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor

from game_player_analysis.config import TARGET
from game_player_analysis.evaluation import (
    build_submission,
    catboost_holdout_shap_values,
    error_by_match_size,
    overfitting_comment,
    snap_to_rank_grid,
)
from game_player_analysis.features import build_model_features
from game_player_analysis.modeling import (
    compare_models,
    load_model_bundle,
    paired_fold_uncertainty,
    save_model_bundle,
)
from game_player_analysis.validation import make_group_folds
from game_player_analysis.visualization import (
    apply_style,
    plot_model_comparison,
    plot_predictions,
    plot_fold_count_study,
    plot_shap_summary,
    plot_target_distribution,
    plot_top_correlations,
)


def test_comparison_returns_baseline_and_one_shared_model(player_frame):
    X = build_model_features(player_frame)
    folds = make_group_folds(player_frame, n_splits=2)
    results, predictions, details = compare_models(
        X,
        player_frame[TARGET],
        folds,
        models={"Small tree": DecisionTreeRegressor(max_depth=2, random_state=42)},
    )
    expected = {"Mean baseline", "Median baseline", "Small tree"}
    assert set(results["model"]) == expected
    assert set(predictions) == set(details) == expected
    assert all(len(values) == len(player_frame) for values in predictions.values())

    uncertainty = paired_fold_uncertainty(details, "Small tree", bootstrap_samples=100)
    assert set(uncertainty["candidate_model"]) == {"Mean baseline", "Median baseline"}
    assert uncertainty["folds"].eq(2).all()


def test_rank_grid_and_submission_are_bounded(player_frame):
    prediction = np.array([-0.1, 0.504, 0.999, 1.2])
    snapped = snap_to_rank_grid(prediction, player_frame["maxRank"])
    submission = build_submission(player_frame, prediction)
    assert np.all((snapped >= 0) & (snapped <= 1))
    assert submission[TARGET].tolist() == snapped.tolist()


def test_saved_bundle_keeps_ordered_feature_contract(tmp_path, player_frame):
    X = build_model_features(player_frame)
    model = DecisionTreeRegressor(max_depth=2, random_state=42).fit(X, player_frame[TARGET])
    benchmark = compare_models(
        X,
        player_frame[TARGET],
        make_group_folds(player_frame, n_splits=2),
        models={"Small tree": DecisionTreeRegressor(max_depth=2, random_state=42)},
    )[0]
    save_model_bundle(model, X.columns, benchmark, output_dir=tmp_path)
    loaded, manifest = load_model_bundle(tmp_path)
    assert loaded.predict(X).shape == (len(X),)
    assert manifest["features"] == list(X.columns)
    assert manifest["feature_count"] == X.shape[1]


def test_error_diagnostics_and_overfitting_labels(player_frame):
    prediction = np.full(len(player_frame), 0.5)
    diagnostic = error_by_match_size(player_frame, prediction)
    assert diagnostic["rows"].sum() == len(player_frame)
    assert overfitting_comment(0.099, 0.1) == "faible écart"
    assert overfitting_comment(0.085, 0.1) == "écart modéré"
    assert overfitting_comment(0.01, 0.1) == "écart élevé"


def test_notebook_visualizations_return_axes(player_frame):
    apply_style()
    results = pd.DataFrame({"model": ["A", "B"], "mae": [0.1, 0.2]})
    axes = [
        plot_target_distribution(player_frame),
        plot_top_correlations(player_frame),
        plot_model_comparison(results),
        plot_predictions(player_frame[TARGET], player_frame[TARGET]),
    ]
    assert all(axis.figure is not None for axis in axes)
    plt.close("all")


def test_catboost_shap_reconstructs_frozen_holdout_predictions():
    features = pd.DataFrame(
        {
            "killRank": np.arange(12, dtype=float),
            "walkDist": np.linspace(0.0, 1200.0, 12),
            "mode_solo": [1, 0] * 6,
        }
    )
    target = pd.Series(np.linspace(0.0, 1.0, len(features)))
    train_index = np.arange(9)
    holdout_index = np.arange(9, 12)
    model = CatBoostRegressor(
        iterations=20,
        depth=3,
        learning_rate=0.1,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    ).fit(features.iloc[train_index], target.iloc[train_index])
    holdout_prediction = np.clip(model.predict(features.iloc[holdout_index]), 0.0, 1.0)

    importance, sample, values = catboost_holdout_shap_values(
        model,
        features,
        target,
        holdout_index,
        holdout_prediction,
        max_samples=2,
    )

    assert importance["feature"].tolist() == ["killRank", "walkDist", "mode_solo"]
    assert len(sample) == 2
    assert len(values) == len(sample) * features.shape[1]
    assert np.allclose(
        values.groupby("source_row", sort=False)["shap_value"].sum().to_numpy()
        + sample["expected_value"].to_numpy(),
        sample["raw_prediction"].to_numpy(),
    )

    axes = plot_shap_summary(importance, values, top_n=3)
    assert all(axis.figure is not None for axis in axes)
    plt.close("all")


def test_fold_count_visualization_returns_axes():
    sensitivity = pd.DataFrame(
        {
            "n_splits": [3, 5, 3, 5],
            "model": ["CatBoost", "CatBoost", "XGBoost", "XGBoost"],
            "mae": [0.062, 0.061, 0.064, 0.063],
            "mae_std": [0.001, 0.0011, 0.0012, 0.0013],
        }
    )
    decision = pd.DataFrame(
        {
            "n_splits": [3, 5],
            "fit_cost_relative_to_5_folds": [0.65, 1.0],
            "mean_validation_rows": [13376, 8026],
        }
    )
    axes = plot_fold_count_study(sensitivity, decision)
    assert all(axis.figure is not None for axis in axes)
    plt.close("all")
