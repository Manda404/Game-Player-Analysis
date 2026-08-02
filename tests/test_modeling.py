"""Tests for evaluation, model comparison and artifact alignment."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor

from game_player_analysis.config import TARGET
from game_player_analysis.evaluation import (
    build_submission,
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
