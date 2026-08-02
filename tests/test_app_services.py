"""Tests for the in-memory privacy and parameter controls used by Streamlit."""

import numpy as np
import pandas as pd
import pytest

from app.services import (
    build_catboost,
    candidate_beats_baseline,
    evaluate_reference_on_uploaded_data,
    private_data_overview,
    read_uploaded_dataset,
    validate_uploaded_pair,
)
from game_player_analysis.data import DataValidationError
from game_player_analysis.config import POST_MATCH_FEATURES


class _ConstantRegressor:
    def predict(self, features):
        return np.full(len(features), 0.5)


def test_uploaded_csv_is_validated_in_memory(player_frame):
    payload = player_frame.to_csv(index=False, sep=";").encode("utf-8")
    loaded = read_uploaded_dataset(payload, require_target=True)

    summary, issues = private_data_overview(loaded)
    assert len(loaded) == len(player_frame)
    assert summary["has_target"]
    assert "kills_without_damage" in issues.index


def test_uploaded_train_and_test_cannot_share_a_match(player_frame):
    train = player_frame.copy()
    test = player_frame.drop(columns="winRankPercentage").copy()

    with pytest.raises(DataValidationError, match="share"):
        validate_uploaded_pair(train, test)


def test_catboost_ui_rejects_unbounded_parameters():
    model = build_catboost({"iterations": 400, "depth": 6})
    assert model.get_params()["iterations"] == 400

    with pytest.raises(ValueError, match="allowed UI ranges"):
        build_catboost({"iterations": 999})

    with pytest.raises(ValueError, match="Unsupported"):
        build_catboost({"learning_rate": 0.05, "random_seed": 12})


def test_candidate_is_only_promotable_when_mae_improves():
    baseline = {"mae": 0.061, "rmse": 0.09, "r2": 0.9}

    assert candidate_beats_baseline({"mae": 0.0609}, baseline)
    assert not candidate_beats_baseline({"mae": 0.061}, baseline)
    assert not candidate_beats_baseline({"mae": 0.0611}, baseline)


def test_reference_is_scored_on_the_same_uploaded_group_holdout(player_frame, monkeypatch):
    train = pd.concat(
        [player_frame.assign(gameId=f"{group:014x}") for group in range(10)],
        ignore_index=True,
    )
    monkeypatch.setattr(
        "app.services.load_model_bundle",
        lambda _: (_ConstantRegressor(), {"features": list(POST_MATCH_FEATURES)}),
    )

    metrics = evaluate_reference_on_uploaded_data(train, "unused")

    assert set(metrics) == {"mae", "rmse", "r2"}
    assert metrics["mae"] >= 0
