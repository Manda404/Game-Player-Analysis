"""Tests for the unique feature-engineering implementation."""

import numpy as np

from game_player_analysis.config import BEHAVIOR_FEATURES, POST_MATCH_FEATURES, TARGET
from game_player_analysis.features import build_model_features, build_train_test_features


def test_feature_contract_is_finite_and_target_free(player_frame):
    features = build_model_features(player_frame, include_kill_rank=False)

    assert tuple(features.columns) == BEHAVIOR_FEATURES
    assert TARGET not in features
    assert np.isfinite(features.to_numpy()).all()
    assert features.loc[0, "walk_distance_per_match_minute"] == 0
    assert features.loc[0, "damage_per_kill"] == 0


def test_post_match_contract_adds_only_killrank(player_frame):
    behavior = build_model_features(player_frame)
    post_match = build_model_features(player_frame, include_kill_rank=True)

    assert tuple(post_match.columns) == POST_MATCH_FEATURES
    assert post_match.columns[0] == "killRank"
    assert post_match.iloc[:, 1:].equals(behavior)


def test_train_and_test_produce_identical_columns(player_frame):
    train_features, test_features = build_train_test_features(
        player_frame,
        player_frame.drop(columns=TARGET),
        include_kill_rank=True,
    )
    assert list(train_features.columns) == list(test_features.columns)
