"""Compact, reproducible analysis of the Gameloft player dataset."""

from game_player_analysis.cleaning import clean_ranking_sentinels
from game_player_analysis.data import load_train_test
from game_player_analysis.features import build_model_features
from game_player_analysis.validation import make_group_folds

__version__ = "1.0.0"

__all__ = [
    "build_model_features",
    "clean_ranking_sentinels",
    "load_train_test",
    "make_group_folds",
]
