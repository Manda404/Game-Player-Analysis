"""Tests for match-grouped validation."""

from game_player_analysis.validation import (
    audit_group_folds,
    make_group_folds,
    safe_game_groups,
    split_leakage_comparison,
)


def test_invalid_game_ids_are_isolated_by_row(player_frame):
    duplicated = player_frame.iloc[[3, 3]].reset_index(drop=True)
    groups = safe_game_groups(duplicated)
    assert groups.nunique() == 2


def test_gameid_never_crosses_grouped_folds(player_frame):
    folds = make_group_folds(player_frame, n_splits=2)
    audit = audit_group_folds(player_frame, folds)
    assert audit["shared_groups"].eq(0).all()
    assert sum(len(validation) for _, validation in folds) == len(player_frame)


def test_grouped_holdout_removes_row_split_leakage(player_frame):
    repeated = player_frame.loc[player_frame.index.repeat(20)].reset_index(drop=True)
    comparison = split_leakage_comparison(repeated)
    assert comparison.loc["Random row split", "shared_games"] > 0
    assert comparison.loc["Grouped gameId split", "shared_games"] == 0
