"""Tests for match-grouped validation."""

from game_player_analysis.validation import (
    audit_group_folds,
    audit_holdout_splits,
    make_group_folds,
    make_holdout_splits,
    safe_game_groups,
    split_leakage_comparison,
)


def test_repeated_invalid_game_ids_stay_in_one_conservative_group(player_frame):
    duplicated = player_frame.iloc[[3, 3]].reset_index(drop=True)
    groups = safe_game_groups(duplicated)
    assert groups.nunique() == 1


def test_gameid_never_crosses_grouped_folds(player_frame):
    folds = make_group_folds(player_frame, n_splits=2)
    audit = audit_group_folds(player_frame, folds)
    assert audit["shared_groups"].eq(0).all()
    assert audit["shared_raw_game_ids"].eq(0).all()
    assert sum(len(validation) for _, validation in folds) == len(player_frame)


def test_grouped_holdout_removes_row_split_leakage(player_frame):
    repeated = player_frame.loc[player_frame.index.repeat(20)].reset_index(drop=True)
    comparison = split_leakage_comparison(repeated)
    assert comparison.loc["Random row split", "shared_games"] > 0
    assert comparison.loc["Grouped gameId split", "shared_games"] == 0


def test_purged_pseudo_temporal_split_removes_seen_games(player_frame):
    repeated = player_frame.loc[player_frame.index.repeat(6)].reset_index(drop=True)
    april_mask = repeated["gameId"].eq("aaaaaaaaaaaaaa") & repeated.index.to_series().mod(2).eq(0)
    repeated.loc[april_mask, "date"] = "2024-04-15"
    repeated.loc[~april_mask, "date"] = "2024-02-15"

    audit = audit_holdout_splits(repeated, make_holdout_splits(repeated))
    assert audit.loc["Naive Jan-Mar → Apr", "shared_groups"] > 0
    assert audit.loc["Purged Jan-Mar → Apr", "shared_groups"] == 0
