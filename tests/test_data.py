"""Tests for the official data contract."""

import pytest

from game_player_analysis.config import TARGET
from game_player_analysis.data import (
    DataValidationError,
    categorical_shift_summary,
    dataset_summary,
    distribution_shift_summary,
    game_mode_summary,
    load_train_test,
    match_structure_summary,
    raw_data_fingerprints,
    validate_dataset,
)


def test_train_and_test_schema_are_explicit(player_frame):
    validate_dataset(player_frame, require_target=True)
    validate_dataset(player_frame.drop(columns=TARGET), require_target=False)


def test_raw_data_fingerprints_protect_sources():
    assert raw_data_fingerprints() == {
        "train": "66ab317bb5fcc0df0e248127a25159f1dff9c3b8b16058281ecd6107b067f69b",
        "test": "4dd388277253326c4155a8c87abac19fa4a70339675af9e4682affe4c1345956",
    }


def test_official_files_load_and_support_documented_summaries():
    train, test = load_train_test()
    summary = dataset_summary(train)
    structure = match_structure_summary(train)
    shift = distribution_shift_summary(train, test, ["walkDist", "kills"])
    modes = game_mode_summary(train)

    assert (len(train), len(test)) == (50_000, 5_000)
    assert summary[["rows", "columns", "has_target"]].tolist() == [50_000, 30, True]
    assert structure["max_observed_rows_per_game"] == 8
    assert set(shift.index) == {"walkDist", "kills"}
    assert modes["rows"].sum() == len(train)
    assert set(modes.index) == {"solo", "duo", "squad", "special"}
    assert {"psi", "ks_statistic", "wasserstein_over_train_std"}.issubset(shift.columns)


def test_categorical_shift_reports_effect_sizes(player_frame):
    reference = player_frame.copy()
    current = player_frame.copy()
    current.loc[0, "gameType"] = "new-mode"
    shift = categorical_shift_summary(reference, current, ["gameType"])
    assert shift.loc["gameType", "test_only_categories"] == 1
    assert shift.loc["gameType", "total_variation_distance"] > 0


def test_invalid_target_is_rejected(player_frame):
    invalid = player_frame.copy()
    invalid.loc[0, TARGET] = 2.0
    with pytest.raises(DataValidationError, match="bounded"):
        validate_dataset(invalid, require_target=True)
