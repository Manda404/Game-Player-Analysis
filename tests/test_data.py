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
    sha256_file,
    validate_dataset,
)


def test_train_and_test_schema_are_explicit(player_frame):
    validate_dataset(player_frame, require_target=True)
    validate_dataset(player_frame.drop(columns=TARGET), require_target=False)


def test_raw_data_fingerprints_protect_explicit_sources(tmp_path, player_frame):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    player_frame.to_csv(train_path, sep=";", index=False)
    player_frame.drop(columns=TARGET).to_csv(test_path, sep=";", index=False)

    assert raw_data_fingerprints(train_path, test_path) == {
        "train": sha256_file(train_path),
        "test": sha256_file(test_path),
    }


def test_uploaded_files_load_and_support_documented_summaries(tmp_path, player_frame):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    test_frame = player_frame.drop(columns=TARGET).copy()
    test_frame["gameId"] = [f"{index + 100:014x}" for index in range(len(test_frame))]
    player_frame.to_csv(train_path, sep=";", index=False)
    test_frame.to_csv(test_path, sep=";", index=False)

    train, test = load_train_test(train_path, test_path)
    summary = dataset_summary(train)
    structure = match_structure_summary(train)
    shift = distribution_shift_summary(train, test, ["walkDist", "kills"])
    modes = game_mode_summary(train)

    assert (len(train), len(test)) == (len(player_frame), len(test_frame))
    assert summary[["rows", "columns", "has_target"]].tolist() == [len(player_frame), 30, True]
    assert structure["max_observed_rows_per_game"] == 2
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
