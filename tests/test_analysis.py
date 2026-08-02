"""Tests for decision-oriented descriptive analysis."""

import numpy as np

from game_player_analysis.analysis import (
    add_analysis_kpis,
    data_quality_table,
    feature_target_profiles,
    kpi_evaluation_table,
    sampling_coverage_table,
)


def test_analysis_kpis_have_explicit_zero_and_undefined_rules(player_frame):
    enriched = add_analysis_kpis(player_frame)
    assert enriched.loc[0, "mobility_per_match_second"] == 0
    assert np.isnan(enriched.loc[0, "headshot_ratio"])
    assert np.isfinite(enriched["damage_per_kill_kpi"]).all()


def test_quality_coverage_and_kpi_tables_are_complete(player_frame):
    test = player_frame.drop(columns="winRankPercentage")
    quality = data_quality_table(player_frame, test)
    coverage = sampling_coverage_table(player_frame, test)
    kpis = kpi_evaluation_table(player_frame)
    profiles = feature_target_profiles(player_frame, ["walkDist"], bins=2)

    assert set(quality["dataset"]) == {"train", "test"}
    assert set(coverage.index) == {"train", "test"}
    assert "headshot_ratio" in kpis.index
    assert profiles["rows"].sum() == len(player_frame)
