"""Tests for documented ranking sentinels."""

import pandas as pd

from game_player_analysis.cleaning import clean_ranking_sentinels


def test_rankpts_minus_one_is_missing_without_losing_rows(player_frame):
    original = player_frame.copy(deep=True)
    cleaned = clean_ranking_sentinels(player_frame)

    assert len(cleaned) == len(player_frame)
    assert cleaned.loc[2, "rank_pts_missing"] == 1
    assert pd.isna(cleaned.loc[2, "rankPts_clean"])
    pd.testing.assert_frame_equal(player_frame, original)


def test_zero_kill_and_win_points_are_conditional(player_frame):
    cleaned = clean_ranking_sentinels(player_frame)

    assert cleaned.loc[1, "kill_pts_missing"] == 1
    assert cleaned.loc[1, "win_pts_missing"] == 1
    assert pd.isna(cleaned.loc[1, "killPts_clean"])
    assert cleaned.loc[2, "kill_pts_missing"] == 0
    assert cleaned.loc[2, "killPts_clean"] == 0
