"""Business-aware cleaning without row deletion or hidden imputation."""

from __future__ import annotations

import numpy as np
import pandas as pd

RANKING_COLUMNS = ("rankPts", "killPts", "winPts")


def clean_ranking_sentinels(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert documented ranking sentinels and preserve availability flags.

    `rankPts == -1` is missing. According to the official statement, zeros in
    `killPts` and `winPts` are missing only when `rankPts` is available.
    """
    missing = set(RANKING_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Cannot clean ranking scores; missing: {sorted(missing)}")

    cleaned = frame.copy()
    rank_missing = cleaned["rankPts"].eq(-1)
    kill_missing = ~rank_missing & cleaned["killPts"].eq(0)
    win_missing = ~rank_missing & cleaned["winPts"].eq(0)

    cleaned["rank_pts_missing"] = rank_missing.astype("int8")
    cleaned["kill_pts_missing"] = kill_missing.astype("int8")
    cleaned["win_pts_missing"] = win_missing.astype("int8")
    cleaned["rankPts_clean"] = cleaned["rankPts"].mask(rank_missing)
    cleaned["killPts_clean"] = cleaned["killPts"].mask(kill_missing)
    cleaned["winPts_clean"] = cleaned["winPts"].mask(win_missing)
    cleaned["ranking_system"] = np.select(
        [rank_missing, kill_missing & win_missing],
        ["legacy", "rank"],
        default="mixed",
    )
    return cleaned


def quality_issues(frame: pd.DataFrame) -> pd.Series:
    """Count the small set of anomalies that affect interpretation."""
    total_distance = frame[["walkDist", "rideDist", "swimDist"]].sum(axis=1)
    return pd.Series(
        {
            "exact_duplicates": int(frame.duplicated().sum()),
            "kills_without_damage": int((frame["kills"].gt(0) & frame["damages"].eq(0)).sum()),
            "combat_without_distance": int(
                (total_distance.eq(0) & (frame["kills"].gt(0) | frame["damages"].gt(0))).sum()
            ),
            "headshots_above_kills": int(frame["headshots"].gt(frame["kills"]).sum()),
            "max_rank_below_num_teams": int(frame["maxRank"].lt(frame["numTeams"]).sum()),
        }
    )
