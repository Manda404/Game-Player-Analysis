"""Shared fixtures for the compact analysis pipeline."""

import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
matplotlib.use("Agg")


@pytest.fixture
def player_frame() -> pd.DataFrame:
    """Small official-shaped frame with sentinel and zero-denominator rows."""
    rows = []
    game_ids = ["aaaaaaaaaaaaaa", "aaaaaaaaaaaaaa", "bbbbbbbbbbbbbb", "5,44E+13"]
    for index, game_id in enumerate(game_ids):
        rows.append(
            {
                "playerId": f"{index + 1:014x}",
                "teamId": f"{index + 101:014x}",
                "gameId": game_id,
                "assists": index % 2,
                "upgrades": index + 1,
                "damages": float(index * 100),
                "knocks": index,
                "headshots": min(index, 1),
                "heals": index + 2,
                "killRank": index + 1,
                "killPts": 0 if index in {1, 2} else 100,
                "kills": index,
                "killStreaks": min(index, 2),
                "highestKill": float(index * 10),
                "gameTime": 0 if index == 0 else 1200,
                "gameType": ["solo", "duo-fpp", "squad", "event"][index],
                "maxRank": 100,
                "numTeams": [100, 50, 25, 20][index],
                "rankPts": -1 if index == 2 else 1500,
                "revives": 0,
                "rideDist": float(index * 50),
                "roadKills": 0,
                "swimDist": 0.0,
                "teamKills": 0,
                "vehicleDestr": 0,
                "walkDist": float(index * 500),
                "weapons": index + 1,
                "winPts": 0 if index in {1, 2} else 100,
                "winRankPercentage": index / 3,
                "date": pd.Timestamp("2024-01-01") + pd.Timedelta(index, unit="D"),
            }
        )
    return pd.DataFrame(rows)
