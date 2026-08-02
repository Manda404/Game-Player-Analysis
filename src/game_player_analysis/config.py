"""Small, explicit configuration for the Game Player analysis."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = RAW_DATA_DIR / "train.csv"
TEST_PATH = RAW_DATA_DIR / "test.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
METRICS_DIR = ARTIFACT_DIR / "metrics"
FIGURES_DIR = ARTIFACT_DIR / "figures"
METADATA_DIR = ARTIFACT_DIR / "metadata"
LOG_DIR = ARTIFACT_DIR / "logs"

TARGET = "winRankPercentage"
ID_COLUMNS = ("playerId", "teamId", "gameId")
GROUP_COLUMN = "gameId"
RANDOM_STATE = 42
N_SPLITS = 5

RAW_REQUIRED_COLUMNS = (
    "playerId",
    "teamId",
    "gameId",
    "assists",
    "upgrades",
    "damages",
    "knocks",
    "headshots",
    "heals",
    "killRank",
    "killPts",
    "kills",
    "killStreaks",
    "highestKill",
    "gameTime",
    "gameType",
    "maxRank",
    "numTeams",
    "rankPts",
    "revives",
    "rideDist",
    "roadKills",
    "swimDist",
    "teamKills",
    "vehicleDestr",
    "walkDist",
    "weapons",
    "winPts",
    "date",
)

# The behavior contract excludes the final kill ranking. The post-match
# contract adds it explicitly so its incremental value remains visible.
BEHAVIOR_FEATURES = (
    "walkDist",
    "rideDist",
    "damages",
    "kills",
    "weapons",
    "upgrades",
    "heals",
    "maxRank",
    "walk_distance_per_match_minute",
    "damage_per_kill",
    "combat_activity",
    "resource_activity",
    "mode_solo",
    "mode_duo",
    "mode_squad",
)
POST_MATCH_FEATURES = ("killRank", *BEHAVIOR_FEATURES)
