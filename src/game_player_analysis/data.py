"""Load and validate the two official Game Player CSV files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from game_player_analysis.config import (
    ID_COLUMNS,
    RAW_REQUIRED_COLUMNS,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
)


class DataValidationError(ValueError):
    """Raised when an input does not satisfy the official data contract."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 fingerprint of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(frame: pd.DataFrame, *, require_target: bool) -> None:
    """Validate schema, target availability and essential value constraints."""
    missing = set(RAW_REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise DataValidationError(f"Missing official columns: {sorted(missing)}")

    has_target = TARGET in frame.columns
    if require_target and not has_target:
        raise DataValidationError(f"Training data must contain '{TARGET}'")
    if not require_target and has_target:
        raise DataValidationError(f"Test data must not contain '{TARGET}'")

    missing_ids = frame.loc[:, list(ID_COLUMNS)].isna().sum()
    if missing_ids.any():
        invalid = missing_ids[missing_ids.gt(0)].to_dict()
        raise DataValidationError(f"Identifier columns contain missing values: {invalid}")
    if frame["gameType"].isna().any() or frame["date"].isna().any():
        raise DataValidationError("gameType and date must not contain missing values")
    if require_target and not frame[TARGET].between(0, 1).all():
        raise DataValidationError(f"'{TARGET}' must be bounded in [0, 1]")

    non_negative = [
        column
        for column in RAW_REQUIRED_COLUMNS
        if column not in {*ID_COLUMNS, "gameType", "date", "rankPts"}
    ]
    if frame.loc[:, non_negative].lt(0).any().any():
        raise DataValidationError("Gameplay measurements must be non-negative")
    if not frame["rankPts"].ge(-1).all():
        raise DataValidationError("rankPts may be -1 (missing) but not lower")


def load_dataset(path: str | Path, *, require_target: bool) -> pd.DataFrame:
    """Read one semicolon-delimited file while preserving identifiers."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Data file not found: {source}")
    frame = pd.read_csv(
        source,
        sep=";",
        dtype={**{column: "string" for column in ID_COLUMNS}, "gameType": "string"},
        parse_dates=["date"],
    )
    validate_dataset(frame, require_target=require_target)
    return frame


def load_train_test(
    train_path: str | Path = TRAIN_PATH,
    test_path: str | Path = TEST_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official pair and reject direct match overlap."""
    train = load_dataset(train_path, require_target=True)
    test = load_dataset(test_path, require_target=False)
    shared_games = set(train["gameId"]).intersection(test["gameId"])
    if shared_games:
        raise DataValidationError(f"Train and test share {len(shared_games)} gameId value(s)")
    return train, test


def dataset_summary(frame: pd.DataFrame) -> pd.Series:
    """Return the compact structural checks used in the final notebook."""
    return pd.Series(
        {
            "rows": len(frame),
            "columns": len(frame.columns),
            "matches": frame["gameId"].nunique(),
            "game_modes": frame["gameType"].nunique(),
            "missing_cells": int(frame.isna().sum().sum()),
            "exact_duplicates": int(frame.duplicated().sum()),
            "date_min": frame["date"].min(),
            "date_max": frame["date"].max(),
            "has_target": TARGET in frame,
        }
    )


def game_mode_family(game_type: pd.Series) -> pd.Series:
    """Map detailed modes to solo, duo, squad or special."""
    family = game_type.astype("string").str.extract(r"(solo|duo|squad)", expand=False)
    return family.fillna("special")


def game_mode_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare the small set of interpretable player KPIs by mode family."""
    required = {"gameType", "gameId", TARGET, "kills", "walkDist", "maxRank"}
    missing = required.difference(frame.columns)
    if missing:
        raise DataValidationError(f"Cannot summarize game modes; missing: {sorted(missing)}")
    enriched = frame.assign(
        mode_family=game_mode_family(frame["gameType"]),
        is_winner=frame[TARGET].eq(1),
    )
    return enriched.groupby("mode_family", observed=True).agg(
        rows=("gameId", "size"),
        target_mean=(TARGET, "mean"),
        win_rate=("is_winner", "mean"),
        kills_mean=("kills", "mean"),
        walk_dist_mean=("walkDist", "mean"),
        max_rank_mean=("maxRank", "mean"),
    )


def match_structure_summary(frame: pd.DataFrame) -> pd.Series:
    """Summarize sparse coverage and within-game date inconsistencies."""
    rows_per_game = frame.groupby("gameId").size()
    rows_per_team = frame.groupby(["gameId", "teamId"]).size()
    multirow_games = rows_per_game[rows_per_game.gt(1)].index
    dates_per_game = frame.groupby("gameId")["date"].nunique()
    spans = (
        frame.groupby("gameId")["date"]
        .agg(lambda values: (values.max() - values.min()).days)
        .reindex(multirow_games)
    )
    return pd.Series(
        {
            "mean_observed_rows_per_game": rows_per_game.mean(),
            "max_observed_rows_per_game": rows_per_game.max(),
            "singleton_team_pct": 100 * rows_per_team.eq(1).mean(),
            "rows_with_observed_teammate_pct": 100
            * frame.set_index(["gameId", "teamId"])
            .index.isin(rows_per_team[rows_per_team.gt(1)].index)
            .mean(),
            "multirow_games_with_distinct_dates_pct": 100
            * dates_per_game.reindex(multirow_games).gt(1).mean(),
            "median_within_game_date_span_days": spans.median(),
        }
    )


def distribution_shift_summary(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Compare numeric distributions with a scale-free mean difference."""
    rows = []
    for column in columns:
        scale = float(reference[column].std(ddof=1))
        standardized_difference = (
            float(current[column].mean() - reference[column].mean()) / scale if scale > 0 else 0.0
        )
        rows.append(
            {
                "feature": column,
                "train_mean": reference[column].mean(),
                "test_mean": current[column].mean(),
                "standardized_mean_difference": standardized_difference,
            }
        )
    return (
        pd.DataFrame(rows)
        .set_index("feature")
        .sort_values("standardized_mean_difference", key=abs, ascending=False)
    )


def raw_data_fingerprints(
    train_path: str | Path = TRAIN_PATH,
    test_path: str | Path = TEST_PATH,
) -> dict[str, str]:
    """Fingerprint both immutable source files."""
    return {"train": sha256_file(train_path), "test": sha256_file(test_path)}
