"""Leakage-safe validation grouped by match."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

from game_player_analysis.config import (
    FINAL_HOLDOUT_RANDOM_STATE,
    GROUP_COLUMN,
    N_SPLITS,
    RANDOM_STATE,
)

FoldIndices = Sequence[tuple[np.ndarray, np.ndarray]]
VALID_GAME_ID = re.compile(r"^[0-9a-f]{14}$")


def safe_game_groups(frame: pd.DataFrame, column: str = GROUP_COLUMN) -> pd.Series:
    """Return conservative game groups, including corrupted identifiers.

    Valid hexadecimal IDs are preserved. Repeated malformed strings are kept
    in the same synthetic group so that a spreadsheet collision cannot cross
    train and validation boundaries. Only genuinely missing IDs are isolated
    row by row because they carry no recoverable grouping information.
    """
    if column not in frame:
        raise ValueError(f"Grouping column is missing: {column}")
    groups = frame[column].astype("string").copy()
    missing = groups.isna() | groups.str.strip().eq("")
    valid = groups.str.fullmatch(VALID_GAME_ID).fillna(False)
    malformed = ~valid & ~missing
    groups.loc[malformed] = "invalid_game_value::" + groups.loc[malformed]
    groups.loc[missing] = [f"missing_game_row_{index}" for index in frame.index[missing]]
    return groups


def make_group_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int = N_SPLITS,
    group_column: str = GROUP_COLUMN,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create the canonical GroupKFold partitions."""
    groups = safe_game_groups(frame, group_column)
    if groups.nunique() < n_splits:
        raise ValueError("The dataset has fewer game groups than requested folds")
    folds = list(GroupKFold(n_splits=n_splits).split(frame, groups=groups))
    audit = audit_group_folds(frame, folds, group_column=group_column)
    if audit["shared_groups"].ne(0).any():
        raise RuntimeError("Grouped validation leaks at least one game")
    return folds


def make_final_group_holdout(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = FINAL_HOLDOUT_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Freeze one match-disjoint holdout for the current audit cycle.

    The split is independent of the inner GroupKFold used for candidate
    comparison and tuning. It is not described as a historical virgin
    holdout because earlier project iterations already explored all labeled
    rows.
    """
    groups = safe_game_groups(frame).reset_index(drop=True)
    development, holdout = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        ).split(frame, groups=groups)
    )
    shared = set(groups.iloc[development]).intersection(groups.iloc[holdout])
    if shared:
        raise RuntimeError("Final grouped holdout leaks at least one game")
    return np.asarray(development, dtype=int), np.asarray(holdout, dtype=int)


def audit_group_folds(
    frame: pd.DataFrame,
    folds: FoldIndices,
    *,
    group_column: str = GROUP_COLUMN,
) -> pd.DataFrame:
    """Report fold sizes and verify that no match crosses a boundary."""
    groups = safe_game_groups(frame, group_column).reset_index(drop=True)
    raw_groups = frame[group_column].astype("string").reset_index(drop=True)
    valid_raw = raw_groups.str.fullmatch(VALID_GAME_ID).fillna(False)
    rows = []
    for fold, (train_index, validation_index) in enumerate(folds, start=1):
        train_groups = set(groups.iloc[train_index])
        validation_groups = set(groups.iloc[validation_index])
        raw_train_groups = set(raw_groups.iloc[train_index].dropna())
        raw_validation_groups = set(raw_groups.iloc[validation_index].dropna())
        rows.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "validation_rows": len(validation_index),
                "shared_groups": len(train_groups.intersection(validation_groups)),
                "shared_raw_game_ids": len(raw_train_groups.intersection(raw_validation_groups)),
                "invalid_train_rows": int((~valid_raw.iloc[train_index]).sum()),
                "invalid_validation_rows": int((~valid_raw.iloc[validation_index]).sum()),
            }
        )
    return pd.DataFrame(rows)


def make_holdout_splits(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    temporal_cutoff: str | pd.Timestamp = "2024-04-01",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build four diagnostic holdouts with explicit leakage trade-offs.

    The chronological splits are sensitivity analyses only. The official
    statement calls ``date`` the match date, but rows sharing a game ID carry
    inconsistent dates. The purged variant therefore removes from the earlier
    period every conservative game group observed in April.
    """
    required = {GROUP_COLUMN, "date"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cannot build holdouts; missing: {sorted(missing)}")

    indices = np.arange(len(frame))
    groups = safe_game_groups(frame).reset_index(drop=True)
    random_train, random_validation = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    grouped_train, grouped_validation = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        ).split(frame, groups=groups)
    )

    dates = pd.to_datetime(frame["date"], errors="raise")
    cutoff = pd.Timestamp(temporal_cutoff)
    chronological_train = indices[dates.lt(cutoff).to_numpy()]
    chronological_validation = indices[dates.ge(cutoff).to_numpy()]
    validation_groups = set(groups.iloc[chronological_validation])
    purged_train = np.asarray(
        [index for index in chronological_train if groups.iloc[index] not in validation_groups],
        dtype=int,
    )

    splits = {
        "Random row": (random_train, random_validation),
        "Grouped gameId": (grouped_train, grouped_validation),
        "Naive Jan-Mar → Apr": (
            chronological_train,
            chronological_validation,
        ),
        "Purged Jan-Mar → Apr": (purged_train, chronological_validation),
    }
    for name, (train_index, validation_index) in splits.items():
        if len(train_index) == 0 or len(validation_index) == 0:
            raise ValueError(f"Holdout '{name}' produced an empty partition")
    return splits


def audit_holdout_splits(
    frame: pd.DataFrame,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Describe leakage, date ranges and purge cost for diagnostic splits."""
    groups = safe_game_groups(frame).reset_index(drop=True)
    raw_groups = frame[GROUP_COLUMN].astype("string").reset_index(drop=True)
    rows = []
    for strategy, (train_index, validation_index) in splits.items():
        train_groups = set(groups.iloc[train_index])
        validation_groups = groups.iloc[validation_index]
        raw_train_groups = set(raw_groups.iloc[train_index].dropna())
        raw_validation_groups = set(raw_groups.iloc[validation_index].dropna())
        rows.append(
            {
                "strategy": strategy,
                "train_rows": len(train_index),
                "validation_rows": len(validation_index),
                "shared_groups": len(train_groups.intersection(validation_groups)),
                "shared_raw_game_ids": len(raw_train_groups.intersection(raw_validation_groups)),
                "validation_rows_from_seen_games_pct": float(
                    100 * validation_groups.isin(train_groups).mean()
                ),
                "train_date_min": frame["date"].iloc[train_index].min(),
                "train_date_max": frame["date"].iloc[train_index].max(),
                "validation_date_min": frame["date"].iloc[validation_index].min(),
                "validation_date_max": frame["date"].iloc[validation_index].max(),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def split_leakage_comparison(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compare row-random and match-grouped holdouts without fitting a model."""
    indices = np.arange(len(frame))
    groups = safe_game_groups(frame).reset_index(drop=True)
    random_train, random_validation = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    grouped_train, grouped_validation = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        ).split(frame, groups=groups)
    )
    return audit_holdout_splits(
        frame,
        {
            "Random row split": (random_train, random_validation),
            "Grouped gameId split": (grouped_train, grouped_validation),
        },
    ).rename(columns={"shared_groups": "shared_games"})
