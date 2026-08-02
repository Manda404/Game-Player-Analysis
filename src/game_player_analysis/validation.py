"""Leakage-safe validation grouped by match."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

from game_player_analysis.config import GROUP_COLUMN, N_SPLITS, RANDOM_STATE

FoldIndices = Sequence[tuple[np.ndarray, np.ndarray]]
VALID_GAME_ID = re.compile(r"^[0-9a-f]{14}$")


def safe_game_groups(frame: pd.DataFrame, column: str = GROUP_COLUMN) -> pd.Series:
    """Return game groups, isolating identifiers corrupted by spreadsheets."""
    if column not in frame:
        raise ValueError(f"Grouping column is missing: {column}")
    groups = frame[column].astype("string").copy()
    valid = groups.str.fullmatch(VALID_GAME_ID).fillna(False)
    groups.loc[~valid] = [f"invalid_game_row_{index}" for index in frame.index[~valid]]
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


def audit_group_folds(
    frame: pd.DataFrame,
    folds: FoldIndices,
    *,
    group_column: str = GROUP_COLUMN,
) -> pd.DataFrame:
    """Report fold sizes and verify that no match crosses a boundary."""
    groups = safe_game_groups(frame, group_column).reset_index(drop=True)
    rows = []
    for fold, (train_index, validation_index) in enumerate(folds, start=1):
        train_groups = set(groups.iloc[train_index])
        validation_groups = set(groups.iloc[validation_index])
        rows.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "validation_rows": len(validation_index),
                "shared_groups": len(train_groups.intersection(validation_groups)),
            }
        )
    return pd.DataFrame(rows)


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
        indices, test_size=test_size, random_state=random_state, shuffle=True
    )
    grouped_train, grouped_validation = next(
        GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state).split(
            frame, groups=groups
        )
    )

    rows = []
    for name, train_index, validation_index in (
        ("Random row split", random_train, random_validation),
        ("Grouped gameId split", grouped_train, grouped_validation),
    ):
        train_groups = set(groups.iloc[train_index])
        validation_groups = groups.iloc[validation_index]
        rows.append(
            {
                "strategy": name,
                "train_rows": len(train_index),
                "validation_rows": len(validation_index),
                "shared_games": len(train_groups.intersection(validation_groups)),
                "validation_rows_from_seen_games_pct": float(
                    100 * validation_groups.isin(train_groups).mean()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")
