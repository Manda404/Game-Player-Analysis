"""Load and validate the two official Game Player CSV files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
    """Compare numeric distributions with complementary drift diagnostics.

    Standardized mean difference only detects a location shift. PSI, the KS
    statistic, normalized Wasserstein distance and zero-rate changes also
    expose changes in shape, tails and point masses. P-values are reported for
    traceability but are not used as an automatic decision rule because they
    depend strongly on sample size and KS is approximate for discrete values.
    """
    rows = []
    for column in columns:
        reference_values = reference[column].dropna().astype(float)
        current_values = current[column].dropna().astype(float)
        if reference_values.empty or current_values.empty:
            raise ValueError(f"Cannot assess drift for empty feature: {column}")
        scale = float(reference_values.std(ddof=1))
        standardized_difference = (
            float(current_values.mean() - reference_values.mean()) / scale if scale > 0 else 0.0
        )
        ks_result = stats.ks_2samp(reference_values, current_values, method="auto")
        wasserstein = float(stats.wasserstein_distance(reference_values, current_values))
        psi = population_stability_index(reference_values, current_values)
        rows.append(
            {
                "feature": column,
                "train_mean": reference_values.mean(),
                "test_mean": current_values.mean(),
                "standardized_mean_difference": standardized_difference,
                "ks_statistic": float(ks_result.statistic),
                "ks_pvalue": float(ks_result.pvalue),
                "psi": psi,
                "wasserstein_over_train_std": wasserstein / scale if scale > 0 else 0.0,
                "train_zero_pct": float(100 * reference_values.eq(0).mean()),
                "test_zero_pct": float(100 * current_values.eq(0).mean()),
                "zero_pct_point_change": float(
                    100 * (current_values.eq(0).mean() - reference_values.eq(0).mean())
                ),
                "train_p95": float(reference_values.quantile(0.95)),
                "test_p95": float(current_values.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows).set_index("feature").sort_values("psi", ascending=False)


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    *,
    bins: int = 10,
) -> float:
    """Return quantile-bin PSI as a descriptive train/current diagnostic."""
    reference_values = reference.dropna()
    current_values = current.dropna()
    if reference_values.empty or current_values.empty:
        raise ValueError("PSI requires non-empty reference and current values")

    if pd.api.types.is_numeric_dtype(reference_values) and reference_values.nunique() > bins:
        edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        reference_counts = pd.cut(reference_values, edges, include_lowest=True).value_counts(
            sort=False
        )
        current_counts = pd.cut(current_values, edges, include_lowest=True).value_counts(sort=False)
    else:
        categories = reference_values.value_counts().index.union(
            current_values.value_counts().index
        )
        reference_counts = reference_values.value_counts().reindex(categories, fill_value=0)
        current_counts = current_values.value_counts().reindex(categories, fill_value=0)

    epsilon = 1e-6
    reference_pct = (reference_counts / reference_counts.sum()).clip(lower=epsilon)
    current_pct = (current_counts / current_counts.sum()).clip(lower=epsilon)
    return float(((current_pct - reference_pct) * np.log(current_pct / reference_pct)).sum())


def categorical_shift_detail(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Return category proportions and signed percentage-point changes."""
    rows: list[dict[str, object]] = []
    for column in columns:
        reference_values = reference[column].astype("string").fillna("<missing>")
        current_values = current[column].astype("string").fillna("<missing>")
        categories = sorted(set(reference_values).union(current_values))
        reference_pct = reference_values.value_counts(normalize=True).reindex(
            categories, fill_value=0.0
        )
        current_pct = current_values.value_counts(normalize=True).reindex(
            categories, fill_value=0.0
        )
        for category in categories:
            rows.append(
                {
                    "feature": column,
                    "category": category,
                    "train_pct": float(100 * reference_pct.loc[category]),
                    "test_pct": float(100 * current_pct.loc[category]),
                    "percentage_point_change": float(
                        100 * (current_pct.loc[category] - reference_pct.loc[category])
                    ),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["feature", "percentage_point_change"],
            key=lambda values: (values.abs() if pd.api.types.is_numeric_dtype(values) else values),
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


def categorical_shift_summary(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Summarize categorical drift with PSI and total variation distance."""
    detail = categorical_shift_detail(reference, current, columns)
    rows: list[dict[str, object]] = []
    for column in columns:
        feature_detail = detail.loc[detail["feature"].eq(column)].copy()
        largest = feature_detail.loc[feature_detail["percentage_point_change"].abs().idxmax()]
        reference_values = reference[column].astype("string").fillna("<missing>")
        current_values = current[column].astype("string").fillna("<missing>")
        rows.append(
            {
                "feature": column,
                "psi": population_stability_index(reference_values, current_values),
                "total_variation_distance": float(
                    0.5 * feature_detail["percentage_point_change"].abs().sum() / 100
                ),
                "largest_shift_category": largest["category"],
                "largest_percentage_point_change": float(largest["percentage_point_change"]),
                "train_unique": int(reference_values.nunique()),
                "test_unique": int(current_values.nunique()),
                "test_only_categories": int(len(set(current_values).difference(reference_values))),
            }
        )
    return pd.DataFrame(rows).set_index("feature").sort_values("psi", ascending=False)


def raw_data_fingerprints(
    train_path: str | Path = TRAIN_PATH,
    test_path: str | Path = TEST_PATH,
) -> dict[str, str]:
    """Fingerprint both immutable source files."""
    return {"train": sha256_file(train_path), "test": sha256_file(test_path)}
