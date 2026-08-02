"""Decision-oriented descriptive analyses used by the final notebook."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from game_player_analysis.cleaning import clean_ranking_sentinels, quality_issues
from game_player_analysis.config import TARGET
from game_player_analysis.validation import VALID_GAME_ID


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    undefined_as_nan: bool = False,
) -> pd.Series:
    values = numerator.div(denominator.where(denominator.gt(0)))
    values = values.replace([np.inf, -np.inf], np.nan)
    return values if undefined_as_nan else values.fillna(0.0)


def add_analysis_kpis(frame: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable analytical KPIs without changing model inputs."""
    enriched = frame.copy()
    enriched["total_distance"] = frame[["walkDist", "rideDist", "swimDist"]].sum(axis=1)
    enriched["mobility_per_match_second"] = _safe_ratio(
        enriched["total_distance"], frame["gameTime"]
    )
    enriched["damage_per_kill_kpi"] = _safe_ratio(frame["damages"], frame["kills"])
    enriched["headshot_ratio"] = _safe_ratio(
        frame["headshots"],
        frame["kills"],
        undefined_as_nan=True,
    )
    enriched["combat_activity_kpi"] = frame["kills"] + frame["assists"] + frame["knocks"]
    enriched["resource_activity_kpi"] = frame["weapons"] + frame["upgrades"] + frame["heals"]
    enriched["support_actions"] = frame["assists"] + frame["revives"]
    return enriched


def data_quality_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Return comparable counts for the quality issues that change decisions."""
    rows: list[dict[str, object]] = []
    for dataset, frame in (("train", train), ("test", test)):
        cleaned = clean_ranking_sentinels(frame)
        issues = quality_issues(frame)
        total_distance = frame[["walkDist", "rideDist", "swimDist"]].sum(axis=1)
        measures = {
            "missing_cells": int(frame.isna().sum().sum()),
            "exact_duplicates": int(frame.duplicated().sum()),
            "invalid_playerId": int(
                (
                    ~frame["playerId"].astype("string").str.fullmatch(VALID_GAME_ID).fillna(False)
                ).sum()
            ),
            "invalid_teamId": int(
                (~frame["teamId"].astype("string").str.fullmatch(VALID_GAME_ID).fillna(False)).sum()
            ),
            "invalid_gameId": int(
                (~frame["gameId"].astype("string").str.fullmatch(VALID_GAME_ID).fillna(False)).sum()
            ),
            "rankPts_-1": int(cleaned["rank_pts_missing"].sum()),
            "killPts_conditional_zero": int(cleaned["kill_pts_missing"].sum()),
            "winPts_conditional_zero": int(cleaned["win_pts_missing"].sum()),
            "kills_without_damage": int(issues["kills_without_damage"]),
            "combat_without_distance": int(
                (total_distance.eq(0) & (frame["kills"].gt(0) | frame["damages"].gt(0))).sum()
            ),
        }
        rows.extend(
            {
                "dataset": dataset,
                "check": check,
                "rows": value,
                "pct": 100 * value / len(frame),
            }
            for check, value in measures.items()
        )
    return pd.DataFrame(rows)


def date_integrity_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Quantify the conflict between official date semantics and game IDs."""
    rows = []
    for dataset, frame in (("train", train), ("test", test)):
        valid = frame["gameId"].astype("string").str.fullmatch(VALID_GAME_ID).fillna(False)
        valid_frame = frame.loc[valid]
        rows_per_game = valid_frame.groupby("gameId").size()
        multirow_games = rows_per_game[rows_per_game.gt(1)].index
        date_counts = valid_frame.groupby("gameId")["date"].nunique().reindex(multirow_games)
        spans = (
            valid_frame.groupby("gameId")["date"]
            .agg(lambda values: (values.max() - values.min()).total_seconds() / 86400)
            .reindex(multirow_games)
        )
        rows.append(
            {
                "dataset": dataset,
                "rows": len(frame),
                "valid_game_ids": int(valid.sum()),
                "games": int(valid_frame["gameId"].nunique()),
                "multirow_games": len(multirow_games),
                "multirow_games_with_multiple_dates_pct": float(100 * date_counts.gt(1).mean()),
                "median_within_game_span_days": float(spans.median()),
                "p95_within_game_span_days": float(spans.quantile(0.95)),
                "max_within_game_span_days": float(spans.max()),
                "date_min": frame["date"].min(),
                "date_max": frame["date"].max(),
                "unique_dates": int(frame["date"].nunique()),
            }
        )
    return pd.DataFrame(rows).set_index("dataset")


def sampling_coverage_table(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Show why observed team and lobby aggregates are not defensible."""
    rows = []
    for dataset, frame in (("train", train), ("test", test)):
        rows_per_game = frame.groupby("gameId").size()
        rows_per_team = frame.groupby(["gameId", "teamId"]).size()
        teammate_groups = rows_per_team[rows_per_team.gt(1)].index
        rows.append(
            {
                "dataset": dataset,
                "games": len(rows_per_game),
                "mean_observed_players_per_game": rows_per_game.mean(),
                "max_observed_players_per_game": rows_per_game.max(),
                "observed_game_teams": len(rows_per_team),
                "singleton_team_pct": 100 * rows_per_team.eq(1).mean(),
                "rows_with_observed_teammate_pct": 100
                * frame.set_index(["gameId", "teamId"]).index.isin(teammate_groups).mean(),
            }
        )
    return pd.DataFrame(rows).set_index("dataset")


def numeric_profile(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    """Profile zero concentration, tails and target association."""
    columns = list(columns)
    profile = pd.DataFrame(
        {
            "zero_pct": frame[columns].eq(0).mean() * 100,
            "mean": frame[columns].mean(),
            "median": frame[columns].median(),
            "p95": frame[columns].quantile(0.95),
            "p99": frame[columns].quantile(0.99),
            "max": frame[columns].max(),
            "spearman_target": frame[columns].corrwith(frame[TARGET], method="spearman"),
        }
    )
    return profile.sort_values("spearman_target", key=abs, ascending=False)


def kpi_evaluation_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate KPI distributions, target relation and source redundancy."""
    enriched = add_analysis_kpis(frame)
    specifications = {
        "total_distance": {
            "formula": "walkDist + rideDist + swimDist",
            "sources": ("walkDist", "rideDist", "swimDist"),
            "model_decision": "analytic_only_redundant",
        },
        "mobility_per_match_second": {
            "formula": "total_distance / gameTime",
            "sources": ("total_distance", "gameTime"),
            "model_decision": "analytic_only_match_time_not_survival",
        },
        "damage_per_kill_kpi": {
            "formula": "damages / kills; 0 when kills=0",
            "sources": ("damages", "kills"),
            "model_decision": "candidate_requires_ablation",
        },
        "headshot_ratio": {
            "formula": "headshots / kills; undefined when kills=0",
            "sources": ("headshots", "kills"),
            "model_decision": "analytic_only_sparse",
        },
        "combat_activity_kpi": {
            "formula": "kills + assists + knocks",
            "sources": ("kills", "assists", "knocks"),
            "model_decision": "candidate_requires_ablation",
        },
        "resource_activity_kpi": {
            "formula": "weapons + upgrades + heals",
            "sources": ("weapons", "upgrades", "heals"),
            "model_decision": "candidate_requires_ablation",
        },
        "support_actions": {
            "formula": "assists + revives",
            "sources": ("assists", "revives"),
            "model_decision": "analytic_only_mode_dependent",
        },
    }
    rows = []
    for kpi, specification in specifications.items():
        values = enriched[kpi]
        defined = values.notna()
        source_columns = [
            column for column in specification["sources"] if column in enriched.columns
        ]
        source_correlations = enriched.loc[defined, source_columns].corrwith(
            values.loc[defined],
            method="spearman",
        )
        most_redundant = (
            source_correlations.abs().idxmax() if not source_correlations.empty else None
        )
        rows.append(
            {
                "kpi": kpi,
                "formula": specification["formula"],
                "defined_pct": 100 * defined.mean(),
                "mean": values.mean(),
                "median": values.median(),
                "p95": values.quantile(0.95),
                "zero_pct_among_defined": 100 * values.loc[defined].eq(0).mean(),
                "spearman_target": values.corr(enriched[TARGET], method="spearman"),
                "most_redundant_source": most_redundant,
                "max_abs_source_spearman": (
                    source_correlations.abs().max() if not source_correlations.empty else np.nan
                ),
                "model_decision": specification["model_decision"],
            }
        )
    return pd.DataFrame(rows).set_index("kpi")


def feature_target_profiles(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """Summarize nonlinear feature-target relations with 95% mean CIs."""
    rows: list[dict[str, object]] = []
    for column in columns:
        valid = frame[[column, TARGET]].dropna()
        quantile = pd.qcut(valid[column], q=bins, duplicates="drop")
        grouped = valid.assign(_bin=quantile).groupby("_bin", observed=True)
        summary = grouped.agg(
            x_median=(column, "median"),
            target_mean=(TARGET, "mean"),
            target_median=(TARGET, "median"),
            target_std=(TARGET, "std"),
            rows=(TARGET, "size"),
        )
        summary["ci95"] = 1.96 * summary["target_std"].fillna(0) / np.sqrt(summary["rows"])
        for order, (_, result) in enumerate(summary.iterrows(), start=1):
            rows.append(
                {
                    "feature": column,
                    "quantile": order,
                    "x_median": result["x_median"],
                    "target_mean": result["target_mean"],
                    "target_median": result["target_median"],
                    "ci95": result["ci95"],
                    "rows": int(result["rows"]),
                }
            )
    return pd.DataFrame(rows)


def target_grid_summary(frame: pd.DataFrame) -> pd.Series:
    """Measure how closely the rounded target follows the maxRank grid."""
    span = frame["maxRank"] - 1
    implied_position = (1 - frame[TARGET]) * span
    distance_to_integer = (implied_position - implied_position.round()).abs()
    return pd.Series(
        {
            "target_mean": frame[TARGET].mean(),
            "target_median": frame[TARGET].median(),
            "target_std": frame[TARGET].std(),
            "target_zero_rows": int(frame[TARGET].eq(0).sum()),
            "target_one_rows": int(frame[TARGET].eq(1).sum()),
            "unique_target_values": frame[TARGET].nunique(),
            "grid_distance_median": distance_to_integer.median(),
            "grid_distance_p95": distance_to_integer.quantile(0.95),
            "grid_distance_max": distance_to_integer.max(),
            "rows_within_0.0001_of_grid": int(distance_to_integer.le(0.0001).sum()),
        }
    )


def adversarial_validation_summary(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    *,
    random_state: int = 42,
) -> pd.Series:
    """Measure multivariate train/test separability with out-of-fold ROC AUC.

    A linear classifier is intentionally used as a compact complement to the
    univariate drift table, not as proof that no nonlinear shift exists.
    """
    if list(train_features.columns) != list(test_features.columns):
        raise ValueError("Adversarial validation requires an identical feature contract")
    features = pd.concat([train_features, test_features], ignore_index=True)
    dataset_label = np.concatenate(
        [np.zeros(len(train_features), dtype=int), np.ones(len(test_features), dtype=int)]
    )
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            random_state=random_state,
        ),
    )
    n_splits = min(5, len(train_features), len(test_features))
    if n_splits < 2:
        raise ValueError("Adversarial validation requires at least two rows per dataset")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    result = cross_validate(
        estimator,
        features,
        dataset_label,
        cv=splitter,
        scoring="roc_auc",
        n_jobs=1,
    )
    scores = result["test_score"]
    return pd.Series(
        {
            "method": "standardized logistic adversarial validation",
            "folds": len(scores),
            "roc_auc_mean": float(scores.mean()),
            "roc_auc_std": float(scores.std(ddof=1)),
            "roc_auc_min": float(scores.min()),
            "roc_auc_max": float(scores.max()),
            "interpretation": (
                "weak multivariate separation" if scores.mean() < 0.60 else "material separation"
            ),
        }
    )
