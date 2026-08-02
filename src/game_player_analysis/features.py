"""Single source of truth for model feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from game_player_analysis.config import BEHAVIOR_FEATURES, POST_MATCH_FEATURES
from game_player_analysis.data import game_mode_family

FEATURE_SOURCES = (
    "walkDist",
    "rideDist",
    "damages",
    "kills",
    "weapons",
    "upgrades",
    "heals",
    "maxRank",
    "gameTime",
    "assists",
    "knocks",
    "gameType",
)

FEATURE_RATIONALE = {
    "walk_distance_per_match_minute": (
        "walkDist / (gameTime / 60)",
        "Normalise la mobilité par la durée globale du match, "
        "sans prétendre mesurer le temps de survie.",
    ),
    "damage_per_kill": (
        "damages / kills si kills > 0, sinon 0",
        "Distingue dégâts et conversion en éliminations; damages brut reste présent.",
    ),
    "combat_activity": (
        "kills + assists + knocks",
        "Résume le volume d'engagement offensif.",
    ),
    "resource_activity": (
        "weapons + upgrades + heals",
        "Résume collecte et progression sans faux rythme individuel.",
    ),
    "mode_solo / mode_duo / mode_squad": (
        "famille extraite de gameType",
        "Encode le contexte principal; les modes spéciaux forment la référence.",
    ),
}


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide by positive denominators and return finite zeros otherwise."""
    values = numerator.div(denominator.where(denominator.gt(0))).fillna(0.0)
    return values.replace([np.inf, -np.inf], 0.0)


def build_model_features(
    frame: pd.DataFrame,
    *,
    include_kill_rank: bool = False,
) -> pd.DataFrame:
    """Build the compact deterministic feature matrix used by every model."""
    required = set(FEATURE_SOURCES)
    if include_kill_rank:
        required.add("killRank")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cannot build model features; missing: {sorted(missing)}")

    features = frame.loc[
        :,
        [
            "walkDist",
            "rideDist",
            "damages",
            "kills",
            "weapons",
            "upgrades",
            "heals",
            "maxRank",
        ],
    ].copy()
    minutes = frame["gameTime"] / 60.0
    features["walk_distance_per_match_minute"] = _safe_ratio(frame["walkDist"], minutes)
    features["damage_per_kill"] = _safe_ratio(frame["damages"], frame["kills"])
    features["combat_activity"] = frame["kills"] + frame["assists"] + frame["knocks"]
    features["resource_activity"] = frame["weapons"] + frame["upgrades"] + frame["heals"]
    family = game_mode_family(frame["gameType"])
    features["mode_solo"] = family.eq("solo").astype("int8")
    features["mode_duo"] = family.eq("duo").astype("int8")
    features["mode_squad"] = family.eq("squad").astype("int8")

    if include_kill_rank:
        features.insert(0, "killRank", frame["killRank"])

    expected = POST_MATCH_FEATURES if include_kill_rank else BEHAVIOR_FEATURES
    features = features.loc[:, list(expected)]
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Feature engineering produced a non-finite value")
    return features


def build_train_test_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    include_kill_rank: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply exactly the same feature contract to train and test."""
    train_features = build_model_features(train, include_kill_rank=include_kill_rank)
    test_features = build_model_features(test, include_kill_rank=include_kill_rank)
    if list(train_features.columns) != list(test_features.columns):
        raise RuntimeError("Train and test feature contracts differ")
    return train_features, test_features
