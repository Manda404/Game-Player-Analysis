"""The four figures used by the final narrative notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from game_player_analysis.config import TARGET


def apply_style() -> None:
    """Apply one restrained plotting style."""
    sns.set_theme(style="whitegrid", context="notebook")


def plot_target_distribution(train: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot the bounded target distribution."""
    axis = ax or plt.subplots(figsize=(9, 4.5))[1]
    sns.histplot(train[TARGET], bins=50, color="#2878B5", ax=axis)
    axis.set(title="Distribution de la cible", xlabel=TARGET, ylabel="Joueurs")
    return axis


def plot_top_correlations(train: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot the strongest monotonic associations with the target."""
    numeric = train.select_dtypes("number")
    correlations = (
        numeric.corr(method="spearman")[TARGET].drop(TARGET).sort_values(key=abs).tail(10)
    )
    axis = ax or plt.subplots(figsize=(9, 5))[1]
    correlations.plot.barh(color="#2878B5", ax=axis)
    axis.set(title="Principales corrélations de Spearman", xlabel="Corrélation")
    return axis


def plot_model_comparison(results: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot cross-validated MAE for the unique benchmark table."""
    axis = ax or plt.subplots(figsize=(9, 4.5))[1]
    ordered = results.sort_values("mae", ascending=False)
    sns.barplot(data=ordered, x="mae", y="model", color="#2878B5", ax=axis)
    axis.set(title="Comparaison GroupKFold", xlabel="MAE (plus faible = mieux)", ylabel="")
    return axis


def plot_predictions(
    target: pd.Series,
    prediction: pd.Series,
    ax: Axes | None = None,
) -> Axes:
    """Plot out-of-fold predictions against observed rankings."""
    axis = ax or plt.subplots(figsize=(6, 6))[1]
    axis.scatter(target, prediction, alpha=0.12, s=9, color="#2878B5")
    axis.plot([0, 1], [0, 1], "--", color="#D9534F")
    axis.set(
        title="Prédictions out-of-fold",
        xlabel="Cible observée",
        ylabel="Prédiction",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    return axis
