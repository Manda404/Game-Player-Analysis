"""Focused figures used by the final narrative notebook and report."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
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


def plot_date_contradiction(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> np.ndarray:
    """Visualize official periods and impossible within-game date spans."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    month_rows = []
    for dataset, frame in (("train", train), ("test", test)):
        counts = frame["date"].dt.to_period("M").value_counts().sort_index()
        month_rows.extend(
            {"dataset": dataset, "month": str(month), "rows": rows}
            for month, rows in counts.items()
        )
    sns.barplot(
        data=pd.DataFrame(month_rows),
        x="month",
        y="rows",
        hue="dataset",
        ax=axes[0],
    )
    axes[0].set(
        title="Périodes officielles des fichiers",
        xlabel="Mois attribué",
        ylabel="Lignes joueur",
    )

    span_rows = []
    for dataset, frame in (("train", train), ("test", test)):
        spans = frame.groupby("gameId")["date"].agg(
            lambda values: (values.max() - values.min()).total_seconds() / 86400
        )
        span_rows.extend(
            {"dataset": dataset, "span_days": value} for value in spans.loc[spans.gt(0)]
        )
    sns.histplot(
        data=pd.DataFrame(span_rows),
        x="span_days",
        hue="dataset",
        bins=30,
        element="step",
        stat="density",
        common_norm=False,
        ax=axes[1],
    )
    axes[1].set(
        title="Même gameId réparti sur plusieurs dates",
        xlabel="Étendue intra-gameId (jours)",
        ylabel="Densité",
    )
    figure.tight_layout()
    return axes


def plot_quality_overview(
    quality: pd.DataFrame,
    profile: pd.DataFrame,
) -> np.ndarray:
    """Plot sentinel/anomaly prevalence and zero concentration."""
    figure, axes = plt.subplots(1, 2, figsize=(15, 5))
    selected_checks = [
        "invalid_gameId",
        "rankPts_-1",
        "killPts_conditional_zero",
        "kills_without_damage",
        "combat_without_distance",
    ]
    quality_plot = quality.loc[quality["check"].isin(selected_checks)]
    sns.barplot(
        data=quality_plot,
        x="pct",
        y="check",
        hue="dataset",
        ax=axes[0],
    )
    axes[0].set(
        title="Sentinelles et anomalies",
        xlabel="Lignes concernées (%)",
        ylabel="",
    )
    profile.sort_values("zero_pct")["zero_pct"].plot.barh(color="#2878B5", ax=axes[1])
    axes[1].set(
        title="Concentration des zéros",
        xlabel="Valeurs nulles (%)",
        ylabel="",
    )
    figure.tight_layout()
    return axes


def plot_sampling_structure(train: pd.DataFrame) -> np.ndarray:
    """Show the sparse observed match and team coverage."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    rows_per_game = train.groupby("gameId").size()
    rows_per_team = train.groupby(["gameId", "teamId"]).size()
    teams_per_game = train.groupby("gameId")["teamId"].nunique()
    sns.histplot(rows_per_game, discrete=True, ax=axes[0], color="#2878B5")
    sns.histplot(teams_per_game, discrete=True, ax=axes[1], color="#3A923A")
    sns.histplot(rows_per_team, discrete=True, ax=axes[2], color="#D9822B")
    axes[0].set(title="Joueurs observés par gameId", xlabel="Lignes", ylabel="Matchs")
    axes[1].set(title="Équipes observées par gameId", xlabel="Équipes", ylabel="Matchs")
    axes[2].set(title="Joueurs observés par équipe", xlabel="Lignes", ylabel="Équipes")
    for axis in axes:
        axis.set_yscale("log")
    figure.tight_layout()
    return axes


def plot_feature_target_profiles(
    profiles: pd.DataFrame,
    features: list[str] | tuple[str, ...],
) -> np.ndarray:
    """Plot target means and 95% confidence intervals by feature quantile."""
    ncols = 2
    nrows = int(np.ceil(len(features) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    flattened = np.atleast_1d(axes).ravel()
    for axis, feature in zip(flattened, features, strict=False):
        subset = profiles.loc[profiles["feature"].eq(feature)]
        axis.errorbar(
            subset["x_median"],
            subset["target_mean"],
            yerr=subset["ci95"],
            marker="o",
            linewidth=2,
            color="#2878B5",
            capsize=3,
        )
        axis.set(
            title=f"Cible par quantile de {feature}",
            xlabel=f"{feature} (médiane du quantile)",
            ylabel=f"Moyenne {TARGET}",
            ylim=(0, 1),
        )
    for axis in flattened[len(features) :]:
        axis.axis("off")
    figure.tight_layout()
    return axes


def plot_targeted_correlation_heatmap(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    ax: Axes | None = None,
) -> Axes:
    """Plot a readable Spearman matrix limited to decision-relevant fields."""
    axis = ax or plt.subplots(figsize=(10, 8))[1]
    correlation = frame.loc[:, list(columns)].corr(method="spearman")
    sns.heatmap(
        correlation,
        cmap="vlag",
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        ax=axis,
    )
    axis.set_title("Corrélations de Spearman ciblées")
    return axis


def plot_validation_comparison(
    audit: pd.DataFrame,
    performance: pd.DataFrame,
) -> np.ndarray:
    """Compare model MAE and structural game leakage across split policies."""
    combined = audit.join(performance[["mae"]], how="inner")
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    combined["mae"].sort_values(ascending=False).plot.barh(color="#2878B5", ax=axes[0])
    axes[0].set(title="MAE du modèle de référence", xlabel="MAE", ylabel="")
    combined["validation_rows_from_seen_games_pct"].sort_values().plot.barh(
        color="#D9534F", ax=axes[1]
    )
    axes[1].set(
        title="Lignes de validation issues de gameId vus",
        xlabel="Lignes (%)",
        ylabel="",
        xlim=(0, 100),
    )
    figure.tight_layout()
    return axes


def plot_feature_ablation(ablation: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot progressive MAE as feature families are added."""
    axis = ax or plt.subplots(figsize=(10, 5))[1]
    ordered = ablation.reset_index(drop=True)
    axis.plot(
        ordered["stage"],
        ordered["mae"],
        marker="o",
        linewidth=2,
        color="#2878B5",
    )
    for _, row in ordered.iterrows():
        axis.annotate(
            f"{row['mae']:.4f}\n{int(row['feature_count'])} f.",
            (row["stage"], row["mae"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    axis.set(
        title="Ablation progressive des familles de features",
        xlabel="Étape",
        ylabel="MAE GroupKFold",
    )
    axis.tick_params(axis="x", rotation=20)
    return axis


def plot_model_diagnostics(
    results: pd.DataFrame,
    fold_details: dict[str, pd.DataFrame],
) -> np.ndarray:
    """Plot validation metrics, fold stability and train-validation gaps."""
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    ordered = results.sort_values("mae", ascending=False)
    sns.barplot(data=ordered, x="mae", y="model", color="#2878B5", ax=axes[0])
    axes[0].set(title="MAE moyenne", xlabel="MAE", ylabel="")

    fold_rows = []
    for model, detail in fold_details.items():
        fold_rows.append(detail.assign(model=model))
    fold_frame = pd.concat(fold_rows, ignore_index=True)
    sns.pointplot(
        data=fold_frame,
        x="fold",
        y="mae",
        hue="model",
        ax=axes[1],
    )
    axes[1].set(title="Stabilité entre folds", xlabel="Fold", ylabel="MAE")
    axes[1].legend(fontsize=7)

    gap = results.assign(mae_gap=results["mae"] - results["train_mae"])
    sns.barplot(
        data=gap.sort_values("mae_gap", ascending=False),
        x="mae_gap",
        y="model",
        color="#D9822B",
        ax=axes[2],
    )
    axes[2].set(
        title="Écart validation − train",
        xlabel="Écart MAE",
        ylabel="",
    )
    figure.tight_layout()
    return axes


def plot_tuning_results(
    tuning: pd.DataFrame,
    baseline_mae: float,
    ax: Axes | None = None,
) -> Axes:
    """Plot bounded randomized-search trials against the frozen baseline."""
    axis = ax or plt.subplots(figsize=(9, 4.5))[1]
    ordered = tuning.sort_values("trial")
    axis.plot(ordered["trial"], ordered["mae"], marker="o", color="#2878B5")
    axis.axhline(baseline_mae, linestyle="--", color="#D9534F", label="baseline")
    axis.set(
        title="Hyperparameter tuning borné",
        xlabel="Essai",
        ylabel="MAE GroupKFold",
        xticks=ordered["trial"],
    )
    axis.legend()
    return axis


def plot_error_diagnostics(
    frame: pd.DataFrame,
    prediction: np.ndarray | pd.Series,
    subgroup_errors: pd.DataFrame,
) -> np.ndarray:
    """Plot residual shape, regression-to-mean and subgroup failures."""
    values = np.asarray(prediction, dtype=float)
    residual = values - frame[TARGET].to_numpy()
    absolute = np.abs(residual)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(residual, bins=60, kde=True, color="#2878B5", ax=axes[0, 0])
    axes[0, 0].axvline(0, linestyle="--", color="#D9534F")
    axes[0, 0].set(title="Distribution des résidus", xlabel="Prédiction − cible")

    axes[0, 1].hexbin(frame[TARGET], values, gridsize=35, cmap="Blues", mincnt=1)
    axes[0, 1].plot([0, 1], [0, 1], "--", color="#D9534F")
    axes[0, 1].set(
        title="Prédiction contre cible",
        xlabel="Cible",
        ylabel="Prédiction",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    target_bins = pd.qcut(frame[TARGET], q=10, duplicates="drop")
    error_by_target = (
        pd.DataFrame({"target": frame[TARGET], "absolute_error": absolute, "bin": target_bins})
        .groupby("bin", observed=True)
        .agg(target_mean=("target", "mean"), mae=("absolute_error", "mean"))
    )
    axes[1, 0].plot(
        error_by_target["target_mean"],
        error_by_target["mae"],
        marker="o",
        color="#2878B5",
    )
    axes[1, 0].set(
        title="Erreur selon le niveau de cible",
        xlabel="Cible moyenne du décile",
        ylabel="MAE",
    )

    modes = subgroup_errors.loc[subgroup_errors["dimension"].eq("mode_family")].sort_values(
        "mae", ascending=False
    )
    sns.barplot(data=modes, x="mae", y="subgroup", color="#D9822B", ax=axes[1, 1])
    axes[1, 1].set(title="Erreur par famille de mode", xlabel="MAE", ylabel="")
    figure.tight_layout()
    return axes


def plot_permutation_importance(
    importance: pd.DataFrame,
    *,
    top_n: int = 12,
    ax: Axes | None = None,
) -> Axes:
    """Plot holdout permutation importance as increase in MAE."""
    axis = ax or plt.subplots(figsize=(9, 5.5))[1]
    selected = importance.head(top_n).sort_values("mae_increase_mean")
    axis.barh(
        selected["feature"],
        selected["mae_increase_mean"],
        xerr=selected["mae_increase_std"],
        color="#2878B5",
        alpha=0.9,
    )
    axis.set(
        title="Importance par permutation sur holdout groupé",
        xlabel="Augmentation de MAE après permutation",
        ylabel="",
    )
    return axis
