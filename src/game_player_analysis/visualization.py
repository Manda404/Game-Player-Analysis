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
    axis.set(title="Target distribution", xlabel=TARGET, ylabel="Players")
    return axis


def plot_top_correlations(train: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot the strongest monotonic associations with the target."""
    numeric = train.select_dtypes("number")
    correlations = (
        numeric.corr(method="spearman")[TARGET].drop(TARGET).sort_values(key=abs).tail(10)
    )
    axis = ax or plt.subplots(figsize=(9, 5))[1]
    correlations.plot.barh(color="#2878B5", ax=axis)
    axis.set(title="Top Spearman correlations", xlabel="Correlation")
    return axis


def plot_model_comparison(results: pd.DataFrame, ax: Axes | None = None) -> Axes:
    """Plot cross-validated MAE for the unique benchmark table."""
    axis = ax or plt.subplots(figsize=(9, 4.5))[1]
    ordered = results.sort_values("mae", ascending=False)
    sns.barplot(data=ordered, x="mae", y="model", color="#2878B5", ax=axis)
    axis.set(title="GroupKFold comparison", xlabel="MAE (lower is better)", ylabel="")
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
        title="Out-of-fold predictions",
        xlabel="Observed target",
        ylabel="Prediction",
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
        title="Official file periods",
        xlabel="Assigned month",
        ylabel="Player rows",
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
        title="One gameId assigned to multiple dates",
        xlabel="Within-gameId span (days)",
        ylabel="Density",
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
        title="Sentinels and anomalies",
        xlabel="Affected rows (%)",
        ylabel="",
    )
    profile.sort_values("zero_pct")["zero_pct"].plot.barh(color="#2878B5", ax=axes[1])
    axes[1].set(
        title="Zero concentration",
        xlabel="Zero values (%)",
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
    axes[0].set(title="Observed players per gameId", xlabel="Rows", ylabel="Matches")
    axes[1].set(title="Observed teams per gameId", xlabel="Teams", ylabel="Matches")
    axes[2].set(title="Observed players per team", xlabel="Rows", ylabel="Teams")
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
            title=f"Target by {feature} quantile",
            xlabel=f"{feature} (quantile median)",
            ylabel=f"Mean {TARGET}",
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
    axis.set_title("Focused Spearman correlations")
    return axis


def plot_validation_comparison(
    audit: pd.DataFrame,
    performance: pd.DataFrame,
) -> np.ndarray:
    """Compare model MAE and structural game leakage across split policies."""
    combined = audit.join(performance[["mae"]], how="inner")
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    combined["mae"].sort_values(ascending=False).plot.barh(color="#2878B5", ax=axes[0])
    axes[0].set(title="Reference-model MAE", xlabel="MAE", ylabel="")
    combined["validation_rows_from_seen_games_pct"].sort_values().plot.barh(
        color="#D9534F", ax=axes[1]
    )
    axes[1].set(
        title="Validation rows from seen gameIds",
        xlabel="Rows (%)",
        ylabel="",
        xlim=(0, 100),
    )
    figure.tight_layout()
    return axes


def plot_drift_diagnostics(
    numeric_shift: pd.DataFrame,
    categorical_shift: pd.DataFrame,
    adversarial_validation: pd.Series,
) -> np.ndarray:
    """Visualize univariate and multivariate train/test drift effect sizes."""
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    numeric_shift.nlargest(10, "psi")["psi"].sort_values().plot.barh(color="#2878B5", ax=axes[0, 0])
    axes[0, 0].axvline(0.1, linestyle="--", color="#D9534F", label="PSI guide 0.1")
    axes[0, 0].set(title="PSI for final features", xlabel="PSI", ylabel="")
    axes[0, 0].legend()

    numeric_shift.nlargest(10, "ks_statistic")["ks_statistic"].sort_values().plot.barh(
        color="#55A868", ax=axes[0, 1]
    )
    axes[0, 1].set(title="Train/test KS distance", xlabel="KS statistic", ylabel="")

    categorical_shift["total_variation_distance"].sort_values().plot.barh(
        color="#D9822B", ax=axes[1, 0]
    )
    axes[1, 0].set(
        title="Categorical drift",
        xlabel="Total variation distance",
        ylabel="",
    )

    auc = float(adversarial_validation["roc_auc_mean"])
    auc_std = float(adversarial_validation["roc_auc_std"])
    axes[1, 1].barh(["Train/test classifier"], [auc], xerr=[auc_std], color="#8172B2")
    axes[1, 1].axvline(0.5, linestyle="--", color="black", label="chance")
    axes[1, 1].set(
        title="Multivariate separability",
        xlabel="Out-of-fold ROC AUC",
        ylabel="",
        xlim=(0.45, max(0.65, auc + 0.05)),
    )
    axes[1, 1].legend()
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
        title="Progressive feature-family ablation",
        xlabel="Stage",
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
    axes[0].set(title="Mean MAE", xlabel="MAE", ylabel="")

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
    axes[1].set(title="Fold stability", xlabel="Fold", ylabel="MAE")
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
        title="Validation − train gap",
        xlabel="MAE gap",
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
        title="Bounded hyperparameter tuning",
        xlabel="Trial",
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
    axes[0, 0].set(title="Residual distribution", xlabel="Prediction − target")

    axes[0, 1].hexbin(frame[TARGET], values, gridsize=35, cmap="Blues", mincnt=1)
    axes[0, 1].plot([0, 1], [0, 1], "--", color="#D9534F")
    axes[0, 1].set(
        title="Prediction versus target",
        xlabel="Target",
        ylabel="Prediction",
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
        title="Error by target level",
        xlabel="Mean target in decile",
        ylabel="MAE",
    )

    modes = subgroup_errors.loc[subgroup_errors["dimension"].eq("mode_family")].sort_values(
        "mae", ascending=False
    )
    sns.barplot(data=modes, x="mae", y="subgroup", color="#D9822B", ax=axes[1, 1])
    axes[1, 1].set(title="Error by mode family", xlabel="MAE", ylabel="")
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
        title="Permutation importance on grouped holdout",
        xlabel="MAE increase after permutation",
        ylabel="",
    )
    return axis


def plot_fold_count_study(
    sensitivity: pd.DataFrame,
    decision: pd.DataFrame,
) -> np.ndarray:
    """Visualize grouped-CV stability and computational cost across fold counts."""
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    palette = {"CatBoost": "#2878B5", "XGBoost": "#D9822B"}
    for model, frame in sensitivity.groupby("model", sort=False):
        ordered = frame.sort_values("n_splits")
        axes[0].errorbar(
            ordered["n_splits"],
            ordered["mae"],
            yerr=ordered["mae_std"],
            marker="o",
            capsize=3,
            linewidth=2,
            color=palette.get(model, "#555555"),
            label=model,
        )
    axes[0].axvline(5, color="#555555", linestyle="--", linewidth=1)
    axes[0].annotate(
        "selected: 5 folds",
        xy=(5, sensitivity["mae"].min()),
        xytext=(5.35, sensitivity["mae"].min() + 0.00045),
        color="#555555",
    )
    axes[0].set(
        title="Model stability by K",
        xlabel="Number of GroupKFold folds",
        ylabel="Mean MAE ± fold standard deviation",
        xticks=sorted(sensitivity["n_splits"].unique()),
    )
    axes[0].legend(frameon=False)

    ordered_decision = decision.sort_values("n_splits")
    bars = axes[1].bar(
        ordered_decision["n_splits"].astype(str),
        ordered_decision["fit_cost_relative_to_5_folds"],
        color=["#9ABED8", "#2878B5", "#D9822B", "#C94A3D"],
    )
    axes[1].bar_label(
        bars, labels=[f"{value:.2f}×" for value in ordered_decision["fit_cost_relative_to_5_folds"]]
    )
    axes[1].set(
        title="Relative protocol cost",
        xlabel="Number of GroupKFold folds",
        ylabel="Training time relative to 5 folds",
    )
    secondary = axes[1].twinx()
    secondary.plot(
        ordered_decision["n_splits"].astype(str),
        ordered_decision["mean_validation_rows"],
        color="#555555",
        marker="o",
        linewidth=1.5,
    )
    secondary.set_ylabel("Validation rows per fold", color="#555555")
    secondary.tick_params(axis="y", colors="#555555")
    figure.tight_layout()
    return axes


def plot_shap_summary(
    shap_importance: pd.DataFrame,
    shap_values: pd.DataFrame,
    *,
    top_n: int = 10,
) -> np.ndarray:
    """Plot global CatBoost TreeSHAP magnitude and signed local contributions."""
    selected = shap_importance.head(top_n).copy()
    features = selected["feature"].tolist()
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.5]})

    ordered = selected.sort_values("mean_abs_shap")
    axes[0].barh(ordered["feature"], ordered["mean_abs_shap"], color="#2878B5", alpha=0.9)
    axes[0].set(
        title="Global SHAP importance",
        xlabel="Mean |SHAP value|",
        ylabel="",
    )

    color_artist = None
    generator = np.random.default_rng(42)
    for position, feature in enumerate(reversed(features)):
        values = shap_values.loc[shap_values["feature"].eq(feature)]
        jitter = generator.uniform(-0.28, 0.28, len(values))
        lower, upper = values["feature_value"].quantile([0.05, 0.95])
        if lower == upper:
            normalized_value = np.full(len(values), 0.5)
        else:
            normalized_value = np.clip(
                (values["feature_value"] - lower) / (upper - lower),
                0.0,
                1.0,
            )
        color_artist = axes[1].scatter(
            values["shap_value"],
            position + jitter,
            c=normalized_value,
            cmap="coolwarm",
            vmin=0.0,
            vmax=1.0,
            s=10,
            alpha=0.55,
            linewidths=0,
        )
    axes[1].axvline(0, color="#555555", linewidth=0.8)
    axes[1].set(
        title="Local contributions on the holdout",
        xlabel="SHAP value: contribution to predicted ranking",
        ylabel="",
        yticks=np.arange(len(features)),
        yticklabels=list(reversed(features)),
    )
    if color_artist is not None:
        colorbar = figure.colorbar(color_artist, ax=axes[1], pad=0.02)
        colorbar.set_label("Feature value")
        colorbar.set_ticks([0.0, 1.0], labels=["low", "high"])
    figure.tight_layout()
    return axes
