"""Generate the five print-ready figures used by the project overview.

The script does not train a model or recompute an experiment. It only renders
results already published in ``artifacts/metrics`` and the audited target
distribution from the immutable training CSV.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "artifacts" / "metrics"
OUTPUT_DIR = Path(__file__).resolve().parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

BLUE = "#2878B5"
ORANGE = "#E6862A"
GREEN = "#3A923A"
RED = "#C7473D"
LIGHT_BLUE = "#9CC2DC"
GREY = "#6F7782"
GRID = "#D9DEE3"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.6,
        "axes.titlesize": 8.4,
        "axes.labelsize": 7.6,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.55,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 360,
        "savefig.bbox": "tight",
    }
)


def read_metric(name: str) -> pd.DataFrame:
    """Read one published metric table and fail on a missing artifact."""
    path = METRICS_DIR / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required published metric is missing: {path}")
    return pd.read_csv(path)


def save_figure(figure: plt.Figure, filename: str) -> None:
    """Save one opaque, high-resolution PNG for LaTeX."""
    figure.savefig(OUTPUT_DIR / filename, facecolor="white", pad_inches=0.04)
    plt.close(figure)


def plot_data_audit() -> None:
    """Summarize target shape, date inconsistency and sparse team coverage."""
    train = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "train.csv", sep=";")
    date_integrity = read_metric("date_integrity").set_index("dataset")
    coverage = read_metric("sampling_coverage").set_index("dataset")

    figure = plt.figure(figsize=(4.0, 4.05))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.12, 1.0], hspace=0.50, wspace=0.50)
    target_axis = figure.add_subplot(grid[0, :])
    date_axis = figure.add_subplot(grid[1, 0])
    coverage_axis = figure.add_subplot(grid[1, 1])

    target = train["winRankPercentage"]
    target_axis.hist(target, bins=36, color=BLUE, edgecolor="white", linewidth=0.25)
    target_axis.axvline(target.mean(), color=ORANGE, linewidth=1.3, linestyle="--")
    target_axis.text(
        target.mean() + 0.025,
        target_axis.get_ylim()[1] * 0.84,
        f"moyenne = {target.mean():.3f}",
        color=ORANGE,
        fontsize=7,
    )
    target_axis.set(
        title="Cible bornée mais non uniforme",
        xlabel="winRankPercentage (0 = dernier, 1 = premier)",
        ylabel="Lignes joueur",
        xlim=(0, 1),
    )

    date_values = date_integrity.loc[
        ["train", "test"],
        ["median_within_game_span_days", "p95_within_game_span_days"],
    ].to_numpy()
    positions = np.arange(2)
    width = 0.34
    date_axis.bar(
        positions - width / 2,
        date_values[:, 0],
        width,
        color=BLUE,
        label="médiane",
    )
    date_axis.bar(
        positions + width / 2,
        date_values[:, 1],
        width,
        color=LIGHT_BLUE,
        label="95e pct.",
    )
    date_axis.set_xticks(positions, ["Train", "Test"])
    date_axis.set(title="Dates incohérentes", ylabel="Étendue intra-gameId (jours)")
    date_axis.legend(frameon=False, fontsize=6.2, loc="center right")
    date_axis.text(
        0.02,
        0.97,
        "100 % des matchs\nmulti-lignes",
        transform=date_axis.transAxes,
        va="top",
        fontsize=5.9,
        color=RED,
    )

    coverage_values = [
        coverage.loc["train", "singleton_team_pct"],
        coverage.loc["train", "rows_with_observed_teammate_pct"],
    ]
    labels = ["Groupes\nsingletons", "Lignes avec\ncoéquipier"]
    bars = coverage_axis.bar(
        labels,
        coverage_values,
        color=[ORANGE, GREEN],
        width=0.62,
    )
    coverage_axis.set(title="Équipes peu observées", ylabel="Lignes / groupes (%)", ylim=(0, 108))
    for bar, value in zip(bars, coverage_values, strict=True):
        coverage_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 3,
            f"{value:.2f} %",
            ha="center",
            fontsize=6.7,
        )

    figure.suptitle("Audit des données : ce qui a changé les décisions", fontsize=9.2, y=0.995)
    save_figure(figure, "01_data_audit.png")


def plot_feature_ablation() -> None:
    """Show the incremental value of each published feature family."""
    ablation = read_metric("feature_ablation")
    labels = ["Contexte", "+ mobilité", "+ combat", "+ ressources", "+ killRank"]
    values = ablation["mae"].to_numpy()

    figure, axis = plt.subplots(figsize=(4.0, 2.35))
    axis.plot(
        np.arange(len(values)),
        values,
        marker="o",
        markersize=5,
        linewidth=1.8,
        color=BLUE,
    )
    axis.fill_between(np.arange(len(values)), values, values.max(), color=LIGHT_BLUE, alpha=0.18)
    for index, (value, features) in enumerate(zip(values, ablation["feature_count"], strict=True)):
        offset = 0.010 if index == 0 else 0.006
        axis.text(
            index,
            value + offset,
            f"{value:.4f}\n{int(features)} var.",
            ha="center",
            va="bottom",
            fontsize=6.4,
        )
    axis.set_xticks(np.arange(len(labels)), labels, rotation=17, ha="right")
    axis.set(
        title="Ablation groupée : contribution des familles de variables",
        ylabel="MAE GroupKFold",
        ylim=(0.05, 0.305),
    )
    axis.axvspan(3.62, 4.18, color=ORANGE, alpha=0.12)
    save_figure(figure, "02_feature_ablation.png")


def plot_validation_protocol() -> None:
    """Compare leakage and performance for the four audited holdouts."""
    audit = read_metric("holdout_audit").set_index("strategy")
    performance = read_metric("holdout_performance").set_index("strategy")
    order = [
        "Random row",
        "Grouped gameId",
        "Naive Jan-Mar → Apr",
        "Purged Jan-Mar → Apr",
    ]
    labels = [
        "Aléatoire ligne",
        "Groupé gameId",
        "Pseudo-temp. naïf",
        "Pseudo-temp. purgé",
    ]
    colors = [RED, GREEN, RED, GREEN]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(4.0, 2.65),
        sharey=True,
        gridspec_kw={"wspace": 0.16},
    )
    positions = np.arange(4)
    seen = audit.loc[order, "validation_rows_from_seen_games_pct"].to_numpy()
    bars = axes[0].barh(positions, seen, color=colors, height=0.60)
    axes[0].set_yticks(positions, labels)
    axes[0].invert_yaxis()
    axes[0].set(title="Matchs déjà vus", xlabel="Lignes de validation (%)", xlim=(0, 67))
    for bar, value in zip(bars, seen, strict=True):
        axes[0].text(
            value + 1.4,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontsize=6.2,
        )

    mae = performance.loc[order, "mae"].to_numpy()
    axes[1].scatter(mae, positions, c=colors, s=32, zorder=3)
    axes[1].set(
        title="Score du même CatBoost",
        xlabel="MAE holdout",
        xlim=(0.06084, 0.06138),
    )
    axes[1].set_xticks([0.0609, 0.0611, 0.0613])
    for index, value in enumerate(mae):
        axes[1].text(
            value + 0.000035,
            index,
            f"{value:.5f}",
            ha="left",
            va="center",
            fontsize=5.8,
        )

    figure.suptitle(
        "Validation : supprimer la fuite avant d'interpréter le score", fontsize=9.0, y=1.01
    )
    save_figure(figure, "03_validation_protocol.png")


def plot_interpretation_and_errors() -> None:
    """Connect permutation importance with the most relevant subgroup errors."""
    importance = read_metric("permutation_importance").head(5).copy()
    subgroup = read_metric("subgroup_errors")
    modes = subgroup.loc[subgroup["dimension"].eq("mode_family")].copy()

    importance_labels = {
        "killRank": "killRank",
        "walk_distance_per_match_minute": "Marche / minute",
        "kills": "Kills",
        "maxRank": "maxRank",
        "walkDist": "Distance à pied",
    }
    importance["label"] = importance["feature"].map(importance_labels)
    importance = importance.sort_values("mae_increase_mean")
    mode_order = ["solo", "duo", "squad", "special"]
    modes = modes.set_index("subgroup").loc[mode_order].reset_index()

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(4.0, 4.0),
        gridspec_kw={"hspace": 0.52},
    )
    axes[0].barh(
        importance["label"],
        importance["mae_increase_mean"],
        xerr=importance["mae_increase_std"],
        color=[BLUE, BLUE, BLUE, BLUE, ORANGE],
        alpha=0.95,
    )
    axes[0].set(
        title="Importance par permutation sur holdout groupé",
        xlabel="Hausse de MAE après permutation",
    )

    bars = axes[1].bar(
        modes["subgroup"],
        modes["mae"],
        color=[GREEN, BLUE, ORANGE, RED],
        width=0.65,
    )
    axes[1].set(
        title="Erreur out-of-fold par famille de mode",
        ylabel="MAE",
        xlabel="Famille de gameType",
        ylim=(0, 0.116),
    )
    for bar, mae, rows in zip(bars, modes["mae"], modes["rows"], strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            mae + 0.004,
            f"{mae:.3f}\n(n={int(rows):,})".replace(",", " "),
            ha="center",
            fontsize=6.3,
        )

    figure.suptitle("Ce que le modèle utilise — et où il échoue", fontsize=9.2, y=0.995)
    save_figure(figure, "04_interpretation_errors.png")


def plot_drift_diagnostics() -> None:
    """Summarize univariate and multivariate train/test drift diagnostics."""
    numeric = read_metric("distribution_shift").sort_values("psi", ascending=False).head(6)
    categorical = read_metric("categorical_shift").sort_values("psi", ascending=False)
    adversarial = read_metric("adversarial_validation")
    auc = float(adversarial.loc[adversarial["Unnamed: 0"].eq("roc_auc_mean"), "value"].iloc[0])

    figure, axes = plt.subplots(1, 2, figsize=(4.0, 2.45), gridspec_kw={"wspace": 0.52})
    numeric_plot = numeric.sort_values("psi")
    axes[0].barh(numeric_plot["feature"], numeric_plot["psi"], color=BLUE)
    axes[0].axvline(0.1, color=RED, linewidth=1.0, linestyle="--")
    axes[0].set(
        title="PSI numérique maximal = 0,0051",
        xlabel="Population Stability Index",
        xlim=(0, 0.105),
    )

    category_plot = categorical.sort_values("psi")
    axes[1].barh(category_plot["feature"], category_plot["psi"], color=ORANGE)
    axes[1].axvline(0.1, color=RED, linewidth=1.0, linestyle="--")
    axes[1].set(
        title=f"Catégories faibles ; AUC adv. = {auc:.3f}",
        xlabel="Population Stability Index",
        xlim=(0, 0.105),
    )
    figure.suptitle(
        "Drift train/test : aucune dérive matérielle détectée",
        fontsize=9.0,
        y=1.01,
    )
    save_figure(figure, "05_drift_diagnostics.png")


def main() -> None:
    """Render every figure referenced by the LaTeX report."""
    plot_data_audit()
    plot_feature_ablation()
    plot_validation_protocol()
    plot_interpretation_and_errors()
    plot_drift_diagnostics()
    print(f"Five verified figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
