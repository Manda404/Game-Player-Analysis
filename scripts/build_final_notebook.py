"""Build the narrative notebook from version-controlled cell sources."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "game_player_analysis.ipynb"


def markdown(source: str):
    """Create one stripped Markdown cell."""
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    """Create one stripped code cell."""
    return nbf.v4.new_code_cell(source.strip())


def build_notebook() -> None:
    """Write the complete, reproducible final notebook."""
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    notebook["cells"] = [
        markdown(
            """
# Game Player Analysis — Final Data Science Review

**Executed primary deliverable.** In this notebook, I predict
`winRankPercentage`, the player's team's normalized final ranking from 0 (last
place) to 1 (first place), using post-match statistics. Each row describes one
player, while the team-level target is repeated for the observed players from
that team.

I prioritize sound interpretation and traceability: every modeled result is
out-of-fold, I group by `gameId`, and I keep the official unlabeled May test
file outside every model-selection decision.
"""
        ),
        markdown(
            """
## 1. Official setting, metric, and questions

- **Train:** 50,000 rows assigned to January–April 2024.
- **Test:** 5,000 rows assigned to May 2024.
- **Primary metric:** MAE, complemented by RMSE and R².
- **Model question:** what post-match accuracy do I obtain from individual
  variables, and how much comes from `killRank`?
- **Product question:** do mobility, combat, and resource behaviours remain
  informative without the post-match ranking?

The brief states that `date` is the match date. I test that definition rather
than assume it is true. `maxRank` defines the reported ranking grid; it is
neither the actual player count nor the number of observed rows per match.
"""
        ),
        code(
            """
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/game-player-analysis-matplotlib")

import pandas as pd
from IPython.display import Image, Markdown, display

from game_player_analysis.inference import predict_frame
from game_player_analysis.logging_config import configure_logging
from game_player_analysis.pipeline import run_final_analysis

pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda value: f"{value:,.6f}")
configure_logging()
"""
        ),
        markdown(
            """
## 2. Reproducible execution and reviewed artifacts

By default, I load the reviewed published artifacts so this notebook stays fast
to inspect and its figures remain visible without re-fitting CatBoost. I set
`REBUILD_ARTIFACTS = True` only when I want to recompute the full pipeline from
the private raw CSV files: validation, cleaning, EDA, feature engineering,
GroupKFold, four model families, a frozen grouped holdout, bounded tuning,
ablations, drift, diagnostics, the final model, its manifest, and the
submission. I use modeling seed 42 and retain the data SHA-256 fingerprints in
the manifest.
"""
        ),
        code(
            """
REBUILD_ARTIFACTS = False

if REBUILD_ARTIFACTS:
    results = run_final_analysis(tuning_iterations=8)
else:
    import json

    from game_player_analysis.cleaning import clean_ranking_sentinels
    from game_player_analysis.config import ARTIFACT_DIR
    from game_player_analysis.data import load_train_test

    table_names = (
        "dataset_summary",
        "data_quality",
        "date_integrity",
        "sampling_coverage",
        "numeric_profile",
        "game_modes",
        "kpi_evaluation",
        "feature_profiles",
        "distribution_shift",
        "categorical_shift",
        "categorical_shift_detail",
        "fold_audit",
        "final_holdout_audit",
        "final_holdout_evaluation",
        "holdout_audit",
        "holdout_performance",
        "model_parameter_audit",
        "pre_audit_configuration_comparison",
        "initial_model_comparison",
        "model_comparison",
        "model_fold_uncertainty",
        "feature_ablation",
        "scenario_comparison",
        "tuning_trials",
        "tuning_comparison",
        "subgroup_errors",
        "largest_errors",
        "permutation_importance",
        "shap_global_importance",
        "shap_sample",
        "shap_values",
    )
    tables = {
        name: pd.read_csv(ARTIFACT_DIR / "metrics" / f"{name}.csv")
        for name in table_names
    }
    for name in ("target_grid", "adversarial_validation", "drift_limitations"):
        tables[name] = pd.read_csv(
            ARTIFACT_DIR / "metrics" / f"{name}.csv", index_col=0
        )["value"]

    _, test_raw = load_train_test()
    metadata_dir = ARTIFACT_DIR / "metadata"
    selection_decision = json.loads(
        (metadata_dir / "final_selection_decision.json").read_text(encoding="utf-8")
    )
    tuning_decision = json.loads(
        (metadata_dir / "tuning_decision.json").read_text(encoding="utf-8")
    )
    figure_names = {
        "figure_date_integrity": "01_date_integrity.png",
        "figure_data_quality": "02_data_quality.png",
        "figure_sampling": "03_sampling_structure.png",
        "figure_target": "04_target_distribution.png",
        "figure_profiles": "05_feature_target_profiles.png",
        "figure_correlations": "06_targeted_correlations.png",
        "figure_validation": "07_validation_strategies.png",
        "figure_drift": "07b_train_test_drift.png",
        "figure_ablation": "08_feature_ablation.png",
        "figure_models": "09_model_diagnostics.png",
        "figure_tuning": "10_tuning_results.png",
        "figure_errors": "11_error_diagnostics.png",
        "figure_importance": "12_permutation_importance.png",
        "figure_shap": "13_catboost_shap_summary.png",
    }
    paths = {
        "model": ARTIFACT_DIR / "model.joblib",
        "manifest": ARTIFACT_DIR / "model_manifest.json",
        "submission": PROJECT_ROOT / "data" / "output" / "submission.csv",
        "tuning_decision": metadata_dir / "tuning_decision.json",
        "selection_decision": metadata_dir / "final_selection_decision.json",
        **{
            f"table_{name}": ARTIFACT_DIR / "metrics" / f"{name}.csv"
            for name in (*table_names, "target_grid", "adversarial_validation", "drift_limitations")
        },
        **{
            key: ARTIFACT_DIR / "figures" / filename
            for key, filename in figure_names.items()
        },
    }
    results = {
        "test": clean_ranking_sentinels(test_raw),
        "winner": selection_decision["selected_configuration"],
        "tables": tables,
        "paths": paths,
        "tuning_decision": tuning_decision,
        "selection_decision": selection_decision,
    }

tables = results["tables"]
display(Markdown(f"**Published model: {results['winner']}**"))
"""
        ),
        markdown("## 3. Loading, data contract, and cleaning"),
        code(
            """
display(tables["dataset_summary"])
display(tables["data_quality"].pivot(index="check", columns="dataset", values="rows"))
"""
        ),
        markdown(
            """
I find neither raw missing values nor exact duplicates. That does not mean every
value is semantically available: `rankPts=-1` is a documented sentinel, and
`killPts=0` and `winPts=0` are missing when the `rankPts` system is active. I
convert them to missing values and add availability flags without dropping rows.
I exclude these scores from the final model so that I do not mix ranking systems.
"""
        ),
        code(
            """
quality = tables["data_quality"]
display(quality.loc[quality["check"].str.contains("Pts|without|invalid")])
display(Image(filename=str(results["paths"]["figure_data_quality"])))
"""
        ),
        markdown("## 4. Date and sampling audit"),
        code(
            """
display(tables["date_integrity"])
display(Image(filename=str(results["paths"]["figure_date_integrity"])))
"""
        ),
        markdown(
            """
**My temporal-audit conclusion.** In 100% of valid multi-row `gameId`s, dates
differ between players; the median span is about 45 days in train. Dates are
therefore incompatible with their official row-level definition. I cannot
observe the technical cause in the files, so I call them inconsistent/corrupted
rather than invent an explanation. Time is only a pseudo-temporal stress test.
"""
        ),
        code(
            """
display(tables["sampling_coverage"])
display(Image(filename=str(results["paths"]["figure_sampling"])))
"""
        ),
        markdown(
            """
Only about 2.34% of train rows have an observed teammate, and nearly 98.82% of
`(gameId, teamId)` groups are singletons. I therefore **reject** team- and
lobby-level aggregates: they would mostly measure the sampling mechanism, not
complete team performance.
"""
        ),
        markdown("## 5. Train/test drift"),
        code(
            """
display(tables["distribution_shift"].head(16))
display(tables["categorical_shift"])
display(tables["adversarial_validation"])
display(tables["drift_limitations"])
display(Image(filename=str(results["paths"]["figure_drift"])))
"""
        ),
        markdown(
            """
I measure drift with PSI, KS, normalized Wasserstein distance, zero-mass
changes, categorical shifts, and adversarial validation. The maximum numeric
PSI is about 0.0051, the maximum categorical PSI is 0.0084, and adversarial ROC
AUC is 0.493: I detect no material drift in the measured feature contract. This
does not prove identical distributions. Without test targets, performance and
concept drift remain unknown; with inconsistent dates, temporal drift is not
interpretable.
"""
        ),
        markdown("## 6. Exploratory analysis and hypotheses"),
        code(
            """
display(tables["target_grid"].to_frame("value"))
display(tables["numeric_profile"])
display(Image(filename=str(results["paths"]["figure_target"])))
display(Image(filename=str(results["paths"]["figure_profiles"])))
display(Image(filename=str(results["paths"]["figure_correlations"])))
"""
        ),
        markdown(
            """
I examine four hypotheses: (1) mobility strongly reflects match progression;
(2) combat and resources add behavioural information; (3) `killRank`, already
computed after the match, should dominate the post-match scenario; and (4)
modes change the scale of behaviours. Quantile profiles and their 95% intervals
show nonlinear relationships, while I keep the Spearman matrix focused so those
effects remain visible.
"""
        ),
        markdown("## 7. Analytical KPIs"),
        code('display(tables["kpi_evaluation"])'),
        markdown(
            """
I define an explicit rule for every zero denominator. I keep `total_distance`,
mobility per second, and headshot ratio descriptive because they are redundant,
use `gameTime` rather than survival time, or are too sparse. I include
`damage_per_kill`, `combat_activity`, and `resource_activity` in the ablation:
I retain them only when they show measured value, not because they sound
plausible.
"""
        ),
        markdown("## 8. Features and progressive ablation"),
        code(
            """
display(tables["feature_ablation"].set_index("stage"))
display(tables["scenario_comparison"].set_index("scenario"))
display(Image(filename=str(results["paths"]["figure_ablation"])))
"""
        ),
        markdown(
            """
Mobility produces my main behavioural gain; combat and resources then improve
MAE modestly. Adding `killRank` reduces MAE from about 0.0927 to 0.0615, so I
explicitly describe the published model as **post-match**, not as early-game
prediction. Snapping to the `maxRank` grid improves MAE slightly but worsens
RMSE; I reserve it for the submission and document that trade-off.
"""
        ),
        markdown("## 9. Validation and leakage checks"),
        code(
            """
display(tables["fold_audit"])
display(tables["holdout_audit"])
display(tables["holdout_performance"])
display(tables["final_holdout_audit"])
display(tables["final_holdout_evaluation"])
display(Image(filename=str(results["paths"]["figure_validation"])))
"""
        ),
        markdown(
            """
I freeze a grouped holdout before model selection. The remaining 40,128 rows
feed 5-fold GroupKFold; I group repeated malformed values conservatively and no
identifier crosses a partition. A row-random split exposes more than half of its
validation rows to already seen matches. The January–March → April stress test
remains non-chronological because `date` is inconsistent. After freezing the
decision, CatBoost reaches about 0.06080 MAE on the cycle's 9,872-row holdout.
"""
        ),
        markdown("## 10. Selecting the number of folds"),
        code(
            """
fold_metrics_path = PROJECT_ROOT / "artifacts" / "metrics" / "fold_count_sensitivity.csv"
fold_decision_path = PROJECT_ROOT / "artifacts" / "metrics" / "fold_count_decision.csv"
fold_figure_path = PROJECT_ROOT / "artifacts" / "figures" / "07c_fold_count_sensitivity.png"
if not (fold_metrics_path.exists() and fold_decision_path.exists() and fold_figure_path.exists()):
    from scripts.run_fold_count_study import main as run_fold_count_study

    run_fold_count_study()

fold_sensitivity = pd.read_csv(fold_metrics_path)
fold_decision = pd.read_csv(fold_decision_path)
fold_comparison = (
    fold_sensitivity.pivot(index="n_splits", columns="model", values="mae")
    .join(fold_sensitivity.loc[fold_sensitivity["model"].eq("CatBoost")].set_index("n_splits")[
        ["mae_std", "mae_standard_error", "mean_validation_rows"]
    ])
    .join(fold_decision.set_index("n_splits")[
        ["catboost_winning_folds", "fit_cost_relative_to_5_folds"]
    ])
)
display(fold_comparison)
display(Image(filename=str(fold_figure_path)))
"""
        ),
        markdown(
            """
I compare `GroupKFold` with K=3, 5, 7, and 10 on the same 40,128 development
rows while keeping the final holdout closed. CatBoost beats XGBoost in all 25
combined folds. Seven folds reports a nominally lower MAE than five (a
**0.000189** gain), but that gap is below the folds' standard error and also
comes from mechanically larger training sets.

**My decision: 5 folds.** Each validation set retains about 8,026 independent
match rows, the CatBoost choice is already stable in all five folds, and seven
folds add meaningful computational cost. Ten folds at least doubles the
measured cost, reduces validation to about 4,013 rows, and increases dispersion.
The exact time ratio depends on the machine and is shown in the figure. Choosing
K=7 solely because its displayed score is lowest would optimize the protocol
after seeing results.
"""
        ),
        markdown("## 11. Baselines and initial comparison"),
        code(
            """
display(tables["initial_model_comparison"].set_index("rank"))
display(tables["model_fold_uncertainty"])
display(tables["pre_audit_configuration_comparison"])
display(Image(filename=str(results["paths"]["figure_models"])))
"""
        ),
        markdown(
            """
I use mean/median `DummyRegressor` models and Ridge as baselines. The four
ensembles use library-default learning hyperparameters, with only the seed,
parallelism, objective, and verbosity stated explicitly. CatBoost beats XGBoost
by about 0.0020 MAE and in all five folds. Replaying the earlier customized
configurations gives XGBoost only a 0.000054 advantage; that pre-tuning explains
the historical reversal of the winner.
"""
        ),
        markdown("## 12. Bounded tuning and anti-overfitting decision"),
        code(
            """
display(tables["tuning_trials"].sort_values("mae").head(8))
display(tables["tuning_comparison"])
display(pd.Series(results["tuning_decision"], name="decision").to_frame())
display(Image(filename=str(results["paths"]["figure_tuning"])))
"""
        ),
        markdown(
            """
I test eight CatBoost configurations with `RandomizedSearchCV` on the same
grouped folds. The best trial (0.061671) is worse than the default configuration
(0.061448) by 0.000223; it does not meet the 0.0001 material-gain threshold. I
reject tuning before opening the final holdout.
"""
        ),
        markdown("## 13. SHAP interpretability and error analysis"),
        code(
            """
display(tables["permutation_importance"].head(16))
display(tables["shap_global_importance"].head(16))
display(Image(filename=str(results["paths"]["figure_importance"])))
display(Image(filename=str(results["paths"]["figure_shap"])))
display(Image(filename=str(results["paths"]["figure_errors"])))
"""
        ),
        code(
            """
important_subgroups = tables["subgroup_errors"].query(
    "dimension in ['mode_family', 'rank_grid_size', 'target_band', 'kill_band']"
)
display(important_subgroups)
display(tables["largest_errors"].head(10))
"""
        ),
        markdown(
            """
Permutation and TreeSHAP give me complementary views. Permutation measures the
MAE loss when I shuffle one variable; TreeSHAP explains the direction and size
of each feature's contribution to every prediction. CatBoost computes TreeSHAP
natively on a deterministic sample of 2,000 rows from the 9,872-row holdout,
after I select the model. The pipeline verifies that SHAP values plus the
expected value reconstruct each raw prediction.

Both diagnostics place `killRank` first, followed by walking distance per
minute, `kills`, and `maxRank`. In the SHAP panel, a positive contribution raises
the predicted ranking and a negative contribution lowers it; color represents a
low (blue) or high (red) value **within the displayed feature**, not a unit that
is comparable across features. I interpret these results as post-match
predictive dependencies, not causal effects.

Special modes and small grids show the largest errors, but their samples are
small: I treat them as alerts rather than stable conclusions. Residuals show
regression to the mean for high targets and a few extreme errors; I export the
hardest cases for reproducible inspection.
"""
        ),
        markdown("## 14. Inference contract"),
        code(
            """
submission_check = predict_frame(
    results["test"],
    results["paths"]["model"],
)
assert list(submission_check.columns) == [
    "playerId", "teamId", "gameId", "winRankPercentage"
]
assert submission_check["winRankPercentage"].between(0, 1).all()
assert len(submission_check) == 5_000
display(submission_check.head())
display(pd.Series(results["paths"], name="path").tail(8).to_frame())
"""
        ),
        markdown(
            """
The bundle enforces the exact order of the 16 features, verifies its SHA-256,
and rejects missing official columns or non-finite predictions. The submission
preserves the 5,000-row order and contains only the three identifiers plus the
target bounded and snapped to the legal grid.
"""
        ),
        markdown(
            """
## 15. Conclusion, limitations, and next steps

With the default CatBoost configuration, I achieve about **0.06145** development
GroupKFold MAE and then **0.06080** MAE with **0.92083** R² on the cycle's
grouped holdout. Without `killRank`, my behavioural MAE is about **0.09266**.
My main conclusion is therefore not only the score: `killRank` changes the use
case itself.

The irreducible limitations are an unlabeled test set, unusable chronology,
highly partial match/team coverage, limited support for some modes, and no
early-game snapshots. For a production deployment, I would collect a coherent
match date, complete rosters, a genuinely labeled future split, and variables
available at the exact time of the decision.
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)


if __name__ == "__main__":
    build_notebook()
