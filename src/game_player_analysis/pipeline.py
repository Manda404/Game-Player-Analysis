"""End-to-end analytical review, model benchmark and artifact publication."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import RegressorMixin, clone

from game_player_analysis.analysis import (
    data_quality_table,
    date_integrity_table,
    feature_target_profiles,
    kpi_evaluation_table,
    numeric_profile,
    sampling_coverage_table,
    target_grid_summary,
)
from game_player_analysis.cleaning import clean_ranking_sentinels
from game_player_analysis.config import (
    ARTIFACT_DIR,
    FIGURES_DIR,
    METADATA_DIR,
    METRICS_DIR,
    OUTPUT_DIR,
    TARGET,
)
from game_player_analysis.data import (
    dataset_summary,
    distribution_shift_summary,
    game_mode_summary,
    load_train_test,
)
from game_player_analysis.evaluation import (
    build_submission,
    holdout_permutation_importance,
    largest_error_cases,
    regression_metrics,
    snap_to_rank_grid,
    subgroup_error_summary,
)
from game_player_analysis.features import build_train_test_features
from game_player_analysis.modeling import (
    BASELINE_MODEL_NAMES,
    XGBOOST_TUNING_SPACE,
    build_model_candidates,
    compare_models,
    cross_validate_model,
    evaluate_feature_sets,
    evaluate_holdout_strategies,
    paired_fold_uncertainty,
    randomized_model_search,
    save_model_bundle,
)
from game_player_analysis.validation import (
    audit_group_folds,
    audit_holdout_splits,
    make_group_folds,
    make_holdout_splits,
)
from game_player_analysis.visualization import (
    apply_style,
    plot_date_contradiction,
    plot_error_diagnostics,
    plot_feature_ablation,
    plot_feature_target_profiles,
    plot_model_diagnostics,
    plot_permutation_importance,
    plot_quality_overview,
    plot_sampling_structure,
    plot_target_distribution,
    plot_targeted_correlation_heatmap,
    plot_tuning_results,
    plot_validation_comparison,
)

logger = logging.getLogger(__name__)

PROFILE_COLUMNS = (
    "walkDist",
    "rideDist",
    "damages",
    "kills",
    "weapons",
    "upgrades",
    "heals",
    "killRank",
    "maxRank",
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "1. context": ("maxRank", "mode_solo", "mode_duo", "mode_squad"),
    "2. + mobility": (
        "maxRank",
        "mode_solo",
        "mode_duo",
        "mode_squad",
        "walkDist",
        "rideDist",
        "walk_distance_per_match_minute",
    ),
    "3. + combat": (
        "maxRank",
        "mode_solo",
        "mode_duo",
        "mode_squad",
        "walkDist",
        "rideDist",
        "walk_distance_per_match_minute",
        "damages",
        "kills",
        "damage_per_kill",
        "combat_activity",
    ),
    "4. + resources": (
        "maxRank",
        "mode_solo",
        "mode_duo",
        "mode_squad",
        "walkDist",
        "rideDist",
        "walk_distance_per_match_minute",
        "damages",
        "kills",
        "damage_per_kill",
        "combat_activity",
        "weapons",
        "upgrades",
        "heals",
        "resource_activity",
    ),
    "5. + killRank post-match": (
        "killRank",
        "maxRank",
        "mode_solo",
        "mode_duo",
        "mode_squad",
        "walkDist",
        "rideDist",
        "walk_distance_per_match_minute",
        "damages",
        "kills",
        "damage_per_kill",
        "combat_activity",
        "weapons",
        "upgrades",
        "heals",
        "resource_activity",
    ),
}


def _write_table(
    table: pd.DataFrame | pd.Series,
    name: str,
) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"{name}.csv"
    if isinstance(table, pd.Series):
        table.rename("value").to_csv(path, header=True)
    else:
        table.to_csv(path, index=not isinstance(table.index, pd.RangeIndex))
    return path


def _save_figure(name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    figure = plt.gcf()
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def _select_candidate(benchmark: pd.DataFrame) -> str:
    candidates = benchmark.loc[~benchmark["model"].isin(BASELINE_MODEL_NAMES)]
    if candidates.empty:
        raise RuntimeError("The benchmark did not evaluate a deployable model")
    return str(candidates.iloc[0]["model"])


def _tune_xgboost(
    estimator: RegressorMixin,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[Any, Any]],
    baseline_mae: float,
    *,
    n_iter: int,
) -> tuple[pd.DataFrame, RegressorMixin, dict[str, Any]]:
    tuning, best_parameters = randomized_model_search(
        estimator,
        X,
        y,
        folds,
        XGBOOST_TUNING_SPACE,
        n_iter=n_iter,
    )
    best_estimator = clone(estimator).set_params(**best_parameters)
    gain = baseline_mae - float(tuning.iloc[0]["mae"])
    decision = {
        "baseline_mae": baseline_mae,
        "best_tuned_mae": float(tuning.iloc[0]["mae"]),
        "mae_gain": gain,
        "minimum_material_gain": 0.0001,
        "accepted": gain >= 0.0001,
        "best_parameters": best_parameters,
    }
    return tuning, best_estimator, decision


def _make_figures(
    train: pd.DataFrame,
    test: pd.DataFrame,
    tables: Mapping[str, pd.DataFrame | pd.Series],
    benchmark: pd.DataFrame,
    fold_details: dict[str, pd.DataFrame],
    oof_prediction: pd.Series,
    tuning: pd.DataFrame,
    tuning_decision: Mapping[str, Any],
) -> dict[str, Path]:
    apply_style()
    paths: dict[str, Path] = {}

    plot_date_contradiction(train, test)
    paths["figure_date_integrity"] = _save_figure("01_date_integrity")
    plot_quality_overview(tables["data_quality"], tables["numeric_profile"])
    paths["figure_data_quality"] = _save_figure("02_data_quality")
    plot_sampling_structure(train)
    paths["figure_sampling"] = _save_figure("03_sampling_structure")
    plot_target_distribution(train)
    paths["figure_target"] = _save_figure("04_target_distribution")
    plot_feature_target_profiles(
        tables["feature_profiles"],
        ["walkDist", "kills", "upgrades", "killRank"],
    )
    paths["figure_profiles"] = _save_figure("05_feature_target_profiles")
    plot_targeted_correlation_heatmap(
        train,
        ["walkDist", "damages", "kills", "weapons", "upgrades", "killRank", TARGET],
    )
    paths["figure_correlations"] = _save_figure("06_targeted_correlations")
    plot_validation_comparison(
        tables["holdout_audit"],
        tables["holdout_performance"],
    )
    paths["figure_validation"] = _save_figure("07_validation_strategies")
    plot_feature_ablation(tables["feature_ablation"])
    paths["figure_ablation"] = _save_figure("08_feature_ablation")
    plot_model_diagnostics(benchmark, fold_details)
    paths["figure_models"] = _save_figure("09_model_diagnostics")
    plot_tuning_results(tuning, float(tuning_decision["baseline_mae"]))
    paths["figure_tuning"] = _save_figure("10_tuning_results")
    plot_error_diagnostics(train, oof_prediction, tables["subgroup_errors"])
    paths["figure_errors"] = _save_figure("11_error_diagnostics")
    plot_permutation_importance(tables["permutation_importance"])
    paths["figure_importance"] = _save_figure("12_permutation_importance")
    return paths


def run_final_analysis(*, tuning_iterations: int = 6) -> dict[str, Any]:
    """Run the full review workflow and return notebook-ready objects."""
    logger.info("Loading and validating official train/test files")
    train_raw, test_raw = load_train_test()
    train = clean_ranking_sentinels(train_raw)
    test = clean_ranking_sentinels(test_raw)
    X_train, X_test = build_train_test_features(train, test, include_kill_rank=True)
    y = train[TARGET]

    folds = make_group_folds(train)
    fold_audit = audit_group_folds(train, folds)
    holdout_splits = make_holdout_splits(train)
    holdout_audit = audit_holdout_splits(train, holdout_splits)

    logger.info("Benchmarking baselines and four ensemble families")
    benchmark, oof_predictions, fold_details = compare_models(X_train, y, folds)
    base_winner = _select_candidate(benchmark)
    estimator = build_model_candidates()[base_winner]
    deployed_name = base_winner
    deployed_oof = oof_predictions[base_winner]

    if base_winner == "XGBoost":
        logger.info("Running bounded XGBoost tuning (%d trials)", tuning_iterations)
        tuning, tuned_estimator, tuning_decision = _tune_xgboost(
            estimator,
            X_train,
            y,
            folds,
            float(benchmark.loc[benchmark["model"].eq(base_winner), "mae"].iloc[0]),
            n_iter=tuning_iterations,
        )
        tuned_summary, tuned_oof, tuned_detail = cross_validate_model(
            "XGBoost tuned",
            tuned_estimator,
            X_train,
            y,
            folds,
        )
        benchmark = pd.concat([benchmark, pd.DataFrame([tuned_summary])], ignore_index=True)
        benchmark = benchmark.drop(columns="rank").sort_values("mae").reset_index(drop=True)
        benchmark.insert(0, "rank", range(1, len(benchmark) + 1))
        fold_details["XGBoost tuned"] = tuned_detail
        oof_predictions["XGBoost tuned"] = tuned_oof
        if tuning_decision["accepted"]:
            estimator = tuned_estimator
            deployed_name = "XGBoost tuned"
            deployed_oof = tuned_oof
    else:
        base_mae = float(benchmark.loc[benchmark["model"].eq(base_winner), "mae"].iloc[0])
        tuning = pd.DataFrame([{"trial": 0, "mae": base_mae}])
        tuning_decision = {
            "baseline_mae": base_mae,
            "best_tuned_mae": base_mae,
            "mae_gain": 0.0,
            "minimum_material_gain": 0.0001,
            "accepted": False,
            "best_parameters": {},
            "reason": "Tuning scope was intentionally limited to the leading XGBoost family.",
        }

    logger.info("Auditing feature families, splits, errors and interpretability")
    feature_ablation = evaluate_feature_sets(estimator, X_train, y, folds, FEATURE_SETS)
    holdout_performance = evaluate_holdout_strategies(
        estimator,
        X_train,
        y,
        holdout_splits,
    )
    grouped_train, grouped_validation = holdout_splits["Grouped gameId"]
    importance, _ = holdout_permutation_importance(
        estimator,
        X_train,
        y,
        grouped_train,
        grouped_validation,
    )
    subgroup_errors = subgroup_error_summary(train, deployed_oof)
    largest_errors = largest_error_cases(train, deployed_oof)

    post_metrics = regression_metrics(y, deployed_oof)
    snapped_prediction = snap_to_rank_grid(deployed_oof, train["maxRank"])
    snapped_metrics = regression_metrics(y, snapped_prediction)
    behavior_row = feature_ablation.loc[feature_ablation["stage"].eq("4. + resources")].iloc[0]
    scenario_comparison = pd.DataFrame(
        [
            {
                "scenario": "post_match_with_killRank",
                **post_metrics,
                "features": X_train.shape[1],
            },
            {
                "scenario": "behavior_without_killRank",
                "mae": behavior_row["mae"],
                "rmse": behavior_row["rmse"],
                "r2": behavior_row["r2"],
                "features": int(behavior_row["feature_count"]),
            },
            {
                "scenario": "post_match_grid_snapped",
                **snapped_metrics,
                "features": X_train.shape[1],
            },
        ]
    )
    fold_uncertainty = paired_fold_uncertainty(fold_details, deployed_name)

    tables: dict[str, pd.DataFrame | pd.Series] = {
        "dataset_summary": pd.concat(
            [dataset_summary(train_raw), dataset_summary(test_raw)],
            axis=1,
            keys=["train", "test"],
        ),
        "data_quality": data_quality_table(train_raw, test_raw),
        "date_integrity": date_integrity_table(train_raw, test_raw),
        "sampling_coverage": sampling_coverage_table(train_raw, test_raw),
        "numeric_profile": numeric_profile(train, PROFILE_COLUMNS),
        "game_modes": game_mode_summary(train),
        "kpi_evaluation": kpi_evaluation_table(train),
        "feature_profiles": feature_target_profiles(
            train,
            ["walkDist", "kills", "upgrades", "killRank"],
        ),
        "target_grid": target_grid_summary(train),
        "distribution_shift": distribution_shift_summary(
            train,
            test,
            PROFILE_COLUMNS,
        ),
        "fold_audit": fold_audit,
        "holdout_audit": holdout_audit,
        "holdout_performance": holdout_performance,
        "model_comparison": benchmark,
        "model_fold_uncertainty": fold_uncertainty,
        "feature_ablation": feature_ablation,
        "scenario_comparison": scenario_comparison,
        "tuning_trials": tuning,
        "subgroup_errors": subgroup_errors,
        "largest_errors": largest_errors,
        "permutation_importance": importance,
    }

    table_paths = {f"table_{name}": _write_table(table, name) for name, table in tables.items()}

    figure_paths = _make_figures(
        train,
        test,
        tables,
        benchmark,
        fold_details,
        pd.Series(deployed_oof, index=train.index),
        tuning,
        tuning_decision,
    )

    logger.info("Fitting and publishing %s", deployed_name)
    final_model = clone(estimator).fit(X_train, y)
    model_paths = save_model_bundle(
        final_model,
        X_train.columns,
        benchmark,
        output_dir=ARTIFACT_DIR,
        model_name=deployed_name,
        metrics=post_metrics,
        training_rows=len(train),
    )
    submission = build_submission(test, final_model.predict(X_test))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    tuning_path = METADATA_DIR / "tuning_decision.json"
    tuning_path.write_text(
        json.dumps(tuning_decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    paths = {
        **model_paths,
        **table_paths,
        **figure_paths,
        "tuning_decision": tuning_path,
        "submission": submission_path,
    }
    logger.info("Analysis complete: MAE %.6f", post_metrics["mae"])
    return {
        "train": train,
        "test": test,
        "features": X_train,
        "winner": deployed_name,
        "oof_prediction": pd.Series(deployed_oof, index=train.index, name="prediction"),
        "tables": tables,
        "paths": paths,
        "tuning_decision": tuning_decision,
    }
