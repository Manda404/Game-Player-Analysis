"""End-to-end analytical review, model benchmark and artifact publication."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone

from game_player_analysis.analysis import (
    data_quality_table,
    date_integrity_table,
    adversarial_validation_summary,
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
    categorical_shift_detail,
    categorical_shift_summary,
    dataset_summary,
    distribution_shift_summary,
    game_mode_family,
    game_mode_summary,
    load_train_test,
)
from game_player_analysis.evaluation import (
    build_submission,
    catboost_holdout_shap_values,
    holdout_permutation_importance,
    largest_error_cases,
    regression_metrics,
    snap_to_rank_grid,
    subgroup_error_summary,
)
from game_player_analysis.features import build_train_test_features
from game_player_analysis.modeling import (
    BASELINE_MODEL_NAMES,
    CATBOOST_TUNING_SPACE,
    XGBOOST_TUNING_SPACE,
    build_model_candidates,
    build_pre_audit_model_candidates,
    compare_models,
    cross_validate_model,
    evaluate_feature_sets,
    evaluate_holdout_strategies,
    model_parameter_audit_table,
    paired_fold_uncertainty,
    randomized_model_search,
    save_model_bundle,
)
from game_player_analysis.validation import (
    audit_group_folds,
    audit_holdout_splits,
    make_group_folds,
    make_final_group_holdout,
    make_holdout_splits,
)
from game_player_analysis.visualization import (
    apply_style,
    plot_date_contradiction,
    plot_drift_diagnostics,
    plot_error_diagnostics,
    plot_feature_ablation,
    plot_feature_target_profiles,
    plot_model_diagnostics,
    plot_permutation_importance,
    plot_quality_overview,
    plot_shap_summary,
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


def _tune_candidate(
    model_name: str,
    estimator: RegressorMixin,
    X: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[Any, Any]],
    baseline_mae: float,
    *,
    n_iter: int,
) -> tuple[pd.DataFrame, RegressorMixin, dict[str, Any]]:
    spaces = {
        "XGBoost": XGBOOST_TUNING_SPACE,
        "CatBoost": CATBOOST_TUNING_SPACE,
    }
    if model_name not in spaces:
        raise ValueError(f"No bounded tuning space is defined for {model_name}")
    tuning, best_parameters = randomized_model_search(
        estimator,
        X,
        y,
        folds,
        spaces[model_name],
        n_iter=n_iter,
    )
    best_estimator = clone(estimator).set_params(**best_parameters)
    gain = baseline_mae - float(tuning.iloc[0]["mae"])
    decision: dict[str, Any] = {
        "model_family": model_name,
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
    diagnostic_frame: pd.DataFrame,
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
    plot_drift_diagnostics(
        tables["distribution_shift"],
        tables["categorical_shift"],
        tables["adversarial_validation"],
    )
    paths["figure_drift"] = _save_figure("07b_train_test_drift")
    plot_feature_ablation(tables["feature_ablation"])
    paths["figure_ablation"] = _save_figure("08_feature_ablation")
    plot_model_diagnostics(benchmark, fold_details)
    paths["figure_models"] = _save_figure("09_model_diagnostics")
    plot_tuning_results(tuning, float(tuning_decision["baseline_mae"]))
    paths["figure_tuning"] = _save_figure("10_tuning_results")
    plot_error_diagnostics(diagnostic_frame, oof_prediction, tables["subgroup_errors"])
    paths["figure_errors"] = _save_figure("11_error_diagnostics")
    plot_permutation_importance(tables["permutation_importance"])
    paths["figure_importance"] = _save_figure("12_permutation_importance")
    plot_shap_summary(tables["shap_global_importance"], tables["shap_values"])
    paths["figure_shap"] = _save_figure("13_catboost_shap_summary")
    return paths


def run_final_analysis(*, tuning_iterations: int = 8) -> dict[str, Any]:
    """Run the audited selection, drift and publication workflow."""
    logger.info("Loading and validating official train/test files")
    train_raw, test_raw = load_train_test()
    train = clean_ranking_sentinels(train_raw)
    test = clean_ranking_sentinels(test_raw)
    X_train, X_test = build_train_test_features(train, test, include_kill_rank=True)
    y = train[TARGET]

    development_index, final_holdout_index = make_final_group_holdout(train)
    development = train.iloc[development_index].reset_index(drop=True)
    X_development = X_train.iloc[development_index].reset_index(drop=True)
    y_development = development[TARGET]
    folds = make_group_folds(development)
    fold_audit = audit_group_folds(development, folds)
    final_holdout_audit = audit_holdout_splits(
        train,
        {"Final cycle holdout": (development_index, final_holdout_index)},
    )
    holdout_splits = make_holdout_splits(train)
    holdout_audit = audit_holdout_splits(train, holdout_splits)

    logger.info("Benchmarking defaults on development folds only")
    benchmark, oof_predictions, fold_details = compare_models(
        X_development,
        y_development,
        folds,
    )
    base_winner = _select_candidate(benchmark)
    estimator = build_model_candidates()[base_winner]
    deployed_name = base_winner
    deployed_oof = oof_predictions[base_winner]

    pre_audit_rows: list[dict[str, Any]] = []
    for model_name in ("CatBoost", "XGBoost"):
        summary, _, _ = cross_validate_model(
            f"{model_name} pre-audit custom",
            build_pre_audit_model_candidates()[model_name],
            X_development,
            y_development,
            folds,
        )
        pre_audit_rows.append(summary)
    pre_audit_comparison = pd.DataFrame(pre_audit_rows).sort_values("mae")

    baseline_summary = benchmark.loc[benchmark["model"].eq(base_winner)].iloc[0].to_dict()
    if base_winner in {"CatBoost", "XGBoost"}:
        logger.info("Tuning fair-comparison winner %s (%d trials)", base_winner, tuning_iterations)
        tuning, tuned_estimator, tuning_decision = _tune_candidate(
            base_winner,
            estimator,
            X_development,
            y_development,
            folds,
            float(baseline_summary["mae"]),
            n_iter=tuning_iterations,
        )
        tuned_summary, tuned_oof, _ = cross_validate_model(
            f"{base_winner} tuned",
            tuned_estimator,
            X_development,
            y_development,
            folds,
        )
        if tuning_decision["accepted"]:
            estimator = tuned_estimator
            deployed_name = f"{base_winner} tuned"
            deployed_oof = tuned_oof
    else:
        tuned_summary = dict(baseline_summary)
        tuned_summary["model"] = f"{base_winner} untuned"
        tuning = pd.DataFrame([{"trial": 0, "mae": baseline_summary["mae"]}])
        tuning_decision = {
            "model_family": base_winner,
            "baseline_mae": baseline_summary["mae"],
            "best_tuned_mae": baseline_summary["mae"],
            "mae_gain": 0.0,
            "minimum_material_gain": 0.0001,
            "accepted": False,
            "best_parameters": {},
            "reason": "No bounded search space is defined for this family.",
        }

    tuning_comparison = pd.DataFrame(
        [
            {**baseline_summary, "configuration": f"{base_winner} library defaults"},
            {**tuned_summary, "configuration": f"{base_winner} randomized-search winner"},
        ]
    )

    logger.info("Evaluating frozen selection once on the cycle holdout")
    final_model_for_evaluation = clone(estimator)
    start = time.perf_counter()
    final_model_for_evaluation.fit(X_train.iloc[development_index], y.iloc[development_index])
    final_fit_seconds = time.perf_counter() - start
    development_prediction = np.clip(
        final_model_for_evaluation.predict(X_train.iloc[development_index]), 0.0, 1.0
    )
    start = time.perf_counter()
    final_holdout_prediction = np.clip(
        final_model_for_evaluation.predict(X_train.iloc[final_holdout_index]), 0.0, 1.0
    )
    final_predict_seconds = time.perf_counter() - start
    final_train_metrics = regression_metrics(y.iloc[development_index], development_prediction)
    final_holdout_metrics = regression_metrics(
        y.iloc[final_holdout_index], final_holdout_prediction
    )
    final_holdout_evaluation = pd.DataFrame(
        [
            {
                "model": deployed_name,
                "development_rows": len(development_index),
                "holdout_rows": len(final_holdout_index),
                "train_mae": final_train_metrics["mae"],
                **final_holdout_metrics,
                "mae_gap": final_holdout_metrics["mae"] - final_train_metrics["mae"],
                "fit_seconds": final_fit_seconds,
                "predict_seconds": final_predict_seconds,
                "shared_games": int(final_holdout_audit.iloc[0]["shared_groups"]),
                "decision_frozen_before_holdout": True,
                "historical_limitation": (
                    "independent of this audit cycle, but earlier EDA used all labeled rows"
                ),
            }
        ]
    )

    logger.info("Auditing feature families, splits, errors and interpretability")
    feature_ablation = evaluate_feature_sets(
        estimator,
        X_development,
        y_development,
        folds,
        FEATURE_SETS,
    )
    holdout_performance = evaluate_holdout_strategies(
        estimator,
        X_train,
        y,
        holdout_splits,
    )
    importance, importance_prediction = holdout_permutation_importance(
        estimator,
        X_train,
        y,
        development_index,
        final_holdout_index,
    )
    if not np.allclose(importance_prediction, final_holdout_prediction):
        raise RuntimeError("Final holdout predictions changed during importance analysis")
    shap_global_importance, shap_sample, shap_values = catboost_holdout_shap_values(
        final_model_for_evaluation,
        X_train,
        y,
        final_holdout_index,
        final_holdout_prediction,
    )
    subgroup_errors = subgroup_error_summary(development, deployed_oof)
    largest_errors = largest_error_cases(development, deployed_oof)

    post_metrics = regression_metrics(y_development, deployed_oof)
    snapped_prediction = snap_to_rank_grid(deployed_oof, development["maxRank"])
    snapped_metrics = regression_metrics(y_development, snapped_prediction)
    behavior_row = feature_ablation.loc[feature_ablation["stage"].eq("4. + resources")].iloc[0]
    scenario_comparison = pd.DataFrame(
        [
            {
                "scenario": "post_match_with_killRank",
                **post_metrics,
                "features": X_development.shape[1],
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
                "features": X_development.shape[1],
            },
        ]
    )
    fold_uncertainty = paired_fold_uncertainty(fold_details, base_winner)

    drift_train = train.assign(mode_family=game_mode_family(train["gameType"]))
    drift_test = test.assign(mode_family=game_mode_family(test["gameType"]))
    categorical_columns = ("gameType", "mode_family", "ranking_system")
    numeric_shift = distribution_shift_summary(X_train, X_test, tuple(X_train.columns))
    categorical_shift = categorical_shift_summary(
        drift_train,
        drift_test,
        categorical_columns,
    )
    categorical_detail = categorical_shift_detail(
        drift_train,
        drift_test,
        categorical_columns,
    )
    adversarial_validation = adversarial_validation_summary(X_train, X_test)
    drift_limitations = pd.Series(
        {
            "data_drift_train_vs_test": "measurable without target",
            "performance_or_concept_drift": "not measurable: official test target is unavailable",
            "temporal_drift": "not interpretable: date conflicts within gameId",
            "decision_rule": "effect sizes first; KS p-values are descriptive only",
        }
    )

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
        "distribution_shift": numeric_shift,
        "categorical_shift": categorical_shift,
        "categorical_shift_detail": categorical_detail,
        "adversarial_validation": adversarial_validation,
        "drift_limitations": drift_limitations,
        "fold_audit": fold_audit,
        "final_holdout_audit": final_holdout_audit,
        "final_holdout_evaluation": final_holdout_evaluation,
        "holdout_audit": holdout_audit,
        "holdout_performance": holdout_performance,
        "model_parameter_audit": model_parameter_audit_table(),
        "pre_audit_configuration_comparison": pre_audit_comparison,
        "initial_model_comparison": benchmark,
        "model_comparison": benchmark,
        "model_fold_uncertainty": fold_uncertainty,
        "feature_ablation": feature_ablation,
        "scenario_comparison": scenario_comparison,
        "tuning_trials": tuning,
        "tuning_comparison": tuning_comparison,
        "subgroup_errors": subgroup_errors,
        "largest_errors": largest_errors,
        "permutation_importance": importance,
        "shap_global_importance": shap_global_importance,
        "shap_sample": shap_sample,
        "shap_values": shap_values,
    }

    table_paths = {f"table_{name}": _write_table(table, name) for name, table in tables.items()}

    figure_paths = _make_figures(
        train,
        test,
        development,
        tables,
        benchmark,
        fold_details,
        pd.Series(deployed_oof, index=development.index),
        tuning,
        tuning_decision,
    )

    logger.info("Fitting and publishing %s on all labeled rows", deployed_name)
    final_model = clone(estimator).fit(X_train, y)
    model_paths = save_model_bundle(
        final_model,
        X_train.columns,
        benchmark,
        output_dir=ARTIFACT_DIR,
        model_name=deployed_name,
        metrics=final_holdout_metrics,
        validation_strategy=(
            "5-fold GroupKFold(gameId) on development; frozen grouped cycle holdout"
        ),
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
    final_holdout_record = json.loads(final_holdout_evaluation.to_json(orient="records"))[0]
    selection_decision = {
        "initial_winner": base_winner,
        "selected_configuration": deployed_name,
        "primary_metric": "MAE",
        "secondary_metrics": ["RMSE", "R2", "fold stability", "train-validation gap"],
        "tuning_accepted": bool(tuning_decision["accepted"]),
        "final_holdout": final_holdout_record,
        "test_used_for_selection": False,
        "temporal_validation_status": "rejected because date conflicts within gameId",
    }
    selection_path = METADATA_DIR / "final_selection_decision.json"
    selection_path.write_text(
        json.dumps(selection_decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    paths = {
        **model_paths,
        **table_paths,
        **figure_paths,
        "tuning_decision": tuning_path,
        "selection_decision": selection_path,
        "submission": submission_path,
    }
    logger.info("Analysis complete: final holdout MAE %.6f", final_holdout_metrics["mae"])
    return {
        "train": train,
        "test": test,
        "features": X_train,
        "winner": deployed_name,
        "oof_prediction": pd.Series(
            deployed_oof,
            index=development.index,
            name="prediction",
        ),
        "development": development,
        "final_holdout_index": final_holdout_index,
        "tables": tables,
        "paths": paths,
        "tuning_decision": tuning_decision,
        "selection_decision": selection_decision,
    }
