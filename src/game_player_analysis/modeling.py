"""Readable model construction, grouped evaluation and persistence."""

from __future__ import annotations

import json
import logging
import math
import platform
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from game_player_analysis.config import ARTIFACT_DIR, N_SPLITS, RANDOM_STATE, TARGET
from game_player_analysis.data import raw_data_fingerprints, sha256_file
from game_player_analysis.evaluation import overfitting_comment, regression_metrics

FoldIndices = Sequence[tuple[np.ndarray, np.ndarray]]
logger = logging.getLogger(__name__)

BASELINE_MODEL_NAMES = (
    "Mean baseline",
    "Median baseline",
    "Linear Ridge baseline",
)

XGBOOST_TUNING_SPACE: dict[str, list[Any]] = {
    "n_estimators": [300, 500, 700],
    "learning_rate": [0.03, 0.05, 0.08],
    "max_depth": [4, 6, 8],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.75, 0.85, 1.0],
    "colsample_bytree": [0.75, 0.85, 1.0],
    "reg_alpha": [0.0, 0.1, 0.5],
    "reg_lambda": [1.0, 5.0, 10.0],
}

CATBOOST_TUNING_SPACE: dict[str, list[Any]] = {
    "iterations": [400, 800, 1_200],
    "learning_rate": [0.03, 0.05, 0.08],
    "depth": [4, 6, 8],
    "l2_leaf_reg": [1.0, 3.0, 5.0, 10.0],
    "random_strength": [0.5, 1.0, 2.0],
    "border_count": [64, 128, 254],
}

TECHNICAL_PARAMETERS: dict[str, tuple[str, ...]] = {
    "Random Forest": ("random_state", "n_jobs"),
    "XGBoost": ("objective", "random_state", "n_jobs", "verbosity"),
    "LightGBM": ("random_state", "n_jobs", "verbosity"),
    "CatBoost": (
        "loss_function",
        "random_seed",
        "verbose",
        "allow_writing_files",
        "thread_count",
    ),
}

LEARNING_PARAMETERS: dict[str, tuple[str, ...]] = {
    "Random Forest": ("n_estimators", "min_samples_leaf", "max_features", "max_depth"),
    "XGBoost": (
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    ),
    "LightGBM": (
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    ),
    "CatBoost": (
        "iterations",
        "learning_rate",
        "depth",
        "l2_leaf_reg",
        "random_strength",
        "bagging_temperature",
        "border_count",
    ),
}


def build_model_candidates(
    random_state: int = RANDOM_STATE,
) -> dict[str, RegressorMixin]:
    """Return a linear baseline and untuned candidate model families.

    Learning hyperparameters retain library defaults. Only operational
    settings for reproducibility, resources, output and regression loss are
    supplied explicitly.
    """
    return {
        "Linear Ridge baseline": make_pipeline(
            StandardScaler(),
            Ridge(),
        ),
        "Random Forest": RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        ),
        "CatBoost": CatBoostRegressor(
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        ),
    }


def build_pre_audit_model_candidates(
    random_state: int = RANDOM_STATE,
) -> dict[str, RegressorMixin]:
    """Reproduce the customized configurations used before this audit."""
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=800,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=5.0,
            loss_function="RMSE",
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        ),
    }


def model_parameter_audit_table() -> pd.DataFrame:
    """Contrast the pre-audit custom benchmark with the fair initial setup."""
    previous = build_pre_audit_model_candidates()
    audited = build_model_candidates()
    rows: list[dict[str, Any]] = []
    for model_name in LEARNING_PARAMETERS:
        previous_parameters = previous[model_name].get_params(deep=False)
        audited_parameters = audited[model_name].get_params(deep=False)
        previous_learning = {
            parameter: previous_parameters[parameter]
            for parameter in LEARNING_PARAMETERS[model_name]
            if parameter in previous_parameters
        }
        audited_technical = {
            parameter: audited_parameters[parameter]
            for parameter in TECHNICAL_PARAMETERS[model_name]
            if parameter in audited_parameters
        }
        rows.append(
            {
                "model": model_name,
                "pre_audit_learning_parameters": json.dumps(previous_learning, sort_keys=True),
                "audited_initial_learning_parameters": "library defaults",
                "audited_technical_parameters": json.dumps(audited_technical, sort_keys=True),
                "pre_audit_comparison_fair": False,
                "audited_initial_comparison_fair": True,
            }
        )
    return pd.DataFrame(rows)


def _baseline_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    folds: FoldIndices,
    feature_count: int,
    *,
    statistic: str,
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    if statistic not in {"mean", "median"}:
        raise ValueError("Baseline statistic must be 'mean' or 'median'")
    oof = np.full(len(y), np.nan)
    rows = []
    for fold, (train_index, validation_index) in enumerate(folds, start=1):
        model = DummyRegressor(strategy=statistic)
        model.fit(X.iloc[train_index], y.iloc[train_index])
        train_prediction = model.predict(X.iloc[train_index])
        validation_prediction = model.predict(X.iloc[validation_index])
        oof[validation_index] = validation_prediction
        train_metrics = regression_metrics(y.iloc[train_index], train_prediction)
        rows.append(
            {
                "fold": fold,
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                **regression_metrics(y.iloc[validation_index], validation_prediction),
                "fit_seconds": 0.0,
                "predict_seconds": 0.0,
            }
        )
    detail = pd.DataFrame(rows)
    model_name = f"{statistic.title()} baseline"
    summary = {
        "model": model_name,
        "mae": detail["mae"].mean(),
        "mae_std": detail["mae"].std(ddof=1),
        "mae_min": detail["mae"].min(),
        "mae_max": detail["mae"].max(),
        "rmse": detail["rmse"].mean(),
        "rmse_std": detail["rmse"].std(ddof=1),
        "r2": detail["r2"].mean(),
        "r2_std": detail["r2"].std(ddof=1),
        "train_mae": detail["train_mae"].mean(),
        "mae_gap": detail["mae"].mean() - detail["train_mae"].mean(),
        "fit_seconds": 0.0,
        "predict_seconds": 0.0,
        "feature_count": feature_count,
        "overfitting": "non applicable",
    }
    return summary, oof, detail


def cross_validate_model(
    name: str,
    estimator: RegressorMixin,
    X: pd.DataFrame,
    y: pd.Series,
    folds: FoldIndices,
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    """Evaluate one model on precomputed match-grouped folds."""
    oof = np.full(len(X), np.nan)
    rows = []
    for fold, (train_index, validation_index) in enumerate(folds, start=1):
        model = clone(estimator)
        start = time.perf_counter()
        model.fit(X.iloc[train_index], y.iloc[train_index])
        fit_seconds = time.perf_counter() - start

        train_prediction = np.clip(model.predict(X.iloc[train_index]), 0.0, 1.0)
        start = time.perf_counter()
        validation_prediction = np.clip(model.predict(X.iloc[validation_index]), 0.0, 1.0)
        predict_seconds = time.perf_counter() - start
        oof[validation_index] = validation_prediction
        train_metrics = regression_metrics(y.iloc[train_index], train_prediction)
        rows.append(
            {
                "fold": fold,
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "train_r2": train_metrics["r2"],
                **regression_metrics(y.iloc[validation_index], validation_prediction),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            }
        )
    if np.isnan(oof).any():
        raise RuntimeError(f"{name} did not predict every row out of fold")

    detail = pd.DataFrame(rows)
    validation_mae = float(detail["mae"].mean())
    train_mae = float(detail["train_mae"].mean())
    summary = {
        "model": name,
        "mae": validation_mae,
        "mae_std": float(detail["mae"].std(ddof=1)),
        "mae_min": float(detail["mae"].min()),
        "mae_max": float(detail["mae"].max()),
        "rmse": float(detail["rmse"].mean()),
        "rmse_std": float(detail["rmse"].std(ddof=1)),
        "r2": float(detail["r2"].mean()),
        "r2_std": float(detail["r2"].std(ddof=1)),
        "train_mae": train_mae,
        "mae_gap": validation_mae - train_mae,
        "fit_seconds": float(detail["fit_seconds"].sum()),
        "predict_seconds": float(detail["predict_seconds"].sum()),
        "feature_count": X.shape[1],
        "overfitting": overfitting_comment(train_mae, validation_mae),
    }
    return summary, oof, detail


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    folds: FoldIndices,
    *,
    models: Mapping[str, RegressorMixin] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """Compare one baseline and all candidate families on identical folds."""
    candidates = dict(models or build_model_candidates())
    mean_baseline, mean_oof, mean_detail = _baseline_evaluation(
        X,
        y,
        folds,
        X.shape[1],
        statistic="mean",
    )
    median_baseline, median_oof, median_detail = _baseline_evaluation(
        X,
        y,
        folds,
        X.shape[1],
        statistic="median",
    )
    summaries = [mean_baseline, median_baseline]
    predictions = {
        "Mean baseline": mean_oof,
        "Median baseline": median_oof,
    }
    details = {
        "Mean baseline": mean_detail,
        "Median baseline": median_detail,
    }
    for name, estimator in candidates.items():
        summary, oof, detail = cross_validate_model(name, estimator, X, y, folds)
        summaries.append(summary)
        predictions[name] = oof
        details[name] = detail
    results = pd.DataFrame(summaries).sort_values("mae").reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    return results, predictions, details


def paired_fold_uncertainty(
    fold_details: Mapping[str, pd.DataFrame],
    reference_model: str,
    *,
    bootstrap_samples: int = 10_000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Bootstrap paired fold-MAE differences against one reference model.

    A positive difference means the candidate has a higher MAE than the
    reference. Five folds provide limited inferential power, so the interval is
    reported as a descriptive uncertainty diagnostic, not a hypothesis test.
    """
    if reference_model not in fold_details:
        raise ValueError(f"Unknown reference model: {reference_model}")
    reference = fold_details[reference_model].set_index("fold")["mae"]
    random = np.random.default_rng(random_state)
    rows: list[dict[str, Any]] = []
    for model_name, detail in fold_details.items():
        if model_name == reference_model:
            continue
        candidate = detail.set_index("fold")["mae"]
        paired = candidate.to_frame("candidate").join(reference.rename("reference"))
        if paired.isna().any().any() or len(paired) != len(reference):
            raise ValueError(f"Fold alignment differs for model: {model_name}")
        differences = (paired["candidate"] - paired["reference"]).to_numpy()
        samples = random.choice(
            differences,
            size=(bootstrap_samples, len(differences)),
            replace=True,
        ).mean(axis=1)
        rows.append(
            {
                "reference_model": reference_model,
                "candidate_model": model_name,
                "folds": len(differences),
                "mean_mae_difference": float(differences.mean()),
                "ci95_lower": float(np.quantile(samples, 0.025)),
                "ci95_upper": float(np.quantile(samples, 0.975)),
                "candidate_worse_folds": int((differences > 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_mae_difference").reset_index(drop=True)


def evaluate_holdout_strategies(
    estimator: RegressorMixin,
    X: pd.DataFrame,
    y: pd.Series,
    splits: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Fit one reference model on competing holdout strategies."""
    rows: list[dict[str, Any]] = []
    for strategy, (train_index, validation_index) in splits.items():
        model = clone(estimator)
        start = time.perf_counter()
        model.fit(X.iloc[train_index], y.iloc[train_index])
        fit_seconds = time.perf_counter() - start
        train_prediction = np.clip(model.predict(X.iloc[train_index]), 0.0, 1.0)
        start = time.perf_counter()
        validation_prediction = np.clip(
            model.predict(X.iloc[validation_index]),
            0.0,
            1.0,
        )
        predict_seconds = time.perf_counter() - start
        train_metrics = regression_metrics(y.iloc[train_index], train_prediction)
        validation_metrics = regression_metrics(
            y.iloc[validation_index],
            validation_prediction,
        )
        rows.append(
            {
                "strategy": strategy,
                "train_rows": len(train_index),
                "validation_rows": len(validation_index),
                "train_mae": train_metrics["mae"],
                **validation_metrics,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def evaluate_feature_sets(
    estimator: RegressorMixin,
    X: pd.DataFrame,
    y: pd.Series,
    folds: FoldIndices,
    feature_sets: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Evaluate an ordered, progressive feature-family ablation."""
    rows: list[dict[str, Any]] = []
    for stage, features in feature_sets.items():
        missing = set(features).difference(X.columns)
        if missing:
            raise ValueError(f"Ablation stage '{stage}' is missing: {sorted(missing)}")
        summary, _, _ = cross_validate_model(
            stage,
            estimator,
            X.loc[:, list(features)],
            y,
            folds,
        )
        rows.append({"stage": stage, **summary})
    result = pd.DataFrame(rows)
    result["mae_gain_vs_previous"] = -result["mae"].diff()
    return result


def randomized_model_search(
    estimator: RegressorMixin,
    X: pd.DataFrame,
    y: pd.Series,
    folds: FoldIndices,
    parameter_space: Mapping[str, Sequence[Any]],
    *,
    n_iter: int = 8,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run reproducible ``RandomizedSearchCV`` on supplied grouped folds."""

    def clipped_negative_mae(
        fitted: RegressorMixin,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> float:
        prediction = np.clip(fitted.predict(features), 0.0, 1.0)
        return -float(mean_absolute_error(target, prediction))

    search = RandomizedSearchCV(
        estimator=clone(estimator),
        param_distributions=dict(parameter_space),
        n_iter=n_iter,
        scoring=clipped_negative_mae,
        cv=list(folds),
        random_state=random_state,
        n_jobs=1,
        refit=False,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(X, y)
    raw = pd.DataFrame(search.cv_results_)
    parameters = pd.DataFrame(raw["params"].tolist())
    results = parameters.assign(
        mae=-raw["mean_test_score"],
        mae_std=raw["std_test_score"],
        train_mae=-raw["mean_train_score"],
        fit_seconds=raw["mean_fit_time"],
        predict_seconds=raw["mean_score_time"],
    )
    results["mae_gap"] = results["mae"] - results["train_mae"]
    results = results.sort_values("mae").reset_index(drop=True)
    results.insert(0, "trial", np.arange(1, len(results) + 1))
    best_parameters = dict(raw.loc[raw["mean_test_score"].idxmax(), "params"])
    return results, best_parameters


def fit_final_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    models: Mapping[str, RegressorMixin] | None = None,
) -> RegressorMixin:
    """Fit the selected family once on all labeled rows."""
    candidates = dict(models or build_model_candidates())
    if model_name not in candidates:
        raise ValueError(f"Unknown model family: {model_name}")
    model = clone(candidates[model_name])
    model.fit(X, y)
    return model


def _runtime_versions() -> dict[str, str]:
    """Return the small dependency fingerprint needed to reuse a model."""
    packages = ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost")
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:  # pragma: no cover - core dependencies are installed
            versions[package] = "not-installed"
    return versions


def _json_compatible(value: Any) -> Any:
    """Convert estimator metadata to strict, portable JSON values."""
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_model_bundle(
    model: RegressorMixin,
    features: Sequence[str],
    benchmark: pd.DataFrame,
    *,
    output_dir: str | Path = ARTIFACT_DIR,
    model_name: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    validation_strategy: str = f"{N_SPLITS}-fold GroupKFold(gameId)",
    training_rows: int | None = None,
    postprocessing: str = "clip_[0,1]_and_snap_to_maxRank_grid",
) -> dict[str, Path]:
    """Publish one aligned model, benchmark and manifest."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "model.joblib"
    benchmark_path = destination / "model_comparison.csv"
    manifest_path = destination / "model_manifest.json"
    joblib.dump(model, model_path)
    benchmark.to_csv(benchmark_path, index=False)
    manifest = {
        "artifact_version": 2,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_name": model_name or type(model).__name__,
        "model_class": type(model).__name__,
        "target": TARGET,
        "scenario": "post_match_with_killRank",
        "primary_metric": "MAE",
        "validation_strategy": validation_strategy,
        "postprocessing": postprocessing,
        "feature_count": len(features),
        "features": list(features),
        "raw_data_sha256": raw_data_fingerprints(),
        "random_state": RANDOM_STATE,
        "training_rows": training_rows,
        "model_parameters": _json_compatible(model.get_params(deep=False)),
        "metrics": _json_compatible(dict(metrics or {})),
        "runtime": _runtime_versions(),
        "benchmark": benchmark_path.name,
    }
    manifest["model_sha256"] = sha256_file(model_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"model": model_path, "benchmark": benchmark_path, "manifest": manifest_path}


def load_model_bundle(
    output_dir: str | Path = ARTIFACT_DIR,
) -> tuple[RegressorMixin, dict[str, Any]]:
    """Load the final model and its ordered feature contract."""
    source = Path(output_dir)
    manifest = json.loads((source / "model_manifest.json").read_text(encoding="utf-8"))
    model = joblib.load(source / "model.joblib")
    if int(manifest["feature_count"]) != len(manifest["features"]):
        raise ValueError("The saved feature count and ordered schema disagree")
    expected_hash = manifest.get("model_sha256")
    if expected_hash and sha256_file(source / "model.joblib") != expected_hash:
        raise ValueError("The saved model checksum does not match the manifest")
    return model, manifest
