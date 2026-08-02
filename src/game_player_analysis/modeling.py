"""Readable model construction, grouped evaluation and persistence."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from game_player_analysis.config import ARTIFACT_DIR, RANDOM_STATE
from game_player_analysis.data import raw_data_fingerprints
from game_player_analysis.evaluation import overfitting_comment, regression_metrics

FoldIndices = Sequence[tuple[np.ndarray, np.ndarray]]


def build_model_candidates(
    random_state: int = RANDOM_STATE,
) -> dict[str, RegressorMixin]:
    """Return the four tree ensembles requested by the assignment."""
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


def _baseline_evaluation(
    y: pd.Series,
    folds: FoldIndices,
    feature_count: int,
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame]:
    oof = np.full(len(y), np.nan)
    rows = []
    for fold, (train_index, validation_index) in enumerate(folds, start=1):
        median = float(y.iloc[train_index].median())
        train_prediction = np.full(len(train_index), median)
        validation_prediction = np.full(len(validation_index), median)
        oof[validation_index] = validation_prediction
        rows.append(
            {
                "fold": fold,
                "train_mae": regression_metrics(y.iloc[train_index], train_prediction)["mae"],
                **regression_metrics(y.iloc[validation_index], validation_prediction),
                "fit_seconds": 0.0,
                "predict_seconds": 0.0,
            }
        )
    detail = pd.DataFrame(rows)
    summary = {
        "model": "Median baseline",
        "mae": detail["mae"].mean(),
        "mae_std": detail["mae"].std(ddof=1),
        "rmse": detail["rmse"].mean(),
        "r2": detail["r2"].mean(),
        "train_mae": detail["train_mae"].mean(),
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
        rows.append(
            {
                "fold": fold,
                "train_mae": regression_metrics(y.iloc[train_index], train_prediction)["mae"],
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
        "rmse": float(detail["rmse"].mean()),
        "r2": float(detail["r2"].mean()),
        "train_mae": train_mae,
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
    baseline, baseline_oof, baseline_detail = _baseline_evaluation(y, folds, X.shape[1])
    summaries = [baseline]
    predictions = {"Median baseline": baseline_oof}
    details = {"Median baseline": baseline_detail}
    for name, estimator in candidates.items():
        summary, oof, detail = cross_validate_model(name, estimator, X, y, folds)
        summaries.append(summary)
        predictions[name] = oof
        details[name] = detail
    results = pd.DataFrame(summaries).sort_values("mae").reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))
    return results, predictions, details


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


def save_model_bundle(
    model: RegressorMixin,
    features: Sequence[str],
    benchmark: pd.DataFrame,
    *,
    output_dir: str | Path = ARTIFACT_DIR,
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
        "model_class": type(model).__name__,
        "feature_count": len(features),
        "features": list(features),
        "raw_data_sha256": raw_data_fingerprints(),
        "random_state": RANDOM_STATE,
        "benchmark": benchmark_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
    return model, manifest
