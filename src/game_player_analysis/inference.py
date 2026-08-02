"""Validated raw-CSV inference using the saved feature contract."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from game_player_analysis.config import ARTIFACT_DIR, TARGET
from game_player_analysis.data import load_dataset
from game_player_analysis.evaluation import build_submission
from game_player_analysis.features import build_model_features
from game_player_analysis.modeling import load_model_bundle

logger = logging.getLogger(__name__)


def _bundle_directory(model_path: str | Path) -> Path:
    source = Path(model_path)
    return source if source.is_dir() else source.parent


def predict_frame(
    frame: pd.DataFrame,
    model_path: str | Path = ARTIFACT_DIR / "model.joblib",
) -> pd.DataFrame:
    """Validate the manifest contract and return a legal submission frame."""
    model, manifest = load_model_bundle(_bundle_directory(model_path))
    if manifest.get("target") != TARGET:
        raise ValueError("The model manifest targets a different response column")

    expected = list(manifest["features"])
    include_kill_rank = "killRank" in expected
    features = build_model_features(frame, include_kill_rank=include_kill_rank)
    if list(features.columns) != expected:
        raise ValueError(
            "Inference feature order differs from the saved contract: "
            f"expected {expected}, got {list(features.columns)}"
        )

    prediction = np.asarray(model.predict(features), dtype=float)
    if prediction.shape != (len(frame),) or not np.isfinite(prediction).all():
        raise ValueError("The model produced an invalid prediction vector")
    return build_submission(frame, prediction)


def predict_from_csv(
    input_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Load a raw official-shaped CSV, predict and persist the submission."""
    logger.info("Loading inference data from %s", input_path)
    frame = load_dataset(input_path, require_target=False)
    submission = predict_frame(frame, model_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(destination, index=False)
    logger.info("Wrote %d predictions to %s", len(submission), destination)
    return destination
