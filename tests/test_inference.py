"""Tests for raw-to-submission inference and its schema contract."""

import json

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from game_player_analysis.config import TARGET
from game_player_analysis.features import build_model_features
from game_player_analysis.inference import predict_frame, predict_from_csv
from game_player_analysis.modeling import save_model_bundle


def _save_test_bundle(tmp_path, player_frame):
    features = build_model_features(player_frame, include_kill_rank=True)
    model = DecisionTreeRegressor(max_depth=2, random_state=42).fit(
        features,
        player_frame[TARGET],
    )
    benchmark = pd.DataFrame([{"model": "Small tree", "mae": 0.1}])
    save_model_bundle(
        model,
        features.columns,
        benchmark,
        output_dir=tmp_path,
        model_name="Small tree",
        training_rows=len(player_frame),
        source_fingerprints={"train": "synthetic", "test": "synthetic"},
    )
    return tmp_path / "model.joblib"


def test_predict_frame_accepts_extra_raw_columns_and_preserves_order(tmp_path, player_frame):
    model_path = _save_test_bundle(tmp_path, player_frame)
    raw_test = player_frame.drop(columns=TARGET).assign(unused_extra="kept_out")
    submission = predict_frame(raw_test, model_path)

    assert list(submission.columns) == ["playerId", "teamId", "gameId", TARGET]
    assert submission["playerId"].tolist() == raw_test["playerId"].tolist()
    assert submission[TARGET].between(0, 1).all()


def test_predict_from_csv_validates_missing_columns(tmp_path, player_frame):
    model_path = _save_test_bundle(tmp_path, player_frame)
    raw_test = player_frame.drop(columns=[TARGET, "kills"])
    input_path = tmp_path / "invalid.csv"
    raw_test.to_csv(input_path, sep=";", index=False)

    with pytest.raises(ValueError, match="Missing official columns"):
        predict_from_csv(input_path, model_path, tmp_path / "submission.csv")


def test_bundle_checksum_is_enforced(tmp_path, player_frame):
    model_path = _save_test_bundle(tmp_path, player_frame)
    manifest_path = tmp_path / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        predict_frame(player_frame.drop(columns=TARGET), model_path)
