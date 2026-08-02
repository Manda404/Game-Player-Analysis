"""Command-line entry point for the validated inference contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from game_player_analysis.config import ARTIFACT_DIR, OUTPUT_DIR
from game_player_analysis.inference import predict_from_csv
from game_player_analysis.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse explicit input, model and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ARTIFACT_DIR / "model.joblib",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_DIR / "submission.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Execute validated CSV inference."""
    arguments = parse_args()
    configure_logging()
    predict_from_csv(
        arguments.input_path,
        arguments.model_path,
        arguments.output_path,
    )


if __name__ == "__main__":
    main()
