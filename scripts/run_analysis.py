"""Reproduce the complete final review, model and submission from raw data."""

from __future__ import annotations

from pathlib import Path

from game_player_analysis.logging_config import configure_logging
from game_player_analysis.pipeline import run_final_analysis


def run() -> dict[str, Path]:
    """Execute the pipeline and expose its published paths."""
    results = run_final_analysis()
    return results["paths"]


def main() -> None:
    """CLI entry point."""
    configure_logging(log_path="analysis.log")
    outputs = run()
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
