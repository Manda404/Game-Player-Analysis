"""Smoke tests for the private Streamlit interface."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "app.py"


def test_streamlit_guided_journey_renders_without_uploaded_data():
    """The guided journey remains understandable before importing a private CSV."""
    app = AppTest.from_file(str(APP_PATH))
    app.run(timeout=30)

    assert len(app.exception) == 0
    assert len(app.radio) == 1

    app.radio[0].set_value("3 · Model").run(timeout=30)
    assert len(app.exception) == 0
    assert len(app.tabs) == 2
    assert len(app.metric) >= 3

    app.radio[0].set_value("4 · Predictions").run(timeout=30)
    assert len(app.exception) == 0
    assert len(app.button) >= 1
