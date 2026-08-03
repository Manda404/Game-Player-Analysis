"""A private, task-oriented Streamlit interface for the GameLoft analysis."""

# flake8: noqa

from __future__ import annotations

import json
import sys
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app.services import (  # noqa: E402
    CATBOOST_LIMITS,
    candidate_beats_baseline,
    evaluate_reference_on_uploaded_data,
    evaluate_uploaded_catboost,
    predict_uploaded_test,
    private_data_overview,
    read_uploaded_dataset,
    search_uploaded_catboost,
    validate_uploaded_pair,
)
from game_player_analysis.config import ARTIFACT_DIR, ID_COLUMNS, TARGET  # noqa: E402
from game_player_analysis.inference import predict_frame  # noqa: E402


st.set_page_config(
    page_title="Game Player Analysis",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)


PAGES = (
    "Discover",
    "1 · Files",
    "2 · Explorer",
    "3 · Model",
    "4 · Predictions",
)

DISPLAY_NAMES = {
    "walkDist": "Walking distance",
    "rideDist": "Vehicle distance",
    "damages": "Damage dealt",
    "kills": "Kills",
    "weapons": "Weapons used",
    "gameTime": "Match duration",
    "killRank": "Kill ranking",
    TARGET: "Normalized final ranking",
}


def _inject_style() -> None:
    """Set a restrained visual hierarchy shared by every screen."""
    st.markdown(
        """
        <style>
        .stApp { background: #f8fafc; color: #172033; }
        .block-container { max-width: 1240px; padding-top: 4rem; padding-bottom: 3rem; }
        header[data-testid="stHeader"] { background: rgba(248, 250, 252, .96); }
        [data-testid="stSidebar"] { background: #10233b; }
        [data-testid="stSidebar"] * { color: #edf5ff; }
        .brand { display: flex; align-items: center; gap: .7rem; margin: .1rem 0 .95rem; }
        .brand-mark { background: #1d72b8; color: white; border-radius: 10px; padding: .3rem .52rem; font-size: 1.05rem; }
        .brand-name { color: #112640; font-size: 1.12rem; font-weight: 760; letter-spacing: -.02em; }
        .topline { color: #6b7a90; font-size: .83rem; margin-left: auto; }
        .welcome {
            padding: 2.1rem 2.3rem; border-radius: 20px;
            background: linear-gradient(125deg, #0d2846 0%, #174f7c 65%, #2c86b7 100%);
            color: white; margin: .5rem 0 1.4rem;
        }
        .welcome h1 { color: white; font-size: 2.35rem; letter-spacing: -.045em; margin: 0; }
        .welcome p { color: #d8edff; font-size: 1.04rem; line-height: 1.55; max-width: 720px; margin: .65rem 0 0; }
        .page-kicker { color: #347eb4; font-size: .75rem; font-weight: 750; letter-spacing: .1em; text-transform: uppercase; margin-top: .4rem; }
        .page-title { color: #122b46; font-size: 1.8rem; font-weight: 760; letter-spacing: -.035em; margin: .12rem 0 .35rem; }
        .page-intro { color: #53657a; font-size: 1.02rem; line-height: 1.55; max-width: 780px; margin-bottom: 1.35rem; }
        .journey-card {
            background: white; border: 1px solid #dce6f0; border-radius: 16px;
            padding: 1.1rem 1.15rem; min-height: 154px;
            box-shadow: 0 6px 18px rgba(15, 35, 60, .035);
        }
        .journey-number { color: #2878b5; font-size: .78rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
        .journey-card h3 { color: #153652; font-size: 1.07rem; margin: .34rem 0; }
        .journey-card p { color: #5b6d80; font-size: .92rem; line-height: 1.45; margin: 0; }
        .notice {
            background: #eef8f4; border: 1px solid #cfeadd; border-left: 4px solid #1c9a70;
            border-radius: 10px; padding: .85rem 1rem; color: #174f3c; margin: .75rem 0 1.1rem;
        }
        .plain-note {
            background: #eff6fc; border: 1px solid #d8e7f4; border-radius: 12px;
            color: #244766; padding: .9rem 1rem; margin: .75rem 0 1rem;
        }
        .section-title { color: #173a58; font-size: 1.28rem; font-weight: 740; margin: 1.25rem 0 .6rem; }
        .step-line { color: #6a7b8d; font-size: .9rem; margin-bottom: .85rem; }
        div[data-testid="stMetric"] {
            background: white; border: 1px solid #dfe7f0; border-radius: 12px;
            padding: .68rem .78rem;
        }
        div[data-testid="stRadio"] > div { column-gap: .45rem; row-gap: .45rem; flex-wrap: wrap; }
        div[data-testid="stRadio"] label {
            background: white; border: 1px solid #d7e2ed; border-radius: 999px;
            margin: 0; min-height: 0; padding: .36rem .7rem;
            transition: background .15s ease, border-color .15s ease;
        }
        div[data-testid="stRadio"] label:hover { background: #f0f7fc; border-color: #9bc8e7; }
        div[data-testid="stRadio"] label:has(input:checked) { background: #e5f2fb; border-color: #2878b5; }
        div[data-testid="stRadio"] label:has(input:checked) p { color: #125987; }
        div[data-testid="stRadio"] label > div:first-child { display: none; }
        div[data-testid="stRadio"] label p { font-weight: 650; font-size: .92rem; }
        .stButton > button[kind="primary"] { border-radius: 9px; font-weight: 700; }
        @media (max-width: 900px) {
            .block-container { padding-top: 3.25rem; }
            .topline { display: none; }
            div[data-testid="stRadio"] label { padding: .3rem .55rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _artifact_csv(name: str) -> pd.DataFrame | None:
    path = ARTIFACT_DIR / "metrics" / name
    return pd.read_csv(path) if path.is_file() else None


def _artifact_figure(name: str) -> Path | None:
    path = ARTIFACT_DIR / "figures" / name
    return path if path.is_file() else None


def _navigate_to(page: str) -> None:
    st.session_state["active_page"] = page


def _reset_model_session() -> None:
    """Invalidate candidate state when the private training file changes."""
    for key in (
        "active_model_result",
        "session_model_result",
        "session_search",
        "session_submission",
    ):
        st.session_state.pop(key, None)


def _page_heading(kicker: str, title: str, description: str) -> None:
    st.markdown(f"<div class='page-kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-intro'>{description}</div>", unsafe_allow_html=True)


def _number(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _format_count(value: int | float) -> str:
    return f"{int(value):,}"


def _require_train() -> pd.DataFrame | None:
    train = st.session_state.get("private_train")
    if train is None:
        st.info("Start by adding your training file in step 1.")
        st.button(
            "Add my files",
            on_click=_navigate_to,
            args=("1 · Files",),
        )
        return None
    return train


def _private_preview(frame: pd.DataFrame, *, rows: int = 12) -> pd.DataFrame:
    return frame.drop(columns=list(ID_COLUMNS), errors="ignore").head(rows)


def _outlier_overview(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in ("walkDist", "rideDist", "damages", "kills", "weapons"):
        values = frame[feature].astype(float)
        q1, q3 = values.quantile([0.25, 0.75])
        threshold = q3 + 1.5 * (q3 - q1)
        positive = values.loc[values.gt(0)]
        positive_rate = np.nan
        if len(positive) >= 4:
            p1, p3 = positive.quantile([0.25, 0.75])
            positive_rate = 100 * positive.gt(p3 + 1.5 * (p3 - p1)).mean()
        rows.append(
            {
                "Feature": DISPLAY_NAMES[feature],
                "Outliers (all values)": 100 * values.gt(threshold).mean(),
                "Outliers (> 0)": positive_rate,
                "High value (P99)": values.quantile(0.99),
            }
        )
    return pd.DataFrame(rows)


def _draw_distribution(frame: pd.DataFrame, feature: str) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 4.1))
    values = frame[feature].dropna()
    axis.hist(values, bins=45, color="#2878B5", alpha=0.88, edgecolor="white")
    axis.axvline(values.median(), color="#D9822B", linestyle="--", label="median value")
    axis.set(
        title=f"Distribution — {DISPLAY_NAMES.get(feature, feature)}",
        xlabel=DISPLAY_NAMES.get(feature, feature),
        ylabel="Number of players",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)


def _draw_target_relationship(frame: pd.DataFrame, feature: str) -> None:
    sampled = frame.sample(min(len(frame), 4_000), random_state=42)
    figure, axis = plt.subplots(figsize=(8.5, 4.1))
    axis.scatter(sampled[feature], sampled[TARGET], s=9, alpha=0.18, color="#2878B5")
    axis.set(
        title=f"Observed relationship — {DISPLAY_NAMES.get(feature, feature)}",
        xlabel=DISPLAY_NAMES.get(feature, feature),
        ylabel="Normalized final ranking",
        ylim=(0, 1),
    )
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)


def _render_home() -> None:
    st.markdown(
        """
        <div class="welcome">
          <h1>Explore a match result, simply.</h1>
          <p>A guided tour of my GameLoft Data Science project: explore game statistics,
          understand model quality, and create predictions without publishing the data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='notice'><b>Your files remain private.</b> They are read in memory only during your session. "
        "The application does not store them or call an external service.</div>",
        unsafe_allow_html=True,
    )
    st.button(
        "Start with my files",
        type="primary",
        on_click=_navigate_to,
        args=("1 · Files",),
    )
    st.markdown("<div class='section-title'>Your four-step journey</div>", unsafe_allow_html=True)
    cards = [
        (
            "Step 1",
            "Add",
            "Upload the training file, then the test file if you have it.",
        ),
        (
            "Step 2",
            "Explorer",
            "Spot useful patterns, outliers, and signals through clear charts.",
        ),
        (
            "Step 3",
            "Understand",
            "See how I checked model reliability and what influences its estimates.",
        ),
        (
            "Step 4",
            "Predict",
            "Use the reference model or adjust a few parameters, then download your file.",
        ),
    ]
    for column, (number, title, description) in zip(st.columns(4), cards):
        column.markdown(
            f"<div class='journey-card'><div class='journey-number'>{number}</div>"
            f"<h3>{title}</h3><p>{description}</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='section-title'>What the project shows, even before upload</div>",
        unsafe_allow_html=True,
    )
    holdout = _artifact_csv("final_holdout_evaluation.csv")
    if holdout is not None and not holdout.empty:
        metrics = holdout.iloc[0]
        cards = st.columns(3)
        cards[0].metric("Model mean error", _number(float(metrics["mae"]), 5))
        cards[1].metric(
            "Matches held out for final evaluation", _format_count(metrics["holdout_rows"])
        )
        cards[2].metric("Overall fit (R²)", _number(float(metrics["r2"]), 3))
    st.caption(
        "The model estimates a normalized final ranking after a match. It is not intended for live in-match prediction."
    )


def _render_uploads() -> None:
    _page_heading(
        "Step 1 · Private data",
        "Add your files with confidence.",
        "The training file enables exploration and model tuning. The test file is optional and is only used to create predictions.",
    )
    st.markdown(
        "<div class='plain-note'><b>Which file should I choose?</b> The <b>training</b> file contains the "
        "<code>winRankPercentage</code> column. The <b>test</b> file does not. Both files must be "
        "semicolon-separated CSV files.</div>",
        unsafe_allow_html=True,
    )
    train_column, test_column = st.columns(2)
    with train_column:
        st.markdown("#### Training file")
        st.caption("Required to explore the data or evaluate a model variant.")
        train_upload = st.file_uploader("Choose the training file", type="csv", key="train_upload")
        if train_upload is not None:
            try:
                payload = train_upload.getvalue()
                signature = sha256(payload).hexdigest()
                train = read_uploaded_dataset(payload, require_target=True)
                if st.session_state.get("private_train_signature") != signature:
                    _reset_model_session()
                st.session_state["private_train"] = train
                st.session_state["private_train_name"] = train_upload.name
                st.session_state["private_train_signature"] = signature
                st.success(f"File ready: {_format_count(len(train))} analyzable rows.")
            except Exception as error:
                st.error(f"This file cannot be used: {error}")
    with test_column:
        st.markdown("#### Test file — optional")
        st.caption("Add it when you want to create a prediction file.")
        test_upload = st.file_uploader("Choose the test file", type="csv", key="test_upload")
        if test_upload is not None:
            try:
                payload = test_upload.getvalue()
                signature = sha256(payload).hexdigest()
                test = read_uploaded_dataset(payload, require_target=False)
                train = st.session_state.get("private_train")
                if train is not None:
                    validate_uploaded_pair(train, test)
                if st.session_state.get("private_test_signature") != signature:
                    st.session_state.pop("session_submission", None)
                st.session_state["private_test"] = test
                st.session_state["private_test_name"] = test_upload.name
                st.session_state["private_test_signature"] = signature
                st.success(f"File ready: {_format_count(len(test))} rows to predict.")
            except Exception as error:
                st.error(f"This file cannot be used: {error}")

    train = st.session_state.get("private_train")
    if train is None:
        st.info("A clear summary will appear here once you add the training file.")
        return
    st.markdown("<div class='section-title'>Instant summary</div>", unsafe_allow_html=True)
    summary, issues = private_data_overview(train)
    cards = st.columns(4)
    cards[0].metric("Players", _format_count(summary["rows"]))
    cards[1].metric("Matches", _format_count(summary["matches"]))
    cards[2].metric("Game modes", int(summary["game_modes"]))
    cards[3].metric("Missing values", int(summary["missing_cells"]))
    with st.expander("View the preview and quality checks"):
        left, right = st.columns([1.45, 1])
        with left:
            st.caption("Preview without player or match identifiers.")
            st.dataframe(_private_preview(train), width="stretch", height=300)
        with right:
            st.caption("Items to review; no row is deleted automatically.")
            st.dataframe(issues.rename("occurrences").to_frame(), width="stretch")


def _render_exploration() -> None:
    _page_heading(
        "Step 2 · Exploration",
        "Understand what the statistics reveal.",
        "Choose a statistic: the application shows its distribution, observed relationship with ranking, and unusual values to review.",
    )
    train = _require_train()
    if train is None:
        return
    available_features = [
        feature for feature in DISPLAY_NAMES if feature in train.columns and feature != TARGET
    ]
    default_feature = (
        available_features.index("walkDist") if "walkDist" in available_features else 0
    )
    feature = st.selectbox(
        "Which statistic would you like to inspect?",
        available_features,
        index=default_feature,
        format_func=lambda value: DISPLAY_NAMES[value],
    )
    st.markdown(
        "<div class='plain-note'><b>How should I read these charts?</b> They describe the uploaded data. "
        "A visible pattern does not establish a causal relationship on its own.</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.1, 1])
    with left:
        _draw_distribution(train, feature)
    with right:
        _draw_target_relationship(train, feature)

    st.markdown(
        "<div class='section-title'>Unusual values: understand before acting</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        _outlier_overview(train).style.format(
            {
                "Outliers (all values)": "{:.2f}%",
                "Outliers (> 0)": "{:.2f}%",
                "High value (P99)": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Some statistics contain many zeros. The second measure therefore compares only positive values to avoid wrongly labeling ordinary observations as outliers."
    )


def _render_model_overview() -> None:
    _page_heading(
        "Step 3 · The model",
        "Why trust this estimate?",
        "This page separates two questions: the model's observed accuracy and the statistics that contribute most to a prediction.",
    )
    tabs = st.tabs(("Measured reliability", "What the model uses"))
    with tabs[0]:
        holdout = _artifact_csv("final_holdout_evaluation.csv")
        if holdout is not None and not holdout.empty:
            metrics = holdout.iloc[0]
            cards = st.columns(3)
            cards[0].metric("Mean absolute error (MAE)", _number(float(metrics["mae"]), 5))
            cards[1].metric("Root mean squared error (RMSE)", _number(float(metrics["rmse"]), 5))
            cards[2].metric("Overall fit (R²)", _number(float(metrics["r2"]), 3))
        st.markdown(
            "<div class='plain-note'><b>MAE is the primary measure.</b> It is the average gap between the estimated and actual rankings, on a 0-to-1 scale. "
            "The lower it is, the more accurate the estimate.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("#### How the evaluation avoids shortcuts")
        st.markdown(
            "1. All players from the same match remain together during validation.  \n"
            "2. Five validation groups compare the configurations.  \n"
            "3. Held-out matches then verify the final result."
        )
        with st.expander("View the study supporting five validation groups"):
            fold_figure = _artifact_figure("07c_fold_count_sensitivity.png")
            if fold_figure is not None:
                st.image(str(fold_figure), width="stretch")
            fold_table = _artifact_csv("fold_count_decision.csv")
            if fold_table is not None:
                st.dataframe(fold_table, width="stretch", hide_index=True)
            st.caption(
                "Five groups were selected: the observed gain from more groups is too small relative to the additional computation time."
            )
    with tabs[1]:
        st.markdown(
            "<div class='plain-note'><b>An explanation, not proof of causality.</b> The analysis below shows how variables change CatBoost estimates. "
            "It does not mean that an action mechanically causes the outcome.</div>",
            unsafe_allow_html=True,
        )
        figure = _artifact_figure("13_catboost_shap_summary.png")
        importance = _artifact_csv("shap_global_importance.csv")
        left, right = st.columns([1.55, 1])
        with left:
            if figure is not None:
                st.image(str(figure), width="stretch")
        with right:
            st.markdown("#### Reading this chart")
            st.markdown(
                "- Each point represents an analyzed player.\n"
                "- On the right, the statistic pushes the estimated ranking upward.\n"
                "- On the left, it pushes the estimate downward.\n"
                "- Color indicates a low or high value for that statistic."
            )
            st.warning(
                "`killRank` is highly informative, but it is known after the match. The model therefore describes a post-match use case, not live prediction."
            )
        if importance is not None:
            with st.expander("View the detailed feature ranking"):
                st.dataframe(
                    importance.head(12).style.format(
                        {
                            "mean_abs_shap": "{:.5f}",
                            "mean_shap": "{:.5f}",
                            "positive_shap_share": "{:.1%}",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )


def _evaluate_candidate(
    train: pd.DataFrame, parameters: dict[str, int | float]
) -> dict[str, object]:
    """Evaluate a candidate and attach a fair comparison with the active baseline."""
    result = evaluate_uploaded_catboost(train, parameters)
    active_model = st.session_state.get("active_model_result")
    if active_model is not None:
        baseline_metrics = active_model["holdout_metrics"]
        baseline_label = "the active model for this session"
    else:
        try:
            baseline_metrics = evaluate_reference_on_uploaded_data(train, ARTIFACT_DIR)
            baseline_label = "the reference model"
        except Exception as error:
            result["promotion_error"] = str(error)
            return result
    candidate_metrics = result["holdout_metrics"]
    result["baseline_metrics"] = baseline_metrics
    result["baseline_label"] = baseline_label
    result["mae_improvement"] = float(baseline_metrics["mae"]) - float(candidate_metrics["mae"])
    result["eligible_for_promotion"] = candidate_beats_baseline(
        candidate_metrics,
        baseline_metrics,
    )
    return result


def _adopt_candidate(result: dict[str, object]) -> None:
    """Switch all session predictions to the verified, better candidate."""
    st.session_state["active_model_result"] = result


def _render_training_result(result: dict[str, object]) -> None:
    st.markdown(
        "<div class='section-title'>Your configuration result</div>", unsafe_allow_html=True
    )
    summary = result["cv_summary"]
    holdout = result["holdout_metrics"]
    cards = st.columns(4)
    cards[0].metric("Validation mean error", _number(float(summary["mae"]), 5))
    cards[1].metric("Variation across groups", _number(float(summary["mae_std"]), 5))
    cards[2].metric("Final holdout error", _number(float(holdout["mae"]), 5))
    cards[3].metric("Overall fit (R²)", _number(float(holdout["r2"]), 3))
    st.caption(
        f"Configuration evaluated on {_format_count(result['development_rows'])} rows, then checked on "
        f"{_format_count(result['holdout_rows'])} rows from separate matches."
    )
    if "promotion_error" in result:
        st.warning(
            "The candidate was evaluated, but the reference model could not be compared on these files: "
            f"{result['promotion_error']}"
        )
    elif "baseline_metrics" in result:
        baseline = result["baseline_metrics"]
        improvement = float(result["mae_improvement"])
        comparison = st.columns(3)
        comparison[0].metric(
            "Candidate MAE",
            _number(float(holdout["mae"]), 5),
        )
        comparison[1].metric(
            "Comparison MAE",
            _number(float(baseline["mae"]), 5),
        )
        comparison[2].metric("MAE improvement", _number(improvement, 5))
        if result["eligible_for_promotion"]:
            st.success(
                f"This candidate performs better than {result['baseline_label']} on the same final holdout."
            )
            st.button(
                "Adopt this model for this session",
                type="primary",
                on_click=_adopt_candidate,
                args=(result,),
            )
            st.caption(
                "Once adopted, it replaces the previous model for every prediction in this session."
            )
        else:
            st.info(
                f"This candidate does not outperform {result['baseline_label']} on the same holdout. "
                "The active model is therefore not replaced."
            )
    with st.expander("View validation-group details"):
        st.dataframe(result["fold_details"], width="stretch", hide_index=True)
    export = BytesIO()
    joblib.dump(
        {
            "model": result["model"],
            "features": result["features"],
            "parameters": result["parameters"],
            "cv_summary": summary,
            "holdout_metrics": holdout,
        },
        export,
    )
    st.download_button(
        "Download this trained model",
        data=export.getvalue(),
        file_name="game_player_catboost_session.joblib",
        mime="application/octet-stream",
    )


def _render_tuning_controls(train: pd.DataFrame) -> None:
    st.caption(
        "This option is intentionally limited: it tests only CatBoost and adopts a variant only when its MAE is better on the same final holdout."
    )
    mode = st.radio(
        "How would you like to proceed?",
        ("Evaluate one configuration", "Search a few configurations"),
        horizontal=True,
    )
    if mode == "Evaluate one configuration":
        with st.form("manual_catboost"):
            fields = st.columns(3)
            parameters: dict[str, int | float] = {}
            for index, (name, options) in enumerate(CATBOOST_LIMITS.items()):
                parameters[name] = fields[index % 3].selectbox(
                    name.replace("_", " ").capitalize(),
                    options,
                    index=min(1, len(options) - 1),
                )
            submitted = st.form_submit_button("Evaluate this configuration", type="primary")
        if submitted:
            with st.spinner("Evaluating on separate matches…"):
                result = _evaluate_candidate(train, parameters)
            st.session_state["session_model_result"] = result
            st.success(
                "Configuration evaluated. The result is compared with the currently active model."
            )
    else:
        trials = st.select_slider(
            "Maximum number of configurations to test", options=(1, 2, 4), value=2
        )
        if st.button("Run the limited search", type="primary"):
            with st.spinner("Searching and evaluating…"):
                search, parameters = search_uploaded_catboost(train, n_trials=trials)
                result = _evaluate_candidate(train, parameters)
            st.session_state["session_model_result"] = result
            st.session_state["session_search"] = search
            st.success(
                "The best configuration found was checked and compared with the active model."
            )
    if "session_search" in st.session_state:
        with st.expander("View the compared configurations"):
            st.dataframe(st.session_state["session_search"], width="stretch", hide_index=True)
    result = st.session_state.get("session_model_result")
    if result is not None:
        _render_training_result(result)


def _render_predictions() -> None:
    _page_heading(
        "Step 4 · Predictions",
        "Create your prediction file.",
        "Use the reference model or a variant you evaluated. The result is a CSV ready to download.",
    )
    test = st.session_state.get("private_test")
    if test is None:
        st.markdown(
            "<div class='plain-note'><b>The test file is missing.</b> Add a CSV without the "
            "<code>winRankPercentage</code> column in step 1, then return here.</div>",
            unsafe_allow_html=True,
        )
        st.button(
            "Add my test file",
            on_click=_navigate_to,
            args=("1 · Files",),
        )
    else:
        selected_model = st.session_state.get("active_model_result")
        source = "your adopted model for this session" if selected_model else "the reference model"
        st.success(f"Test file loaded: {_format_count(len(test))} rows. Source used: {source}.")
        if st.button("Create predictions", type="primary"):
            try:
                with st.spinner("Creating predictions…"):
                    if selected_model:
                        submission = predict_uploaded_test(
                            test,
                            selected_model["model"],
                            selected_model["features"],
                        )
                    else:
                        submission = predict_frame(test, ARTIFACT_DIR / "model.joblib")
                st.session_state["session_submission"] = submission
                st.success(f"{_format_count(len(submission))} valid predictions were created.")
            except Exception as error:
                st.error(f"Prediction failed: {error}")
        submission = st.session_state.get("session_submission")
        if submission is not None:
            with st.expander("Preview the first 20 predictions"):
                st.dataframe(submission.head(20), width="stretch", hide_index=True)
            st.download_button(
                "Download the prediction CSV",
                data=submission.to_csv(index=False).encode("utf-8"),
                file_name="game_player_submission.csv",
                mime="text/csv",
                type="primary",
            )

    st.markdown(
        "<div class='section-title'>Tune the model — optional</div>", unsafe_allow_html=True
    )
    with st.expander("I want to test a CatBoost variant"):
        train = _require_train()
        if train is not None:
            _render_tuning_controls(train)


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Your session")
        train = st.session_state.get("private_train")
        test = st.session_state.get("private_test")
        st.caption("Files are not stored.")
        if train is None:
            st.info("Training file not added")
        else:
            st.success(f"Training: {_format_count(len(train))} rows")
        if test is None:
            st.info("Test file not added")
        else:
            st.success(f"Test: {_format_count(len(test))} rows")
        if st.session_state.get("active_model_result") is not None:
            st.success("CatBoost variant adopted")
        st.divider()
        st.markdown("#### What the application does")
        st.caption(
            "• analyzes files in memory\n\n"
            "• keeps matches separate to evaluate the model\n\n"
            "• limits advanced tuning\n\n"
            "• contacts no external API"
        )


def main() -> None:
    _inject_style()
    _render_sidebar()
    st.markdown(
        "<div class='brand'><span class='brand-mark'>🎮</span><span class='brand-name'>Game Player Analysis</span>"
        "<span class='topline'>Data Science project · Post-match analysis</span></div>",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation principale",
        PAGES,
        key="active_page",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.divider()
    pages = {
        "Discover": _render_home,
        "1 · Files": _render_uploads,
        "2 · Explorer": _render_exploration,
        "3 · Model": _render_model_overview,
        "4 · Predictions": _render_predictions,
    }
    pages[page]()


if __name__ == "__main__":
    main()
