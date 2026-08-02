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
    "Découvrir",
    "1 · Fichiers",
    "2 · Explorer",
    "3 · Modèle",
    "4 · Prédictions",
)

DISPLAY_NAMES = {
    "walkDist": "Distance parcourue à pied",
    "rideDist": "Distance en véhicule",
    "damages": "Dégâts infligés",
    "kills": "Éliminations",
    "weapons": "Armes utilisées",
    "gameTime": "Durée de la partie",
    "killRank": "Rang selon les éliminations",
    TARGET: "Classement final normalisé",
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
    return f"{value:.{digits}f}".replace(".", ",")


def _format_count(value: int | float) -> str:
    return f"{int(value):,}".replace(",", " ")


def _require_train() -> pd.DataFrame | None:
    train = st.session_state.get("private_train")
    if train is None:
        st.info("Commencez par ajouter votre fichier d'entraînement à l'étape 1.")
        st.button(
            "Ajouter mes fichiers",
            on_click=_navigate_to,
            args=("1 · Fichiers",),
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
                "Variable": DISPLAY_NAMES[feature],
                "Valeurs atypiques (toutes)": 100 * values.gt(threshold).mean(),
                "Valeurs atypiques (> 0)": positive_rate,
                "Valeur élevée (P99)": values.quantile(0.99),
            }
        )
    return pd.DataFrame(rows)


def _draw_distribution(frame: pd.DataFrame, feature: str) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 4.1))
    values = frame[feature].dropna()
    axis.hist(values, bins=45, color="#2878B5", alpha=0.88, edgecolor="white")
    axis.axvline(values.median(), color="#D9822B", linestyle="--", label="valeur médiane")
    axis.set(
        title=f"Répartition — {DISPLAY_NAMES.get(feature, feature)}",
        xlabel=DISPLAY_NAMES.get(feature, feature),
        ylabel="Nombre de joueurs",
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)


def _draw_target_relationship(frame: pd.DataFrame, feature: str) -> None:
    sampled = frame.sample(min(len(frame), 4_000), random_state=42)
    figure, axis = plt.subplots(figsize=(8.5, 4.1))
    axis.scatter(sampled[feature], sampled[TARGET], s=9, alpha=0.18, color="#2878B5")
    axis.set(
        title=f"Lien observé — {DISPLAY_NAMES.get(feature, feature)}",
        xlabel=DISPLAY_NAMES.get(feature, feature),
        ylabel="Classement final normalisé",
        ylim=(0, 1),
    )
    figure.tight_layout()
    st.pyplot(figure, clear_figure=True)


def _render_home() -> None:
    st.markdown(
        """
        <div class="welcome">
          <h1>Analysez le résultat d'une partie, simplement.</h1>
          <p>Une visite guidée de mon projet Data Science GameLoft : explorer des statistiques de jeu,
          comprendre la qualité du modèle et produire des prédictions, sans publier les données.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='notice'><b>Vos fichiers restent privés.</b> Ils sont lus uniquement en mémoire pendant votre session. "
        "L'application ne les enregistre pas et n'appelle aucun service externe.</div>",
        unsafe_allow_html=True,
    )
    st.button(
        "Commencer avec mes fichiers",
        type="primary",
        on_click=_navigate_to,
        args=("1 · Fichiers",),
    )
    st.markdown(
        "<div class='section-title'>Votre parcours en quatre étapes</div>", unsafe_allow_html=True
    )
    cards = [
        (
            "Étape 1",
            "Ajouter",
            "Déposez le fichier d'entraînement, puis le fichier test si vous l'avez.",
        ),
        (
            "Étape 2",
            "Explorer",
            "Repérez les tendances, valeurs atypiques et signaux utiles avec des graphiques lisibles.",
        ),
        (
            "Étape 3",
            "Comprendre",
            "Voyez comment la fiabilité du modèle a été contrôlée et ce qui influence ses estimations.",
        ),
        (
            "Étape 4",
            "Prédire",
            "Utilisez le modèle de référence ou ajustez quelques paramètres, puis téléchargez votre fichier.",
        ),
    ]
    for column, (number, title, description) in zip(st.columns(4), cards):
        column.markdown(
            f"<div class='journey-card'><div class='journey-number'>{number}</div>"
            f"<h3>{title}</h3><p>{description}</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='section-title'>Ce que montre le projet, même sans import</div>",
        unsafe_allow_html=True,
    )
    holdout = _artifact_csv("final_holdout_evaluation.csv")
    if holdout is not None and not holdout.empty:
        metrics = holdout.iloc[0]
        cards = st.columns(3)
        cards[0].metric("Erreur moyenne du modèle", _number(float(metrics["mae"]), 5))
        cards[1].metric(
            "Parties gardées pour le contrôle final", _format_count(metrics["holdout_rows"])
        )
        cards[2].metric("Qualité globale (R²)", _number(float(metrics["r2"]), 3))
    st.caption(
        "Le modèle estime un classement final normalisé après une partie. Il ne sert pas à prédire en direct pendant la partie."
    )


def _render_uploads() -> None:
    _page_heading(
        "Étape 1 · Données privées",
        "Ajoutez vos fichiers en toute confiance.",
        "Le fichier d'entraînement permet l'exploration et le réglage du modèle. Le fichier test est facultatif et sert uniquement à créer des prédictions.",
    )
    st.markdown(
        "<div class='plain-note'><b>Quel fichier choisir ?</b> Le fichier <b>d'entraînement</b> contient la colonne "
        "<code>winRankPercentage</code>. Le fichier <b>test</b> ne contient pas cette colonne. Les deux fichiers "
        "doivent être des CSV séparés par des points-virgules.</div>",
        unsafe_allow_html=True,
    )
    train_column, test_column = st.columns(2)
    with train_column:
        st.markdown("#### Fichier d'entraînement")
        st.caption("Indispensable pour explorer les données ou tester une variante du modèle.")
        train_upload = st.file_uploader(
            "Choisir le fichier d'entraînement", type="csv", key="train_upload"
        )
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
                st.success(f"Fichier prêt : {_format_count(len(train))} lignes analysables.")
            except Exception as error:
                st.error(f"Ce fichier ne peut pas être utilisé : {error}")
    with test_column:
        st.markdown("#### Fichier test — facultatif")
        st.caption("Ajoutez-le lorsque vous souhaitez créer un fichier de prédictions.")
        test_upload = st.file_uploader("Choisir le fichier test", type="csv", key="test_upload")
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
                st.success(f"Fichier prêt : {_format_count(len(test))} lignes à prédire.")
            except Exception as error:
                st.error(f"Ce fichier ne peut pas être utilisé : {error}")

    train = st.session_state.get("private_train")
    if train is None:
        st.info("Une fois le fichier d'entraînement ajouté, un résumé clair apparaîtra ici.")
        return
    st.markdown("<div class='section-title'>Résumé instantané</div>", unsafe_allow_html=True)
    summary, issues = private_data_overview(train)
    cards = st.columns(4)
    cards[0].metric("Joueurs", _format_count(summary["rows"]))
    cards[1].metric("Parties", _format_count(summary["matches"]))
    cards[2].metric("Modes de jeu", int(summary["game_modes"]))
    cards[3].metric("Valeurs manquantes", int(summary["missing_cells"]))
    with st.expander("Voir un aperçu et les contrôles de qualité"):
        left, right = st.columns([1.45, 1])
        with left:
            st.caption("Aperçu sans identifiants de joueur ni de partie.")
            st.dataframe(_private_preview(train), width="stretch", height=300)
        with right:
            st.caption("Points à examiner ; aucune ligne n'est supprimée automatiquement.")
            st.dataframe(issues.rename("occurrences").to_frame(), width="stretch")


def _render_exploration() -> None:
    _page_heading(
        "Étape 2 · Exploration",
        "Comprendre ce que racontent les statistiques.",
        "Choisissez une statistique : l'application montre sa répartition, son lien observé avec le classement et les valeurs inhabituelles à examiner.",
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
        "Quelle statistique souhaitez-vous regarder ?",
        available_features,
        index=default_feature,
        format_func=lambda value: DISPLAY_NAMES[value],
    )
    st.markdown(
        "<div class='plain-note'><b>Comment lire ces graphiques ?</b> Ils décrivent les données importées. "
        "Une tendance visible n'établit pas à elle seule une relation de cause à effet.</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.1, 1])
    with left:
        _draw_distribution(train, feature)
    with right:
        _draw_target_relationship(train, feature)

    st.markdown(
        "<div class='section-title'>Valeurs inhabituelles : comprendre avant d'agir</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        _outlier_overview(train).style.format(
            {
                "Valeurs atypiques (toutes)": "{:.2f}%",
                "Valeurs atypiques (> 0)": "{:.2f}%",
                "Valeur élevée (P99)": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Certaines statistiques comportent beaucoup de zéros. La seconde mesure compare alors uniquement les valeurs positives, pour éviter de qualifier abusivement des observations d'atypiques."
    )


def _render_model_overview() -> None:
    _page_heading(
        "Étape 3 · Le modèle",
        "Pourquoi faire confiance à cette estimation ?",
        "Cette page sépare deux questions : la précision observée du modèle et les statistiques qui contribuent le plus à une prédiction.",
    )
    tabs = st.tabs(("Fiabilité mesurée", "Ce que le modèle regarde"))
    with tabs[0]:
        holdout = _artifact_csv("final_holdout_evaluation.csv")
        if holdout is not None and not holdout.empty:
            metrics = holdout.iloc[0]
            cards = st.columns(3)
            cards[0].metric("Erreur moyenne (MAE)", _number(float(metrics["mae"]), 5))
            cards[1].metric("Erreur quadratique (RMSE)", _number(float(metrics["rmse"]), 5))
            cards[2].metric("Qualité globale (R²)", _number(float(metrics["r2"]), 3))
        st.markdown(
            "<div class='plain-note'><b>La mesure principale est la MAE.</b> Elle correspond à l'écart moyen entre le classement estimé et le classement réel, sur une échelle de 0 à 1. "
            "Plus elle est faible, plus l'estimation est précise.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("#### Comment l'évaluation évite les raccourcis")
        st.markdown(
            "1. Tous les joueurs d'une même partie restent ensemble pendant la validation.  \n"
            "2. Cinq groupes de validation servent à comparer les configurations.  \n"
            "3. Des parties gardées à l'écart vérifient ensuite le résultat final."
        )
        with st.expander("Voir l'étude qui justifie le choix de cinq groupes de validation"):
            fold_figure = _artifact_figure("07c_fold_count_sensitivity.png")
            if fold_figure is not None:
                st.image(str(fold_figure), width="stretch")
            fold_table = _artifact_csv("fold_count_decision.csv")
            if fold_table is not None:
                st.dataframe(fold_table, width="stretch", hide_index=True)
            st.caption(
                "Cinq groupes ont été retenus : le gain observé avec davantage de groupes est trop faible au regard du temps de calcul supplémentaire."
            )
    with tabs[1]:
        st.markdown(
            "<div class='plain-note'><b>Une explication, pas une preuve de causalité.</b> L'analyse ci-dessous montre comment les variables ont fait varier les estimations de CatBoost. "
            "Elle ne dit pas qu'une action provoque mécaniquement le résultat.</div>",
            unsafe_allow_html=True,
        )
        figure = _artifact_figure("13_catboost_shap_summary.png")
        importance = _artifact_csv("shap_global_importance.csv")
        left, right = st.columns([1.55, 1])
        with left:
            if figure is not None:
                st.image(str(figure), width="stretch")
        with right:
            st.markdown("#### Lire ce graphique")
            st.markdown(
                "- Chaque point représente un joueur analysé.\n"
                "- À droite, la statistique pousse le classement estimé vers le haut.\n"
                "- À gauche, elle le pousse vers le bas.\n"
                "- La couleur indique une valeur faible ou élevée pour cette statistique."
            )
            st.warning(
                "`killRank` est très informatif, mais il est connu après la partie. Le modèle décrit donc un usage post-match, pas une prédiction en direct."
            )
        if importance is not None:
            with st.expander("Voir le classement détaillé des statistiques"):
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
        baseline_label = "le modèle actif de cette session"
    else:
        try:
            baseline_metrics = evaluate_reference_on_uploaded_data(train, ARTIFACT_DIR)
            baseline_label = "le modèle de référence"
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
        "<div class='section-title'>Résultat de votre configuration</div>", unsafe_allow_html=True
    )
    summary = result["cv_summary"]
    holdout = result["holdout_metrics"]
    cards = st.columns(4)
    cards[0].metric("Erreur moyenne sur validation", _number(float(summary["mae"]), 5))
    cards[1].metric("Variabilité entre groupes", _number(float(summary["mae_std"]), 5))
    cards[2].metric("Erreur sur contrôle final", _number(float(holdout["mae"]), 5))
    cards[3].metric("Qualité globale (R²)", _number(float(holdout["r2"]), 3))
    st.caption(
        f"Configuration évaluée sur {_format_count(result['development_rows'])} lignes, puis contrôlée sur "
        f"{_format_count(result['holdout_rows'])} lignes de parties distinctes."
    )
    if "promotion_error" in result:
        st.warning(
            "Le candidat a été évalué, mais le modèle de référence n'a pas pu être comparé sur ces fichiers : "
            f"{result['promotion_error']}"
        )
    elif "baseline_metrics" in result:
        baseline = result["baseline_metrics"]
        improvement = float(result["mae_improvement"])
        comparison = st.columns(3)
        comparison[0].metric(
            "MAE du candidat",
            _number(float(holdout["mae"]), 5),
        )
        comparison[1].metric(
            "MAE de comparaison",
            _number(float(baseline["mae"]), 5),
        )
        comparison[2].metric("Gain de MAE", _number(improvement, 5))
        if result["eligible_for_promotion"]:
            st.success(
                f"Ce candidat est meilleur que {result['baseline_label']} sur le même contrôle final."
            )
            st.button(
                "Adopter ce modèle pour cette session",
                type="primary",
                on_click=_adopt_candidate,
                args=(result,),
            )
            st.caption(
                "Une fois adopté, il remplace le modèle précédent pour toutes les prédictions de cette session."
            )
        else:
            st.info(
                f"Ce candidat ne fait pas mieux que {result['baseline_label']} sur le même contrôle. "
                "Le modèle actif n'est donc pas remplacé."
            )
    with st.expander("Voir le détail par groupe de validation"):
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
        "Télécharger ce modèle entraîné",
        data=export.getvalue(),
        file_name="game_player_catboost_session.joblib",
        mime="application/octet-stream",
    )


def _render_tuning_controls(train: pd.DataFrame) -> None:
    st.caption(
        "Cette option est volontairement limitée : elle teste uniquement CatBoost et n'adopte une variante que si sa MAE est meilleure sur le même contrôle final."
    )
    mode = st.radio(
        "Comment souhaitez-vous procéder ?",
        ("Évaluer une configuration", "Chercher parmi quelques configurations"),
        horizontal=True,
    )
    if mode == "Évaluer une configuration":
        with st.form("manual_catboost"):
            fields = st.columns(3)
            parameters: dict[str, int | float] = {}
            for index, (name, options) in enumerate(CATBOOST_LIMITS.items()):
                parameters[name] = fields[index % 3].selectbox(
                    name.replace("_", " ").capitalize(),
                    options,
                    index=min(1, len(options) - 1),
                )
            submitted = st.form_submit_button("Évaluer cette configuration", type="primary")
        if submitted:
            with st.spinner("Évaluation en cours sur des parties distinctes…"):
                result = _evaluate_candidate(train, parameters)
            st.session_state["session_model_result"] = result
            st.success(
                "Configuration évaluée. Le résultat est comparé au modèle actuellement actif."
            )
    else:
        trials = st.select_slider(
            "Nombre maximal de configurations à tester", options=(1, 2, 4), value=2
        )
        if st.button("Lancer la recherche limitée", type="primary"):
            with st.spinner("Recherche et évaluation en cours…"):
                search, parameters = search_uploaded_catboost(train, n_trials=trials)
                result = _evaluate_candidate(train, parameters)
            st.session_state["session_model_result"] = result
            st.session_state["session_search"] = search
            st.success(
                "La meilleure configuration trouvée a été contrôlée et comparée au modèle actif."
            )
    if "session_search" in st.session_state:
        with st.expander("Voir les configurations comparées"):
            st.dataframe(st.session_state["session_search"], width="stretch", hide_index=True)
    result = st.session_state.get("session_model_result")
    if result is not None:
        _render_training_result(result)


def _render_predictions() -> None:
    _page_heading(
        "Étape 4 · Prédictions",
        "Créez votre fichier de prédictions.",
        "Utilisez le modèle de référence ou une variante que vous avez évaluée. Le résultat est un CSV prêt à être téléchargé.",
    )
    test = st.session_state.get("private_test")
    if test is None:
        st.markdown(
            "<div class='plain-note'><b>Il manque le fichier test.</b> Ajoutez un CSV sans la colonne "
            "<code>winRankPercentage</code> dans l'étape 1, puis revenez ici.</div>",
            unsafe_allow_html=True,
        )
        st.button(
            "Ajouter mon fichier test",
            on_click=_navigate_to,
            args=("1 · Fichiers",),
        )
    else:
        selected_model = st.session_state.get("active_model_result")
        source = (
            "votre modèle adopté dans cette session" if selected_model else "le modèle de référence"
        )
        st.success(
            f"Fichier test chargé : {_format_count(len(test))} lignes. Source utilisée : {source}."
        )
        if st.button("Créer les prédictions", type="primary"):
            try:
                with st.spinner("Création des prédictions…"):
                    if selected_model:
                        submission = predict_uploaded_test(
                            test,
                            selected_model["model"],
                            selected_model["features"],
                        )
                    else:
                        submission = predict_frame(test, ARTIFACT_DIR / "model.joblib")
                st.session_state["session_submission"] = submission
                st.success(f"{_format_count(len(submission))} prédictions valides ont été créées.")
            except Exception as error:
                st.error(f"Prédiction impossible : {error}")
        submission = st.session_state.get("session_submission")
        if submission is not None:
            with st.expander("Prévisualiser les 20 premières prédictions"):
                st.dataframe(submission.head(20), width="stretch", hide_index=True)
            st.download_button(
                "Télécharger le CSV de prédictions",
                data=submission.to_csv(index=False).encode("utf-8"),
                file_name="game_player_submission.csv",
                mime="text/csv",
                type="primary",
            )

    st.markdown(
        "<div class='section-title'>Ajuster le modèle — facultatif</div>", unsafe_allow_html=True
    )
    with st.expander("Je souhaite tester une variante de CatBoost"):
        train = _require_train()
        if train is not None:
            _render_tuning_controls(train)


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Votre session")
        train = st.session_state.get("private_train")
        test = st.session_state.get("private_test")
        st.caption("Les fichiers ne sont pas enregistrés.")
        if train is None:
            st.info("Fichier d'entraînement non ajouté")
        else:
            st.success(f"Entraînement : {_format_count(len(train))} lignes")
        if test is None:
            st.info("Fichier test non ajouté")
        else:
            st.success(f"Test : {_format_count(len(test))} lignes")
        if st.session_state.get("active_model_result") is not None:
            st.success("Variante CatBoost adoptée")
        st.divider()
        st.markdown("#### Ce que fait l'application")
        st.caption(
            "• analyse les fichiers en mémoire\n\n"
            "• sépare les parties pour contrôler le modèle\n\n"
            "• limite les réglages avancés\n\n"
            "• ne contacte aucune API externe"
        )


def main() -> None:
    _inject_style()
    _render_sidebar()
    st.markdown(
        "<div class='brand'><span class='brand-mark'>🎮</span><span class='brand-name'>Game Player Analysis</span>"
        "<span class='topline'>Projet Data Science · Analyse post-match</span></div>",
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
        "Découvrir": _render_home,
        "1 · Fichiers": _render_uploads,
        "2 · Explorer": _render_exploration,
        "3 · Modèle": _render_model_overview,
        "4 · Prédictions": _render_predictions,
    }
    pages[page]()


if __name__ == "__main__":
    main()
