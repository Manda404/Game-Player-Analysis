"""Build the narrative notebook from version-controlled cell sources."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "game_player_analysis.ipynb"


def markdown(source: str):
    """Create one stripped Markdown cell."""
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    """Create one stripped code cell."""
    return nbf.v4.new_code_cell(source.strip())


def build_notebook() -> None:
    """Write the complete, reproducible final notebook."""
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    notebook["cells"] = [
        markdown(
            """
# Game Player Analysis — revue Data Science finale

**Livrable principal exécuté.** L'objectif officiel est de prédire
`winRankPercentage`, le score de classement de l'équipe du joueur normalisé
entre 0 (dernière place) et 1 (première place), à partir de statistiques
post-partie. Une ligne décrit un joueur, tandis que la cible d'équipe est
répétée sur les joueurs observés de cette équipe.

Ce notebook privilégie la justesse des interprétations et la traçabilité :
chaque résultat modélisé est out-of-fold, le `gameId` est groupé, et le fichier
test officiel de mai reste sans labels et hors de toute sélection de modèle.
"""
        ),
        markdown(
            """
## 1. Cadre officiel, métrique et questions

- **Train :** 50 000 lignes attribuées à janvier–avril 2024.
- **Test :** 5 000 lignes attribuées à mai 2024.
- **Métrique principale :** MAE, complétée par RMSE et R².
- **Question modèle :** quelle précision post-match obtient-on avec les
  variables individuelles, et quelle part provient de `killRank` ?
- **Question produit :** les comportements mobilité, combat et ressources
  restent-ils informatifs sans le classement post-match ?

Le sujet affirme que `date` est la date du match. Cette définition sera testée,
pas supposée vraie. `maxRank` définit la grille de classement déclarée ; ce
n'est ni le nombre réel de joueurs ni le nombre de lignes observées par match.
"""
        ),
        code(
            """
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "pyproject.toml").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/game-player-analysis-matplotlib")

import pandas as pd
from IPython.display import Image, Markdown, display

from game_player_analysis.inference import predict_frame
from game_player_analysis.logging_config import configure_logging
from game_player_analysis.pipeline import run_final_analysis

pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda value: f"{value:,.6f}")
configure_logging()
"""
        ),
        markdown(
            """
## 2. Exécution reproductible du pipeline

La cellule suivante repart des CSV bruts et exécute une seule chaîne commune :
validation, nettoyage, EDA, feature engineering, GroupKFold, quatre ensembles,
holdout groupé gelé, tuning borné, ablations, drift, diagnostics, modèle final,
manifeste et soumission. La seed de modélisation vaut 42 ; les empreintes
SHA-256 des données sont conservées dans le manifeste.
"""
        ),
        code(
            """
results = run_final_analysis(tuning_iterations=8)
tables = results["tables"]
display(Markdown(f"**Modèle publié : {results['winner']}**"))
"""
        ),
        markdown("## 3. Chargement, contrat de données et nettoyage"),
        code(
            """
display(tables["dataset_summary"])
display(tables["data_quality"].pivot(index="check", columns="dataset", values="rows"))
"""
        ),
        markdown(
            """
Il n'y a ni valeur manquante brute ni doublon exact. Cela ne signifie pas que
toutes les valeurs sont sémantiquement disponibles : `rankPts=-1` est une
sentinelle documentée ; `killPts=0` et `winPts=0` sont manquants lorsque le
système `rankPts` est actif. Le nettoyage les convertit en valeurs absentes et
ajoute des indicateurs, sans supprimer de ligne. Ces scores ne sont pas retenus
dans le modèle final, ce qui évite de mélanger plusieurs systèmes de ranking.
"""
        ),
        code(
            """
quality = tables["data_quality"]
display(quality.loc[quality["check"].str.contains("Pts|without|invalid")])
display(Image(filename=str(results["paths"]["figure_data_quality"])))
"""
        ),
        markdown("## 4. Audit de la date et de l'échantillonnage"),
        code(
            """
display(tables["date_integrity"])
display(Image(filename=str(results["paths"]["figure_date_integrity"])))
"""
        ),
        markdown(
            """
**Conclusion de l'audit temporel.** Pour 100 % des `gameId` valides ayant
plusieurs lignes, les dates diffèrent entre joueurs ; l'étendue médiane atteint
environ 45 jours dans le train. Les dates sont donc incompatibles avec la
définition officielle au niveau ligne. La cause technique n'est pas observable
dans les fichiers : on les qualifie d'incohérentes/corrompues, sans inventer une
explication. Le temps ne devient qu'un stress test pseudo-temporel.
"""
        ),
        code(
            """
display(tables["sampling_coverage"])
display(Image(filename=str(results["paths"]["figure_sampling"])))
"""
        ),
        markdown(
            """
Seulement ~2,34 % des lignes train ont un coéquipier observé et près de 98,82 %
des groupes `(gameId, teamId)` sont des singletons. Les agrégats équipe/lobby
sont donc **rejetés** : ils mesureraient surtout le mécanisme d'échantillonnage,
pas la performance complète de l'équipe.
"""
        ),
        markdown("## 5. Drift train/test"),
        code(
            """
display(tables["distribution_shift"].head(16))
display(tables["categorical_shift"])
display(tables["adversarial_validation"])
display(tables["drift_limitations"])
display(Image(filename=str(results["paths"]["figure_drift"])))
"""
        ),
        markdown(
            """
Le drift est mesuré par PSI, KS, Wasserstein normalisée, variations de masse à
zéro, déplacements catégoriels et validation adversariale. Le PSI numérique
maximal vaut environ 0,0051, le PSI catégoriel maximal 0,0084 et le ROC AUC
adversarial 0,493 : aucune dérive matérielle n'est détectée sur le contrat de
features mesuré. Cela ne prouve pas l'identité des distributions. Sans cible
test, le drift de performance/concept reste inconnu ; avec la date incohérente,
le drift temporel n'est pas interprétable.
"""
        ),
        markdown("## 6. Analyse exploratoire et hypothèses"),
        code(
            """
display(tables["target_grid"].to_frame("valeur"))
display(tables["numeric_profile"])
display(Image(filename=str(results["paths"]["figure_target"])))
display(Image(filename=str(results["paths"]["figure_profiles"])))
display(Image(filename=str(results["paths"]["figure_correlations"])))
"""
        ),
        markdown(
            """
Hypothèses examinées : (1) la mobilité reflète fortement la progression dans
la partie ; (2) combat et ressources ajoutent une information comportementale ;
(3) `killRank`, déjà calculé après le match, doit dominer le scénario post-match ;
(4) les modes changent l'échelle des comportements. Les profils par quantiles
et leurs intervalles à 95 % montrent les relations non linéaires ; la matrice
de Spearman reste ciblée pour ne pas masquer ces effets.
"""
        ),
        markdown("## 7. KPI analytiques"),
        code('display(tables["kpi_evaluation"])'),
        markdown(
            """
Chaque ratio a une règle explicite pour les dénominateurs nuls. Les KPI
`total_distance`, mobilité/seconde et headshot ratio restent descriptifs car
ils sont redondants, utilisent `gameTime` qui n'est pas un temps de survie, ou
sont trop clairsemés. `damage_per_kill`, `combat_activity` et
`resource_activity` entrent dans l'ablation : leur maintien dépend de leur gain
mesuré, pas de leur plausibilité seule.
"""
        ),
        markdown("## 8. Features et ablation progressive"),
        code(
            """
display(tables["feature_ablation"].set_index("stage"))
display(tables["scenario_comparison"].set_index("scenario"))
display(Image(filename=str(results["paths"]["figure_ablation"])))
"""
        ),
        markdown(
            """
La mobilité produit le principal gain comportemental ; combat et ressources
améliorent ensuite modestement la MAE. L'ajout de `killRank` fait passer la MAE
d'environ 0,0927 à 0,0615 : le modèle publié est donc explicitement un modèle
**post-match**, et non une prédiction early-game. La projection sur la grille
`maxRank` améliore légèrement la MAE mais dégrade le RMSE ; elle est réservée à
la soumission, avec cette contrepartie documentée.
"""
        ),
        markdown("## 9. Validation et contrôles de fuite"),
        code(
            """
display(tables["fold_audit"])
display(tables["holdout_audit"])
display(tables["holdout_performance"])
display(tables["final_holdout_audit"])
display(tables["final_holdout_evaluation"])
display(Image(filename=str(results["paths"]["figure_validation"])))
"""
        ),
        markdown(
            """
Un holdout groupé est gelé avant la sélection. Les 40 128 lignes restantes
alimentent le GroupKFold à 5 folds ; les valeurs malformées répétées sont
groupées conservativement et aucun identifiant ne traverse une partition. Un
split aléatoire de lignes expose plus de la moitié de sa validation à des matchs
déjà vus. Le stress test Jan–mars → avril reste non chronologique puisque
`date` est incohérente. Après gel de la décision, CatBoost atteint environ
0,06080 de MAE sur les 9 872 lignes du holdout du cycle.
"""
        ),
        markdown("## 10. Justification du nombre de folds"),
        code(
            """
fold_metrics_path = PROJECT_ROOT / "artifacts" / "metrics" / "fold_count_sensitivity.csv"
fold_decision_path = PROJECT_ROOT / "artifacts" / "metrics" / "fold_count_decision.csv"
fold_figure_path = PROJECT_ROOT / "artifacts" / "figures" / "07c_fold_count_sensitivity.png"
if not (fold_metrics_path.exists() and fold_decision_path.exists() and fold_figure_path.exists()):
    from scripts.run_fold_count_study import main as run_fold_count_study

    run_fold_count_study()

fold_sensitivity = pd.read_csv(fold_metrics_path)
fold_decision = pd.read_csv(fold_decision_path)
fold_comparison = (
    fold_sensitivity.pivot(index="n_splits", columns="model", values="mae")
    .join(fold_sensitivity.loc[fold_sensitivity["model"].eq("CatBoost")].set_index("n_splits")[
        ["mae_std", "mae_standard_error", "mean_validation_rows"]
    ])
    .join(fold_decision.set_index("n_splits")[
        ["catboost_winning_folds", "fit_cost_relative_to_5_folds"]
    ])
)
display(fold_comparison)
display(Image(filename=str(fold_figure_path)))
"""
        ),
        markdown(
            """
L'étude compare `GroupKFold` avec K=3, 5, 7 et 10 sur les mêmes 40 128 lignes
de développement ; le holdout final reste fermé. CatBoost gagne XGBoost dans
les 25 folds cumulés. Sept folds affiche une MAE nominalement plus basse que
cinq (gain de **0,000189**), mais cet écart est inférieur à l'erreur standard
des folds et provient aussi d'apprentissages mécaniquement plus grands.

**Décision : 5 folds.** Chaque validation conserve environ 8 026 lignes de
matchs indépendants, la décision CatBoost est déjà stable dans les cinq folds,
et sept folds alourdit sensiblement le calcul. Dix folds double au moins la
charge mesurée, réduit la validation à environ 4 013 lignes et augmente la
dispersion. Le ratio de temps exact dépend de la machine ; il est affiché dans
la figure. Choisir K=7 uniquement parce que son score affiché est le plus faible
reviendrait à optimiser le protocole après observation des résultats.
"""
        ),
        markdown("## 11. Baselines et comparaison initiale"),
        code(
            """
display(tables["initial_model_comparison"].set_index("rank"))
display(tables["model_fold_uncertainty"])
display(tables["pre_audit_configuration_comparison"])
display(Image(filename=str(results["paths"]["figure_models"])))
"""
        ),
        markdown(
            """
Les `DummyRegressor` moyenne/médiane et Ridge constituent les baselines. Les
quatre ensembles utilisent leurs hyperparamètres d'apprentissage par défaut,
avec seulement seed, parallélisme, objectif et verbosité explicités. CatBoost
bat XGBoost d'environ 0,0020 MAE et dans les cinq folds. Le rejeu des anciennes
configurations personnalisées ne donnait à XGBoost qu'un avantage de 0,000054 :
ce pré-réglage expliquait l'inversion historique du gagnant.
"""
        ),
        markdown("## 12. Tuning borné et décision anti-surajustement"),
        code(
            """
display(tables["tuning_trials"].sort_values("mae").head(8))
display(tables["tuning_comparison"])
display(pd.Series(results["tuning_decision"], name="décision").to_frame())
display(Image(filename=str(results["paths"]["figure_tuning"])))
"""
        ),
        markdown(
            """
Huit configurations CatBoost sont testées par `RandomizedSearchCV` sur les
mêmes folds groupés. Le meilleur essai (0,061671) dégrade la configuration par
défaut (0,061448) de 0,000223 ; le seuil de gain matériel de 0,0001 n'est pas
atteint. Le tuning est rejeté avant ouverture du holdout final.
"""
        ),
        markdown("## 13. Interprétabilité SHAP et analyse d'erreurs"),
        code(
            """
display(tables["permutation_importance"].head(16))
display(tables["shap_global_importance"].head(16))
display(Image(filename=str(results["paths"]["figure_importance"])))
display(Image(filename=str(results["paths"]["figure_shap"])))
display(Image(filename=str(results["paths"]["figure_errors"])))
"""
        ),
        code(
            """
important_subgroups = tables["subgroup_errors"].query(
    "dimension in ['mode_family', 'rank_grid_size', 'target_band', 'kill_band']"
)
display(important_subgroups)
display(tables["largest_errors"].head(10))
"""
        ),
        markdown(
            """
La permutation et TreeSHAP apportent deux lectures complémentaires. La
permutation mesure la perte de MAE lorsqu'une variable est mélangée ; TreeSHAP
explique le sens et l'amplitude de la contribution de chaque variable pour
chaque prédiction. CatBoost calcule TreeSHAP nativement sur 2 000 lignes tirées
de façon déterministe parmi les 9 872 lignes du holdout, après le choix du
modèle. Le pipeline vérifie que la somme des valeurs SHAP et de la valeur
attendue reconstruit chaque prédiction brute.

Les deux diagnostics placent `killRank` au premier rang, puis la marche par
minute, `kills` et `maxRank`. Sur le panneau SHAP, une contribution positive
augmente le classement prédit et une contribution négative le diminue ; la
couleur représente une valeur faible (bleu) ou élevée (rouge) **au sein de la
variable affichée**, et non une unité comparable entre variables. Ces résultats
décrivent des dépendances prédictives post-match, pas des effets causaux.

Les modes spéciaux et les petites grilles affichent les erreurs les plus hautes,
mais leurs effectifs sont faibles : ce sont des alertes, pas des conclusions
stables. Les résidus montrent une régression vers la moyenne aux cibles élevées
et quelques erreurs extrêmes ; les cas les plus difficiles sont exportés pour
inspection reproductible.
"""
        ),
        markdown("## 14. Contrat d'inférence"),
        code(
            """
submission_check = predict_frame(
    results["test"],
    results["paths"]["model"],
)
assert list(submission_check.columns) == [
    "playerId", "teamId", "gameId", "winRankPercentage"
]
assert submission_check["winRankPercentage"].between(0, 1).all()
assert len(submission_check) == 5_000
display(submission_check.head())
display(pd.Series(results["paths"], name="chemin").tail(8).to_frame())
"""
        ),
        markdown(
            """
Le bundle impose l'ordre exact des 16 features, vérifie son SHA-256 et refuse
les colonnes officielles manquantes ou les prédictions non finies. La soumission
préserve l'ordre des 5 000 lignes et contient uniquement les trois identifiants
et la cible bornée/projetée sur la grille légale.
"""
        ),
        markdown(
            """
## 15. Conclusion, limites et prochaines étapes

CatBoost par défaut atteint une MAE GroupKFold développement d'environ
**0,06145**, puis **0,06080** avec un R² de **0,92083** sur le holdout groupé du
cycle. Sans `killRank`, la MAE comportementale est d'environ **0,09266**. La
principale conclusion n'est donc pas seulement le score : `killRank` change la
nature du cas d'usage.

Limites irréductibles : test sans cible, chronologie inutilisable, couverture
très partielle des matchs/équipes, faible support de certains modes et absence
de snapshots early-game. Pour un déploiement produit, il faut collecter une
date de match cohérente, les rosters complets, un split futur réellement
étiqueté et des variables disponibles au moment précis de la décision.
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)


if __name__ == "__main__":
    build_notebook()
