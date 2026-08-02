# Game Player Analysis

Analyse Data Science reproductible du test technique Gameloft : prédire
`winRankPercentage`, le classement normalisé de l'équipe d'un joueur après une
partie de Battle Royale (`0` dernière place, `1` première place).

Le livrable principal est le notebook exécuté
[`notebooks/game_player_analysis.ipynb`](notebooks/game_player_analysis.ipynb).
Il repart des CSV bruts et présente Data Cleaning, Analysis & Visualization,
Feature Engineering, Modeling, interprétabilité, erreurs et inférence.

## Résultat vérifié

| Modèle/scénario | MAE | RMSE | R² |
|---|---:|---:|---:|
| CatBoost, holdout final groupé | **0,06080** | 0,08660 | 0,92083 |
| CatBoost, GroupKFold développement | 0,06145 ± 0,00109 | 0,08646 | 0,92083 |
| Comportement sans `killRank`, GroupKFold | 0,09266 | 0,12924 | 0,82314 |
| Ridge linéaire, GroupKFold | 0,08831 | 0,12225 | 0,84177 |
| Médiane constante, GroupKFold | 0,26788 | 0,30781 | -0,00294 |

La comparaison initiale auditée emploie les paramètres d'apprentissage par
défaut. CatBoost bat XGBoost de 0,00200 MAE en moyenne et dans les cinq folds.
L'ancien gagnant XGBoost provenait de configurations initiales personnalisées ;
leur rejeu ne lui donne qu'un avantage négligeable de 0,000054. Huit essais de
tuning CatBoost n'améliorent pas le défaut et sont rejetés.

## Points critiques

- Une ligne décrit un joueur, mais la cible est le score de son équipe répété
  sur les joueurs observés.
- `killRank` est une information post-match. Son usage interdit de présenter le
  modèle final comme une prédiction early-game.
- Le split principal est un GroupKFold à 5 folds sur `gameId`, avec zéro ID brut
  ou groupe conservateur partagé. Un holdout groupé de 9 872 lignes, gelé avant
  la sélection du présent audit, fournit l'évaluation finale du cycle.
- Une sensibilité à 3, 5, 7 et 10 folds confirme CatBoost dans les 25 folds.
  Cinq folds est conservé : environ 8 026 validations par fold ; sept folds
  coûte 52 % de plus pour seulement 0,000189 de MAE nominale.
- La colonne `date`, officiellement date du match, est incohérente : 100 % des
  `gameId` multi-lignes valides portent plusieurs dates. Le pseudo-temporel est
  seulement un stress test purgé.
- Près de 98,82 % des couples `(gameId, teamId)` sont singletons. Les agrégats
  équipe/lobby ont été rejetés comme non défendables.
- Le drift train/test est faible sur les variables mesurées : PSI numérique
  maximal 0,00509, PSI catégoriel maximal 0,00844 et validation adversariale
  ROC AUC 0,49325. Le drift de performance reste inconnu sans cible test.

## Installation

Python 3.11 à 3.14 et Poetry :

```bash
poetry install --with dev
```

Les CSV officiels ne sont pas versionnés. Les placer ici :

```text
data/raw/train.csv
data/raw/test.csv
```

## Reproduction

Notebook complet :

```bash
poetry run jupyter nbconvert \
  --to notebook --execute --inplace \
  notebooks/game_player_analysis.ipynb \
  --ExecutePreprocessor.timeout=1800
```

Pipeline en ligne de commande :

```bash
poetry run python scripts/run_analysis.py
```

## Interface Streamlit privée

L'interface propose une lecture accessible de l'analyse, l'import de CSV
privés, l'exploration, les diagnostics de validation et SHAP, un réglage
CatBoost borné ainsi que l'export de prédictions. Les fichiers téléversés ne
sont jamais écrits par l'application : ils restent en mémoire dans la session
Streamlit.

```bash
poetry run streamlit run app/app.py
```

Téléversez un train officiel séparé par `;` contenant `winRankPercentage`, puis
facultativement un test de même schéma sans cible. Le modèle est explicitement
post-match : `killRank` est requis et ne doit pas être présenté comme un signal
early-game.

Une variante CatBoost évaluée est comparée au modèle de référence sur le même
holdout groupé. Elle ne devient le modèle actif de la session que si sa MAE est
strictement plus faible ; les prédictions suivantes utilisent alors cette
variante. Cette adoption ne modifie pas le dépôt ni le modèle publié : cela
évite qu'un visiteur d'une application publique puisse écraser une version
reproductible du modèle.

## CI/CD et déploiement Streamlit

Le contrôle continu est défini dans
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) : à chaque pull request
vers `main` et après chaque push sur `main`, GitHub Actions exécute les tests,
Black et Flake8 avec Python 3.12.

Le déploiement continu est géré nativement par Streamlit Community Cloud une
fois l'application reliée au dépôt : les nouveaux commits de la branche suivie
sont pris en compte automatiquement. La configuration visuelle et la limite
d'import sont versionnées dans [`.streamlit/config.toml`](.streamlit/config.toml).

Pour créer l'application une seule fois dans Streamlit Community Cloud :

1. Dans votre espace Streamlit, cliquez sur **Create app**.
2. Choisissez `Manda404/Game-Player-Analysis`, la branche `main` et le point
   d'entrée `app/app.py`.
3. Conservez Python 3.12, puis cliquez sur **Deploy**.

L'application ne nécessite aucun secret. Les CSV des visiteurs restent en
mémoire ; ils ne sont donc ni présents dans GitHub ni envoyés vers une API.
Consultez la [documentation officielle de déploiement Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
si vous souhaitez choisir une URL personnalisée.

Inférence depuis un nouveau CSV officiel :

```bash
poetry run python scripts/predict_from_csv.py data/raw/test.csv \
  --model-path artifacts/model.joblib \
  --output-path data/output/submission.csv
```

Qualité :

```bash
poetry run pytest
poetry run black --check app src tests scripts
poetry run flake8 app src tests scripts
```

## Architecture

```text
src/game_player_analysis/
├── data.py             lecture, schéma et empreintes
├── cleaning.py         sentinelles métier
├── analysis.py         qualité, KPI et profils
├── features.py         contrat unique de 15/16 features
├── validation.py       folds, holdouts et audits de fuite
├── modeling.py         baselines, ensembles, tuning et bundle
├── evaluation.py       métriques, sous-groupes et importance
├── inference.py        CSV brut → soumission validée
├── visualization.py    figures décisionnelles, dont drift et validation
└── pipeline.py         orchestration reproductible
```

## Sorties

- modèle et contrat : `artifacts/model.joblib`, `model_manifest.json` ;
- métriques détaillées : `artifacts/metrics/` ;
- figures : `artifacts/figures/` ;
- décision de tuning : `artifacts/metadata/tuning_decision.json` ;
- décision finale : `artifacts/metadata/final_selection_decision.json` ;
- journal CLI : `artifacts/logs/analysis.log` ;
- prédictions : `data/output/submission.csv`.

Le [rapport final](docs/final_report.md) synthétise les conclusions. Les audits
de départ et la matrice de couverture sont sous [`docs/review/`](docs/review/).
L'état pré-refactoring reste récupérable dans
`dist/pre_refactor_2026-08-02.zip`.
