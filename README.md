# Game Player Analysis

Analyse Data Science reproductible du test technique Gameloft : prédire
`winRankPercentage`, le classement normalisé de l'équipe d'un joueur après une
partie de Battle Royale (`0` dernière place, `1` première place).

Le livrable principal est le notebook exécuté
[`notebooks/game_player_analysis.ipynb`](notebooks/game_player_analysis.ipynb).
Il repart des CSV bruts et présente Data Cleaning, Analysis & Visualization,
Feature Engineering, Modeling, interprétabilité, erreurs et inférence.

## Résultat vérifié

| Modèle/scénario | MAE GroupKFold | RMSE | R² |
|---|---:|---:|---:|
| XGBoost post-match | **0,06165** | 0,08680 | 0,92030 |
| Comportement sans `killRank` | 0,09326 | 0,13039 | 0,82010 |
| Ridge linéaire | 0,08823 | 0,12230 | 0,84175 |
| Médiane constante | 0,26799 | 0,30781 | -0,00242 |

LightGBM et CatBoost sont à moins d'un écart-type de fold de XGBoost : le
projet ne présente donc pas le gagnant comme structurellement supérieur. Six
essais de tuning n'améliorent pas la configuration figée et sont rejetés.

## Points critiques

- Une ligne décrit un joueur, mais la cible est le score de son équipe répété
  sur les joueurs observés.
- `killRank` est une information post-match. Son usage interdit de présenter le
  modèle final comme une prédiction early-game.
- Le split principal est un GroupKFold à 5 folds sur `gameId`, avec zéro ID brut
  ou groupe conservateur partagé.
- La colonne `date`, officiellement date du match, est incohérente : 100 % des
  `gameId` multi-lignes valides portent plusieurs dates. Le pseudo-temporel est
  seulement un stress test purgé.
- Près de 98,82 % des couples `(gameId, teamId)` sont singletons. Les agrégats
  équipe/lobby ont été rejetés comme non défendables.

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

Inférence depuis un nouveau CSV officiel :

```bash
poetry run python scripts/predict_from_csv.py data/raw/test.csv \
  --model-path artifacts/model.joblib \
  --output-path data/output/submission.csv
```

Qualité :

```bash
poetry run pytest
poetry run black --check src tests scripts
poetry run flake8 src tests scripts
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
├── visualization.py    12 figures décisionnelles
└── pipeline.py         orchestration reproductible
```

## Sorties

- modèle et contrat : `artifacts/model.joblib`, `model_manifest.json` ;
- métriques détaillées : `artifacts/metrics/` ;
- figures : `artifacts/figures/` ;
- décision de tuning : `artifacts/metadata/tuning_decision.json` ;
- journal CLI : `artifacts/logs/analysis.log` ;
- prédictions : `data/output/submission.csv`.

Le [rapport final](docs/final_report.md) synthétise les conclusions. Les audits
de départ et la matrice de couverture sont sous [`docs/review/`](docs/review/).
L'état pré-refactoring reste récupérable dans
`dist/pre_refactor_2026-08-02.zip`.
