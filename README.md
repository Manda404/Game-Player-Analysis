# Game Player Analysis

Projet de Data Science pour prédire `winRankPercentage`, le classement final
normalisé d'un joueur de Battle Royale (`0` dernière place, `1` victoire).

Le livrable principal est le notebook exécuté
[`notebooks/game_player_analysis.ipynb`](notebooks/game_player_analysis.ipynb).
Il présente Data Cleaning, Analysis & Visualization, Feature Engineering,
Validation, Modeling et Evaluation dans un seul chemin narratif.

## Données et pièges principaux

- train : 50 000 lignes, 30 colonnes ; test : 5 000 lignes, 29 colonnes ;
- une ligne = un joueur après la partie ;
- `rankPts=-1` et certains zéros de `killPts`/`winPts` sont des valeurs absentes ;
- `teamId` est interprété dans le contexte de `gameId` ;
- la pseudo-date n'est pas un timestamp fiable de match ;
- les rosters sont incomplets, donc aucun agrégat équipe/lobby n'est utilisé ;
- les cibles moyennes par famille de mode restent proches (0,461 à 0,486) ;
- `killRank` est réservé au scénario explicitement post-match.

## Approche

```text
raw CSV → validation → sentinelles → 16 features post-match
        → GroupKFold(gameId) → 4 modèles + baseline
        → erreurs/ablation → modèle + manifeste → soumission
```

Le contrat comportemental contient 15 features. Le contrat post-match ajoute
uniquement `killRank`. Les IDs, la cible, la date, les scores externes et les
agrégats de groupe sont exclus.

## Résultats

| Modèle | MAE | RMSE | R² |
|---|---:|---:|---:|
| XGBoost | **0,06156** | 0,08666 | 0,92053 |
| LightGBM | 0,06173 | 0,08671 | 0,92043 |
| CatBoost | 0,06177 | 0,08667 | 0,92051 |
| Random Forest | 0,06555 | 0,09263 | 0,90920 |
| Médiane | 0,26799 | 0,30781 | -0,00244 |

La projection sur la grille définie par `maxRank` ramène la MAE XGBoost à
**0,06111**. Sans `killRank`, la MAE vaut **0,09335**.

## Architecture

```text
src/game_player_analysis/
├── config.py         chemins, seed et contrats
├── data.py           lecture et contrôles
├── cleaning.py       sentinelles métier
├── features.py       feature engineering unique
├── validation.py     folds groupés et audit
├── modeling.py       modèles, CV et bundle
├── evaluation.py     métriques et grille
└── visualization.py figures du notebook
```

## Installation

Python 3.11 à 3.14 et Poetry :

```bash
poetry install
```

Les datasets ne sont pas versionnés. Après le clonage, placer les fichiers
officiels aux emplacements suivants :

```text
data/raw/train.csv
data/raw/test.csv
```

## Exécution

Notebook complet :

```bash
poetry run jupyter nbconvert \
  --to notebook --execute --inplace \
  notebooks/game_player_analysis.ipynb \
  --ExecutePreprocessor.timeout=1800
```

Pipeline sans interface graphique :

```bash
poetry run python scripts/run_analysis.py
```

Tests et qualité :

```bash
poetry run pytest
poetry run black --check src tests scripts
poetry run flake8 src tests scripts
```

## Sorties

- `artifacts/model.joblib` et `model_manifest.json` ;
- `artifacts/model_comparison.csv` et `killrank_ablation.csv` ;
- `data/output/submission.csv`.

Les décisions détaillées sont dans
[`docs/final_report.md`](docs/final_report.md) et
[`docs/refactoring/`](docs/refactoring/). `docs.old/` reste l'archive intacte
de la première analyse. L'état complet pré-refactoring reste récupérable dans
l'archive locale non versionnée `dist/pre_refactor_2026-08-02.zip`.

## Limites et suite

Le test n'est pas étiqueté, la date n'est pas fiable et les matchs sont
partiellement observés. Un vrai modèle early-game nécessiterait des snapshots
temporels, un roster complet, une identité joueur stable et un holdout futur.
