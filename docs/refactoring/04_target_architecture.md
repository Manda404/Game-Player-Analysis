# 04 — Architecture cible

## Structure retenue

```text
Game-Player-Analysis/
├── data/
│   ├── raw/                         # seules sources, immuables
│   └── output/                      # soumission reproductible
├── notebooks/
│   └── game_player_analysis.ipynb   # livrable narratif principal
├── src/game_player_analysis/
│   ├── __init__.py
│   ├── config.py                    # chemins, cible, seed, contrats
│   ├── data.py                      # lecture, validation, résumé
│   ├── cleaning.py                  # sentinelles et contrôles métier
│   ├── features.py                  # unique feature engineering
│   ├── validation.py                # GroupKFold et anti-fuite
│   ├── modeling.py                  # modèles, CV, entraînement, sauvegarde
│   ├── evaluation.py                # métriques, grille, segments
│   └── visualization.py             # graphiques utilisés dans le notebook
├── scripts/
│   └── run_analysis.py              # reproduction sans interface graphique
├── tests/
│   ├── test_data.py
│   ├── test_cleaning.py
│   ├── test_features.py
│   ├── test_validation.py
│   └── test_modeling.py
├── artifacts/                       # benchmark et modèle republiés ensemble
├── docs/
│   ├── analysis/
│   ├── refactoring/
│   └── final_report.md
├── docs.old/                        # archive historique intacte
├── README.md
└── pyproject.toml
```

## Responsabilités

| Module | Responsabilité unique |
|---|---|
| `config.py` | Définir les chemins relatifs, la cible, les IDs, la seed et les deux contrats de features |
| `data.py` | Charger les CSV en préservant les IDs comme chaînes et valider le schéma |
| `cleaning.py` | Convertir les sentinelles sans supprimer de lignes ni muter l'entrée |
| `features.py` | Construire les mêmes colonnes numériques pour train et test |
| `validation.py` | Produire et auditer les folds groupés par match |
| `modeling.py` | Construire les candidats, les évaluer sur les mêmes folds et sauvegarder le gagnant |
| `evaluation.py` | Calculer une seule définition des métriques et du post-traitement |
| `visualization.py` | Fournir uniquement les figures effectivement montrées |

## Choix de simplicité

- Le nom de package `game_player_analysis` est conservé pour éviter un renommage
  sans valeur métier.
- Aucun dossier `application/domain/infrastructure/presentation` n'est gardé :
  il n'existe qu'une implémentation locale.
- Aucun CSV préparé n'est nécessaire : les transformations sont déterministes
  et rapides depuis `data/raw`.
- Aucun `utils.py` n'est créé.
- Le notebook n'implémente aucune règle métier ; il appelle les fonctions et
  explique les décisions.
- Le tuning et la calibration sont retirés du chemin principal : leurs gains
  historiques sont trop faibles pour justifier la complexité.
- Les artefacts sont republiés depuis un seul benchmark aligné, avec leurs
  hashes et leur liste de features.

## Flux final

```text
CSV bruts
  → validation du schéma
  → nettoyage des sentinelles
  → features déterministes
  → GroupKFold(gameId)
  → baseline + RF + XGBoost + LightGBM + CatBoost
  → métriques et erreurs
  → modèle gagnant + manifeste + soumission
```

Cette architecture garde toutes les règles critiques tout en ramenant chaque
concept à un seul emplacement lisible par l'évaluateur.
