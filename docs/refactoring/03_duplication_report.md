# 03 — Rapport de duplication

## Duplication exacte

| Emplacements | Logique dupliquée | Référence à conserver | Risque | Test requis |
|---|---|---|---|---|
| `reports/.../numeric_pair_correlations.csv` et `feature_pair_correlations.csv` | Fichier strictement identique | Synthèse analytique finale | Deux noms pour une même preuve | Valeurs clés citées dans la documentation |
| `reports/.../train_test_numeric_drift.csv` et `train_test_drift.csv` | Fichier strictement identique | Synthèse analytique finale | Confusion de provenance | Contrôle de dérive unique |
| cellules de chargement EDA 02.3/02.4 | Même code | Notebook final | Multiplication des points d'entrée | Exécution intégrale du notebook |
| cellules de chargement EDA 02.7/02.8 | Même code | Notebook final | Même règle recopiée | Exécution intégrale du notebook |

## Duplication logique

| Emplacements | Logique | Version retenue | Fichiers à modifier | Risque de consolidation |
|---|---|---|---|---|
| `domain.entities`, `eda.py`, `feature_engineering.py` | Ratios combat/mobilité | Fonctions vectorisées de `features.py` | Suppression des trois variantes | Convention zéro différente |
| `domain.entities` et `postprocessing.py` | Projection sur grille | `evaluation.snap_to_rank_grid` | Fusion | Cas `maxRank<=1` |
| trois scripts analytiques | `save_json`, `save_csv`, hash, Cramér V, eta², PSI/VIF | Documentation figée + fonctions finales nécessaires seulement | Suppression scripts | Perte de reproductibilité détaillée, couverte par archive |
| `model_evaluation.py`, `feature_selection.py`, `production_evaluation.py`, `hyperparameter_tuning.py` | Boucles GroupKFold et MAE | `modeling.compare_models` | Fusion | Folds ou clipping différents |
| `model_artifacts.py`, `selection_artifacts.py`, `workflow_artifacts.py` | Persistance et validation JSON | Manifeste unique | Fusion | Rupture de compatibilité historique voulue |
| `game_player_workflow.py`, `production_model_workflow.py`, cas d'usage | Orchestration | Appels directs dans notebook/script | Suppression wrappers | Imports à mettre à jour |
| `drift.py` | PSI, KS, Evidently | Comparaison descriptive légère | Refactor | Seuils à documenter |
| `feature_sets.py` et notebooks 03/04 | Listes de features | Constantes documentées dans `features.py` | Fusion | Changement de contrat volontaire |

## Duplication documentaire

- 27 documents racontent plusieurs fois dimensions, sentinelles, mobilité,
  cible et validation ;
- deux fichiers portent le numéro 10, deux le numéro 11 et deux le numéro 12 ;
- le README, le rapport LaTeX et plusieurs notebooks présentent comme courant
  un bundle désormais obsolète ;
- les neuf EDA séparent bien les questions scientifiques, mais sont trop longs
  pour le livrable principal.

Référence retenue : trois documents sous `docs/analysis/`, un notebook final et
un rapport final. `docs.old/` et l'archive de restauration conservent l'histoire.

## Duplication architecturale

La chaîne actuelle est souvent :

```text
notebook → workflow → use case → protocol → service → fonction pandas/sklearn
```

Pour un seul backend local, les quatre niveaux intermédiaires n'apportent ni
substitution réelle ni test statistique supplémentaire. La cible devient :

```text
notebook → fonction métier testée
```

## Risques et contrôles

| Risque | Contrôle |
|---|---|
| Changer silencieusement les sentinelles | Tests dédiés `rankPts`, `killPts`, `winPts` |
| Introduire des infinis | Test de toutes les features sur dénominateurs nuls |
| Mélanger les matchs | Audit de chaque fold |
| Faire entrer la cible ou un ID | Test de contrat de colonnes |
| Divergence train/test | Test d'ordre et d'identité des colonnes |
| Perdre un résultat historique | Archive ZIP testée + journal des suppressions |
| Régression non expliquée | Nouveau tableau MAE/RMSE/R² et comparaison au benchmark historique |
