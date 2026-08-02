# Comparaison des modèles

Tous les modèles utilisent les mêmes 16 features, folds et bornes de prédiction.
Les résultats complets sont dans
[`model_comparison.csv`](../../artifacts/metrics/model_comparison.csv).

| Modèle | MAE | Écart-type | Train MAE | Lecture |
|---|---:|---:|---:|---|
| XGBoost | 0,06165 | 0,00063 | 0,05261 | retenu |
| LightGBM | 0,06168 | 0,00058 | 0,05531 | statistiquement proche |
| CatBoost | 0,06170 | 0,00059 | 0,05886 | statistiquement proche |
| Random Forest | 0,06542 | 0,00058 | 0,03033 | surajustement plus fort |
| Ridge | 0,08823 | 0,00086 | 0,08817 | baseline linéaire utile |
| médiane | 0,26799 | 0,00176 | 0,26798 | baseline constante |

XGBoost gagne sur la MAE mais l'écart aux deux ensembles suivants est inférieur
à la variabilité entre folds. La sélection est pragmatique et non triomphaliste.
Le bootstrap descriptif des différences appariées par fold est publié dans
`model_fold_uncertainty.csv`. Avec cinq folds seulement, ses intervalles
quantifient l'incertitude mais ne constituent pas une preuve asymptotique.
