# Ablation des features

L'ablation emploie CatBoost avec ses paramètres par défaut et les mêmes cinq
folds groupés du développement. Pour les ensembles non linéaires, cette
variation hors échantillon est plus informative qu'un R² ajusté.

| Étape | Variables | MAE | RMSE | R² | Gain MAE |
|---|---:|---:|---:|---:|---:|
| contexte | 4 | 0,267714 | 0,307390 | -0,000213 | — |
| + mobilité | 7 | 0,098414 | 0,134965 | 0,807144 | 0,169300 |
| + combat | 11 | 0,095515 | 0,131921 | 0,815748 | 0,002899 |
| + ressources | 15 | 0,092661 | 0,129242 | 0,823144 | 0,002854 |
| + `killRank` post-match | 16 | 0,061500 | 0,086609 | 0,920562 | 0,031161 |

Les quatre ajouts améliorent la MAE moyenne sur les mêmes folds. Mobilité
porte l'essentiel du signal comportemental. Combat et ressources apportent des
gains plus petits mais cohérents. `killRank` produit un gain majeur, tout en
changeant le scénario : le modèle final est post-match, pas early-game.

La comparaison directe des scénarios confirme 0,061448 de MAE avec `killRank`
contre 0,092661 sans cette variable. La projection sur la grille de `maxRank`
améliore la MAE à 0,060982 mais dégrade légèrement le RMSE à 0,086882 ; elle
n'est donc pas présentée comme un gain universel.

Voir
[`feature_ablation.csv`](../../artifacts/metrics/feature_ablation.csv) et
[`scenario_comparison.csv`](../../artifacts/metrics/scenario_comparison.csv).
