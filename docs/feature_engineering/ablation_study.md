# Étude d'ablation

Résultats :
[`feature_ablation.csv`](../../artifacts/metrics/feature_ablation.csv) et
[`08_feature_ablation.png`](../../artifacts/figures/08_feature_ablation.png).

| Étape | Nombre | MAE | Gain |
|---|---:|---:|---:|
| contexte | 4 | 0,26779 | — |
| + mobilité | 7 | 0,09907 | 0,16872 |
| + combat | 11 | 0,09616 | 0,00291 |
| + ressources | 15 | 0,09326 | 0,00289 |
| + `killRank` | 16 | 0,06160 | 0,03166 |

Le test emploie le même estimateur XGBoost et les mêmes cinq folds. Les
features combat/ressources sont maintenues parce que leur gain est reproductible
et leur sens lisible. `killRank` est maintenu uniquement dans le contrat
explicitement post-match.
