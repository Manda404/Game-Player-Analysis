# Interprétabilité

L'importance par permutation est calculée sur un holdout groupé par match, avec
cinq répétitions et un scorer MAE qui borne les prédictions.

| Feature | Hausse moyenne de MAE |
|---|---:|
| `killRank` | 0,26568 |
| marche/minute de match | 0,08723 |
| `kills` | 0,07518 |
| `maxRank` | 0,05424 |
| `walkDist` | 0,01620 |

Les modes et les ratios apportent ensuite des incréments plus faibles. Le
classement obtenu n'est ni causal ni une mesure de valeur produit ; les
features corrélées peuvent se partager l'importance.

Voir [`permutation_importance.csv`](../../artifacts/metrics/permutation_importance.csv)
et [`12_permutation_importance.png`](../../artifacts/figures/12_permutation_importance.png).
