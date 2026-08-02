# Analyse des erreurs

Les prédictions analysées sont out-of-fold. Les résidus moyens sont proches de
zéro, mais l'erreur augmente sur les cibles très hautes, ce qui révèle une
régression vers la moyenne.

| Famille de mode | Lignes | MAE |
|---|---:|---:|
| solo | 7 990 | 0,04722 |
| duo | 14 897 | 0,05290 |
| squad | 26 995 | 0,07058 |
| spécial | 118 | 0,10046 |

Les grilles `maxRank≤5` ont une MAE de 0,329 mais seulement 20 lignes ; ce
segment est une alerte à collecter, pas une conclusion générale. Les groupes
par cible, kills, mode, couverture observée et pseudo-mois sont disponibles
dans `subgroup_errors.csv`. Les 25 cas extrêmes sont dans `largest_errors.csv`.

Figure :
[`11_error_diagnostics.png`](../../artifacts/figures/11_error_diagnostics.png).
