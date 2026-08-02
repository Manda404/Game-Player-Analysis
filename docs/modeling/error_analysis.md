# Analyse des erreurs

Les prédictions analysées sont out-of-fold sur les 40 128 lignes de
développement. Les résidus moyens sont proches de zéro, mais l'erreur augmente
sur les cibles très hautes, ce qui révèle une régression vers la moyenne.

| Famille de mode | Lignes | MAE |
|---|---:|---:|
| solo | 6 429 | 0,04711 |
| duo | 11 866 | 0,05230 |
| squad | 21 731 | 0,07050 |
| spécial | 102 | 0,10105 |

Les grilles `maxRank≤5` ont une MAE de 0,329 mais seulement 20 lignes ; ce
segment est une alerte à collecter, pas une conclusion générale. Les groupes
par cible, kills, mode, couverture observée et pseudo-mois sont disponibles
dans `subgroup_errors.csv`. Les 25 cas extrêmes sont dans `largest_errors.csv`.

Figure :
[`11_error_diagnostics.png`](../../artifacts/figures/11_error_diagnostics.png).
