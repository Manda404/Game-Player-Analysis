# Analyse exploratoire

## Cible

`winRankPercentage` a une moyenne de 0,47233, une médiane de 0,45785 et un
écart-type de 0,30745. La cible suit presque exactement la grille définie par
`maxRank`, sans être présentée comme littéralement parfaite.

## Relations principales

Les corrélations de Spearman les plus fortes avec la cible sont `walkDist`
(0,866), `killRank` (-0,715), `upgrades` (0,681), `weapons` (0,665), `heals`
(0,564), `damages` (0,449), `rideDist` (0,433) et `kills` (0,425).

Les profils par quantiles et intervalles à 95 % sont préférés aux seuls
coefficients : ils montrent la forme non linéaire et les plateaux. Voir
[`05_feature_target_profiles.png`](../../artifacts/figures/05_feature_target_profiles.png).

## Structure et date

Tous les `gameId` valides multi-lignes portent plusieurs dates. Le span médian
train est 45,03 jours. En parallèle, seulement 2,34 % des lignes ont un
coéquipier observé. Ces deux résultats interdisent respectivement une validation
temporelle naïve et des agrégats équipe/lobby crédibles.

Le détail figure dans `date_integrity.csv`, `sampling_coverage.csv` et
`distribution_shift.csv`. Les écarts de moyenne standardisés train/test restent
faibles sur les features retenues (maximum absolu observé : 0,0235 pour
`damages`).
