# Stratégie de validation

## Protocole principal de sélection

Un `GroupShuffleSplit` gèle d'abord 9 872 lignes pour la validation finale du
cycle. Les 40 128 lignes restantes servent seules à la comparaison, l'ablation
et le tuning via un `GroupKFold` à cinq folds. Le groupe conservateur dérivé de
`gameId` empêche aussi une valeur malformée répétée de traverser un split.

Chaque fold et le holdout final ont zéro groupe prudent et zéro `gameId` brut
partagés. La décision CatBoost par défaut est gelée avant l'ouverture du
holdout final.

## Pourquoi cinq folds ?

Une étude de sensibilité compare 3, 5, 7 et 10 folds avec CatBoost et XGBoost.
CatBoost gagne respectivement 3/3, 5/5, 7/7 et 10/10 folds. Sept folds affiche
une MAE nominalement meilleure de 0,000189 que cinq folds, mais alourdit
sensiblement le calcul ; cet écart est inférieur à l'incertitude des folds.
Dix folds augmente encore la charge et la dispersion. Cinq folds reste donc le
compromis retenu :
environ 8 026 lignes de validation par fold, classement stable et coût adapté
au benchmark, à l'ablation et au tuning. Voir
[`fold_count_selection.md`](fold_count_selection.md).

## Diagnostics de stratégies sur le train complet

| Split | Lignes train/validation | Jeux déjà vus | MAE CatBoost |
|---|---|---:|---:|
| aléatoire ligne | 40 000 / 10 000 | 56,64 % | 0,060949 |
| groupé | 39 901 / 10 099 | 0 % | 0,061160 |
| pseudo-temporel naïf | 37 815 / 12 185 | 54,97 % | 0,060944 |
| pseudo-temporel purgé | 29 279 / 12 185 | 0 % | 0,061277 |

Le pseudo-temporel janvier–mars vers avril est un stress test, pas une preuve
chronologique : tous les `gameId` valides multi-lignes portent plusieurs dates.
`TimeSeriesSplit` au niveau des lignes serait donc trompeur. Une vraie
validation temporelle requiert une date de match cohérente au niveau `gameId`.

## Validation finale du cycle

CatBoost par défaut atteint 0,060797 de MAE, 0,086596 de RMSE et 0,920827 de
R² sur les 9 872 lignes gelées, sans match partagé. Ce holdout est indépendant
des décisions du présent audit mais pas historiquement vierge, car des EDA
antérieures ont observé l'ensemble des lignes étiquetées. Le test officiel sans
cible reste exclu de toute décision.
