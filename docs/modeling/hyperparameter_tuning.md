# Optimisation des hyperparamètres

## Candidat et protocole

CatBoost est choisi avant le tuning grâce à la comparaison initiale. Un
`RandomizedSearchCV` de huit configurations utilise exactement les cinq folds
groupés du développement, `scoring="neg_mean_absolute_error"`, `random_state=42`
et un seul job externe pour éviter la surallocation. Aucun early stopping ni
jeu de validation externe n'est injecté dans la recherche.

L'espace borné couvre : `iterations` (400, 800, 1200), `depth` (4, 6, 8),
`learning_rate` (0,03, 0,05, 0,08), `l2_leaf_reg` (1, 3, 5, 10),
`random_strength` (0,5, 1, 2) et `border_count` (64, 128, 254). Cette recherche
est volontairement petite : elle teste des compromis plausibles sans prétendre
épuiser l'espace.

## Résultat

| Configuration | MAE | Écart-type | RMSE | R² | MAE train | Écart |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost défaut | **0,061448** | 0,001095 | 0,086460 | 0,920835 | 0,055269 | 0,006180 |
| meilleur essai aléatoire | 0,061671 | 0,001222 | 0,086746 | 0,920308 | 0,057417 | 0,004254 |

Le meilleur essai utilise 1 200 itérations, profondeur 6, learning rate 0,05,
L2 10, `random_strength=2` et `border_count=254`. Il **dégrade** la MAE de
0,000223. Le seuil de promotion fixé avant lecture du holdout était un gain
minimal de 0,0001 ; le tuning est donc rejeté et le modèle plus simple est
conservé.

Preuves :
[`tuning_trials.csv`](../../artifacts/metrics/tuning_trials.csv),
[`tuning_comparison.csv`](../../artifacts/metrics/tuning_comparison.csv) et
[`tuning_decision.json`](../../artifacts/metadata/tuning_decision.json).
