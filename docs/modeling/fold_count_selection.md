# Choix du nombre de folds

## Pourquoi cette étude était nécessaire

Le choix initial de cinq folds était un compromis standard, mais il n'était pas
encore démontré sur ce dataset. Le nombre de folds est un hyperparamètre du
protocole d'évaluation : trop peu de folds réduit le nombre de réplications et
entraîne chaque modèle sur une fraction plus faible ; trop de folds augmente le
coût, réduit la taille de chaque validation et peut rendre les scores de folds
plus volatils.

## Protocole

L'étude compare `GroupKFold` avec 3, 5, 7 et 10 folds sur les mêmes 40 128
lignes de développement. Le holdout final reste fermé. CatBoost et XGBoost
utilisent leurs paramètres d'apprentissage par défaut, les mêmes 16 features et
exactement les mêmes groupes `gameId` pour chaque valeur de K. Chaque ligne est
prédite une fois hors échantillon et aucun match n'est partagé.

## Résultats

| Folds | Train par fold | Validation par fold | CatBoost MAE | Écart-type | Erreur standard | XGBoost MAE | CatBoost gagnant | Coût relatif |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 26 752 | 13 376 | 0,061564 | 0,000698 | 0,000403 | 0,063748 | 3/3 | 0,53× |
| 5 | 32 102 | 8 026 | 0,061448 | 0,001095 | 0,000490 | 0,063448 | 5/5 | 1,00× |
| 7 | 34 395 | 5 733 | 0,061260 | 0,000873 | 0,000330 | 0,063488 | 7/7 | 1,58× |
| 10 | 36 115 | 4 013 | 0,061368 | 0,001354 | 0,000428 | 0,063318 | 10/10 | 2,14× |

Le classement est parfaitement robuste : CatBoost bat XGBoost dans chacun des
25 folds cumulés. Le score nominal de CatBoost est le plus bas avec sept folds,
mais le gain sur cinq folds n'est que de **0,000189 MAE**, inférieur à l'erreur
standard et très inférieur à la dispersion entre folds. Il ne constitue pas
une amélioration du modèle : avec sept folds, chaque apprentissage voit
mécaniquement davantage de lignes.

## Choix recommandé : 5 folds

Pour le pipeline courant, **5 folds est le meilleur compromis** :

- 80 % du développement pour apprendre et environ 8 026 lignes réellement
  indépendantes par validation ;
- décision CatBoost confirmée dans 5 folds sur 5 ;
- coût raisonnable pour le benchmark, l'ablation et le tuning imbriqué ;
- passer à 7 folds augmente fortement le coût mesuré pour un écart nominal non matériel ;
- 10 folds double au moins le coût mesuré, réduit chaque validation à environ 4 013 lignes et
  présente la plus forte dispersion CatBoost.

Les ratios de coût sont des temps observés sur la machine d'exécution et peuvent
varier selon les ressources disponibles ; ils ne sont pas utilisés comme seuil
numérique de sélection.

Sept folds serait défendable pour une estimation ponctuelle lorsque le budget
de calcul est secondaire. Il n'est pas retenu comme défaut, car sélectionner K
après avoir observé sa plus petite MAE introduirait une nouvelle optimisation
du protocole. Le holdout final ne doit pas servir à départager les valeurs de K.

La reproduction est assurée par
[`scripts/run_fold_count_study.py`](../../scripts/run_fold_count_study.py). Les
preuves sont dans
[`fold_count_sensitivity.csv`](../../artifacts/metrics/fold_count_sensitivity.csv),
[`fold_count_details.csv`](../../artifacts/metrics/fold_count_details.csv) et
[`fold_count_decision.json`](../../artifacts/metadata/fold_count_decision.json).
