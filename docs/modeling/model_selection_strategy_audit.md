# Audit de la stratégie de sélection des modèles

## Verdict

La comparaison antérieure au présent audit n'était pas une comparaison initiale
équitable. Les quatre ensembles recevaient des hyperparamètres d'apprentissage
personnalisés avant même la sélection du gagnant. Ces réglages avantageaient
particulièrement XGBoost et dégradaient CatBoost par rapport à leurs
configurations par défaut. Le pipeline corrigé sépare désormais : baseline
naïve, comparaison initiale, sélection, tuning, validation finale et refit.

## Protocole audité

- cible : `winRankPercentage` ; métrique principale : MAE ;
- métriques secondaires : RMSE, R², stabilité et écart train-validation ;
- matrice commune : les mêmes 16 variables numériques pour tous les modèles ;
- développement : 40 128 lignes, après gel préalable d'un holdout groupé ;
- validation : 5-fold `GroupKFold` sur le groupe conservateur dérivé de
  `gameId`, mêmes indices pour tous les candidats ;
- seeds : 42 pour la comparaison et le tuning, 20260802 pour le holdout final ;
- test officiel : utilisé uniquement pour le diagnostic de drift et
  l'inférence, jamais pour la sélection ;
- early stopping : non utilisé ;
- valeurs manquantes : aucune valeur non finie dans la matrice finale ;
- catégories : les trois familles de mode sont les mêmes indicateurs numériques
  pour tous les modèles. Le pipeline n'évalue donc pas ici l'avantage spécifique
  des catégories natives de CatBoost.

La représentation commune répond à la question « quel algorithme apprend le
mieux sur le même contrat de features ? ». Une seconde étude avec pipelines
natifs par famille répondrait à une autre question et devrait être annoncée
comme telle ; elle n'est pas nécessaire pour décider le modèle publié actuel.

## Paramètres avant et après correction

| Modèle | Paramètres d'apprentissage pré-audit | Paramètres initiaux corrigés | Paramètres techniques conservés | Équitable après correction |
|---|---|---|---|---|
| Random Forest | 250 arbres, `min_samples_leaf=2`, `max_features=0.8` | défauts bibliothèque | `random_state=42`, `n_jobs=-1` | oui |
| XGBoost | 500 arbres, lr 0,05, profondeur 6, child weight 3, subsampling 0,85 | défauts bibliothèque | objectif régression, seed, parallélisme, silence | oui |
| LightGBM | 500 arbres, lr 0,05, 31 feuilles, min child 30, subsampling 0,85 | défauts bibliothèque | seed, parallélisme, silence | oui |
| CatBoost | 800 itérations, profondeur 6, lr 0,05, L2 5 | défauts bibliothèque | loss RMSE, seed, silence, aucun fichier auxiliaire | oui |

Le détail sérialisé est dans
[`model_parameter_audit.csv`](../../artifacts/metrics/model_parameter_audit.csv).
`objective`, `loss_function`, `random_state`, `n_jobs`, `verbosity` et
`allow_writing_files` sont techniques : ils ne cherchent pas à améliorer la
capacité prédictive. Profondeur, nombre d'arbres, learning rate,
régularisation, sous-échantillonnage et taille des feuilles sont des
hyperparamètres d'apprentissage et restent à leurs défauts lors de la
comparaison initiale.

## Baselines et candidats

Les seules baselines naïves sont `DummyRegressor(strategy="mean")` et
`DummyRegressor(strategy="median")`. Ridge constitue une baseline linéaire.
Random Forest, XGBoost, LightGBM et CatBoost sont des candidats initiaux non
optimisés, pas des « baseline models ».

## Défauts méthodologiques corrigés

1. Le précédent tableau intitulé comparaison initiale mélangeait sélection et
   pré-optimisation implicite.
2. Le même corpus étiqueté avait servi à de nombreuses décisions successives,
   sans holdout gelé pour le cycle final.
3. Le tuning reposait sur une boucle manuelle ; il utilise maintenant
   `RandomizedSearchCV` avec les folds groupés pré-calculés.
4. La baseline constante n'était pas matérialisée par `DummyRegressor` ; elle
   l'est désormais.
5. Le drift train/test ne reposait que sur des différences de moyennes ; PSI,
   KS, Wasserstein, changements de masse à zéro, catégories et validation
   adversariale ont été ajoutés.

## Limite sur l'indépendance finale

Le holdout de 9 872 lignes est disjoint par `gameId` et n'a été ouvert qu'après
gel de la décision dans ce cycle d'audit. Il n'est cependant pas historiquement
vierge : les analyses exploratoires antérieures avaient regardé l'ensemble des
50 000 lignes étiquetées. Le rapport l'appelle donc « holdout indépendant du
cycle d'audit », et non test totalement externe. Une validation produit exige
un futur échantillon étiqueté jamais observé.

## R² ajusté

Le R² ajusté peut compléter une comparaison de régressions linéaires emboîtées,
mais n'est pas un critère de sélection pertinent pour des arbres boostés. Avec
un grand `n`, sa pénalité par colonne est faible et le nombre de colonnes ne
représente pas la complexité effective d'un ensemble. La pertinence d'une
feature est donc décidée par ablation groupée, variation de MAE, stabilité,
importance par permutation et erreurs par sous-groupes.

## Preuves reproductibles

- [comparaison initiale](../../artifacts/metrics/initial_model_comparison.csv) ;
- [incertitude appariée par fold](../../artifacts/metrics/model_fold_uncertainty.csv) ;
- [rejeu des configurations pré-audit](../../artifacts/metrics/pre_audit_configuration_comparison.csv) ;
- [décision finale](../../artifacts/metadata/final_selection_decision.json).
