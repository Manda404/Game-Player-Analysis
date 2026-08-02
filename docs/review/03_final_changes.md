# Changements finaux après autocritique

## Problèmes corrigés

- collision des `gameId` malformés : regroupement conservateur et audit des IDs
  bruts ;
- validation : comparaison aléatoire, groupée, pseudo-temporelle naïve et
  purgée ;
- analyse : qualité, date, couverture, KPI, profils avec IC, dérive et grille ;
- features : ablation progressive des familles ;
- modèles : deux baselines constantes, Ridge, quatre ensembles, tuning borné ;
- robustesse : stabilité des folds, écart train-validation, sous-groupes,
  erreurs extrêmes, permutation et TreeSHAP sur holdout ;
- publication : manifeste strict JSON, checksum, versions, schéma ordonné ;
- inférence : CSV brut validé vers soumission, CLI et tests ;
- reproductibilité : groupe Poetry dev valide, logging, 23 tests, notebook
  entièrement exécuté ;
- documentation : suppression des rapports actifs redondants et remplacement
  par des pages reliées aux artefacts.

## Décisions rejetées

- agrégats équipe/lobby, faute de couverture ;
- validation temporelle comme protocole principal, faute de date cohérente ;
- tuning XGBoost, faute de gain ;
- KPI redondants ou trop clairsemés comme features ;
- affirmation d'un gagnant robuste entre XGBoost, LightGBM et CatBoost ;
- qualification early-game du scénario sans `killRank`.

## État vérifié

- notebook : 15/15 cellules de code exécutées, zéro erreur ;
- tests : 28 réussis ;
- formatage/lint : Black et Flake8 réussis ;
- manifeste : JSON strict, modèle et SHA-256 alignés ;
- soumission : 5 000 lignes, ordre et bornes contrôlés.
