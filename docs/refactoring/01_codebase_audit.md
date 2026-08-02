# 01 — Audit de la codebase

## Verdict

Le projet fonctionne, mais son architecture est disproportionnée pour un test
technique livré principalement sous forme de notebook. Les 7 352 lignes de
`src/` transforment des opérations tabulaires simples en 17 ports, 18 cas
d'usage, des entités métier ligne par ligne, deux workflows et plusieurs
couches de persistance. Cette structure augmente la distance entre le
raisonnement scientifique et le code sans améliorer la validité statistique.

## Audit par fichier ou groupe cohérent

| Fichier ou groupe | Rôle actuel | Problème identifié | Duplication | Décision | Justification |
|---|---|---|---|---|---|
| `src/game_player_analysis/__init__.py` | Version et exports | Exports liés à l'ancienne architecture | Faible | REFACTOR | Exposer seulement l'API fonctionnelle finale |
| `application/interfaces.py` | Port de chargement | Interface pour un seul adaptateur | Architecturale | DELETE | Une fonction de chargement testée suffit |
| `application/modeling_interfaces.py` | 17 Protocols ML | 216 lignes de ports pour des appels locaux uniques | Architecturale | DELETE | Aucun second backend n'existe |
| `application/modeling_use_cases.py` | 18 wrappers `execute` | Délégation quasi mécanique | Architecturale | DELETE | Fusionner dans des fonctions explicites |
| `application/use_cases.py` | Conversion DataFrame vers entités | Matérialise 50 000 objets inutiles pour l'analyse tabulaire | Logique | DELETE | Pandas est l'unité réelle du projet |
| `domain/entities.py` | Entités joueur/combat/ranking | Ratios dupliqués avec feature engineering et post-traitement | Forte | MERGE | Conserver uniquement les formules utiles dans `features.py`/`evaluation.py` |
| `domain/enums.py` | Familles de modes | Classe enum pour une extraction de chaîne | Faible | MERGE | Une fonction `game_mode_family` est plus directe |
| `domain/modeling.py` | Dataclasses de contrats | Contrats riches autour de simples tuples/tables | Architecturale | DELETE | DataFrame de résultats et constantes suffisent |
| `config/__init__.py` | Pydantic + YAML | Configuration dispersée entre YAML, modèles et notebooks | Partielle | REFACTOR | Remplacer par constantes et chemins centraux simples |
| `logging_config.py` | Logging global | Écrit un log de développement pour un notebook | Faible | DELETE | Le notebook et les scripts affichent les résultats utiles |
| `infrastructure/data_loader.py` | Chargement et validation | Deux classes, conversion mutante et schéma dupliqué | Partielle | MERGE | `data.py` portera schéma, lecture et validation |
| `infrastructure/data_profile.py` | Profils de colonnes | Mélange diagnostic notebook et code réutilisable | Partielle | MERGE | Garder un résumé compact dans `data.py` |
| `infrastructure/eda.py` | 12 fonctions EDA | Catalogue plus large que le notebook final | Forte | MERGE | Conserver seulement contrôles décisionnels |
| `infrastructure/feature_engineering.py` | Exploration + production | 352 lignes, deux contrats, faux « per minute », encodeur inutilisé | Forte | REFACTOR | Une seule fonction déterministe dans `features.py` |
| `infrastructure/feature_sets.py` | Listes historiques | Pool manuel présenté comme issu de l'EDA | Forte | DELETE | Le contrat final sera documenté à côté de sa fonction |
| `infrastructure/feature_selection.py` | Sélection CatBoost | 464 lignes, sélection faite sur un pool incomplet | Logique | DELETE | Contrat gelé à partir des analyses et ablation explicite |
| `infrastructure/split_analysis.py` | Audit de quatre splits | Entraîne CatBoost pour démontrer une règle structurelle | Partielle | MERGE | `validation.py` garde GroupKFold et un audit de chevauchement |
| `infrastructure/model_evaluation.py` | Construction et CV de cinq modèles | Bonne base, métriques et timings incomplets | Partielle | REFACTOR | Centraliser comparaison dans `modeling.py` |
| `infrastructure/modeling_services.py` | Wrapper de comparaison/diagnostics | Double le module d'évaluation | Architecturale | MERGE | Une seule implémentation de benchmark |
| `infrastructure/model_factory.py` | Lit paramètres YAML | Une fonction de 21 lignes pour un modèle | Architecturale | DELETE | Constructeurs regroupés dans `modeling.py` |
| `infrastructure/hyperparameter_tuning.py` | Random search et Optuna | Sophistication non justifiée par un gain de 0,000066 | Forte | DELETE | Conserver les paramètres simples et reproductibles |
| `infrastructure/tuning_service.py` | Orchestration tuning | Deuxième couche autour du tuning | Architecturale | DELETE | Étape retirée du livrable principal |
| `infrastructure/reliability.py` | Calibration/conformal | Analyse rigoureuse mais secondaire | Partielle | REVIEW | Conserver la conclusion, pas la chaîne dans le notebook principal |
| `infrastructure/production_evaluation.py` | Évaluation CatBoost spécialisée | Répète CV, calibration et métriques | Forte | DELETE | Benchmark unique, métriques uniques |
| `infrastructure/production_training.py` | Publication atomique | Production-grade disproportionnée et liée au contrat obsolète | Forte | REFACTOR | Sauvegarde simple du gagnant et métadonnées alignées |
| `infrastructure/production_services.py` | Réexport | Fichier de 17 lignes sans responsabilité | Architecturale | DELETE | Inutile |
| `infrastructure/explainability.py` | SHAP/permutation OOF | 183 lignes et coût élevé pour un livrable déjà dense | Partielle | REVIEW | Garder importance native simple du gagnant si disponible |
| `infrastructure/drift.py` | PSI/KS/Evidently | Trois approches pour une conclusion déjà stable | Forte | MERGE | Conserver un contrôle train/test léger, sans Evidently |
| `infrastructure/visualization.py` | 10 helpers de graphiques | Beaucoup ne servent pas au notebook final | Partielle | REFACTOR | Quatre graphiques narratifs maximum |
| `infrastructure/postprocessing.py` | Projection sur grille | Formule utile et testée | Avec entité ranking | MOVE | Déplacer vers `evaluation.py` |
| `infrastructure/inference.py` | Charge bundle historique | Dépend d'un schéma incompatible | Forte | REFACTOR | Prédiction depuis le nouveau bundle uniquement |
| `infrastructure/model_artifacts.py` | Bundle atomique complexe | Métadonnées incohérentes et double contrat | Forte | REFACTOR | Un artefact modèle et un JSON manifeste |
| `infrastructure/*_artifacts.py`, `tabular_store.py` | Persistance générique | Trois abstractions pour JSON/CSV locaux | Architecturale | DELETE | `Path`, pandas et joblib suffisent |
| `presentation/api.py` | API de résumé | Aucun besoin d'API dans l'énoncé | Architecturale | DELETE | Le notebook est le point d'entrée demandé |
| `presentation/game_player_workflow.py` | Façade notebook | 417 lignes de composition | Architecturale | DELETE | Imports directs de fonctions explicites |
| `presentation/production_model_workflow.py` | Façade scripts | Double le workflow principal | Architecturale | DELETE | Un script `run_analysis.py` suffit |
| `presentation/workflow_context.py` | Résolution des dépendances | Conteneur manuel inutile | Architecturale | DELETE | Chemins centralisés dans `config.py` |
| `presentation/notebook_setup.py` | Ajoute `src` au path | Utile seulement sans installation | Faible | MERGE | Une petite fonction dans `__init__` n'est pas nécessaire avec installation editable |
| `scripts/analyze_raw_data_from_scratch.py` | Analyse v1, 1 229 lignes | Duplique analyses et helpers | Forte | DELETE | Résultats consolidés et archive de restauration |
| `scripts/run_lead_scientific_analysis.py` | Analyse v2, 1 564 lignes | Recouvre v1 et code package | Forte | DELETE | Résultats consolidés et archive de restauration |
| `scripts/run_consolidation_analysis.py` | Audit ciblé, 498 lignes | Importe directement le script v2 | Forte | DELETE | Conclusions transférées dans la documentation finale |
| quatre scripts ML courts | Points d'entrée spécialisés | Quatre commandes autour de deux actions | Partielle | MERGE | Un seul script reproductible avec options simples |
| huit notebooks de workflow | Étapes éclatées | Ordre caché, plusieurs non exécutés | Forte | MERGE | Un notebook narratif autonome demandé par Gameloft |
| neuf notebooks EDA | Analyses détaillées | Preuves utiles mais répétitives pour l'évaluateur | Documentaire | MERGE | Conserver conclusions dans trois documents et le notebook final |
| `config/settings.dev.yaml` | Configuration actuelle | Valeurs obsolètes et plusieurs responsabilités | Partielle | DELETE | `config.py` devient la source unique |
| `models/*` | Bundle/benchmarks historiques | Contrats et hashes incompatibles | Forte | DELETE | Republier uniquement des artefacts reproduits |
| `data/processed/*` | Données préparées | Générées par un contrat devenu ambigu | Forte | DELETE | Transformation déterministe depuis `data/raw` |
| `data/output/submission.csv` | Soumission historique | Non reproductible depuis le bundle courant | Forte | DELETE | Régénérer après le benchmark final |
| `reports/*` | Preuves analytiques détaillées | 97 fichiers, forte redondance | Documentaire | MERGE | Synthèse finale + archive récupérable |
| `docs.old/*` | Première analyse | Historique demandé | Non | KEEP | Ne jamais modifier |
| `docs/*.md` historiques | Deuxième analyse | 17 fichiers avec numérotation en doublon | Documentaire | MERGE | Trois documents analytiques finaux et journal de refactoring |
| `Gameloft-Data-Science-Overview/*` | Rapport historique | Décrit le bundle incohérent et dix features inexistantes | Documentaire | DELETE | Archive de restauration ; notebook final devient livrable principal |

## Cartographie des flux et dépendances

| Groupe audité | Entrées | Sorties | Dépendances principales | Consommateurs |
|---|---|---|---|---|
| `data.py` | CSV bruts, DataFrames | DataFrames validés, résumés, hashes | pandas, `config.py` | notebook, script, tests, `modeling.py` pour les hashes |
| `cleaning.py` | DataFrame brut | Copie enrichie des valeurs/flags de ranking | pandas, NumPy | notebook, tests |
| `features.py` | DataFrames train/test | Matrices numériques ordonnées | NumPy, pandas, `config.py`, famille de mode de `data.py` | notebook, script, tests |
| `validation.py` | lignes et `gameId` | indices de folds et audits | scikit-learn, `config.py` | notebook, script, `modeling.py`, tests |
| `modeling.py` | features, cible, folds | benchmark, OOF, modèle, bundle | quatre bibliothèques ML, `evaluation.py`, `data.py` | notebook, script, tests |
| `evaluation.py` | cible, prédictions, `maxRank` | métriques, diagnostics, soumission | NumPy, pandas, scikit-learn | notebook, script, tests |
| `visualization.py` | tables d'analyse/benchmark | quatre objets `Axes` | matplotlib, seaborn | notebook, tests |
| `scripts/run_analysis.py` | `data/raw/*` | `artifacts/*`, `data/output/submission.csv` | tous les modules métier sauf visualisation | commande de reproduction |
| notebook final | package + CSV bruts | narration exécutée et mêmes artefacts | API publique du package | évaluateur |
| ancienne architecture en quatre couches | CSV/config/artefacts | DataFrames, rapports et modèles concurrents | Pydantic, YAML, pandas, ML, wrappers internes | anciens notebooks, scripts et tests |

Les références inverses ont été vérifiées avant suppression : aucun import actif
ne cible désormais `application`, `domain`, `infrastructure`, `presentation` ou
`settings.dev.yaml`.

## Ce qui fonctionne et doit être préservé

- validation groupée par `gameId` ;
- vérification de la grille de cible définie par `maxRank` ;
- nettoyage explicite des sentinelles de ranking ;
- transformation train/test identique ;
- comparaison des familles sur les mêmes folds ;
- métriques MAE, RMSE et R² ;
- hashes des données brutes ;
- prudence sur `killRank`, les agrégats d'équipe et la pseudo-date.

## Conclusion

La simplification ne vise pas à changer ces règles. Elle vise à rendre leur
implémentation visible : une règle métier, une fonction de référence, un test
ciblé et un endroit dans le notebook où la décision est expliquée.
