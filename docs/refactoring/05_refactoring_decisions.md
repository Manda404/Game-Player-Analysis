# 05 — Journal des décisions de refactoring

| ID | Problème initial | Solution retenue | Alternative écartée | Impact | Validation |
|---|---|---|---|---|---|
| D01 | Aucun dépôt Git | Archive ZIP de restauration testée | Initialiser Git sans demande explicite | Changements récupérables | `unzip -tq` sans erreur |
| D02 | Quatre couches pour un backend local unique | Neuf fichiers Python fonctionnels | Conserver ports/use cases/workflows | Lecture directe, moins d'indirection | Imports + tests |
| D03 | Trois configurations et contrats divergents | `config.py` unique | YAML/Pydantic + JSON multiples | Chemins et features visibles | Tests du contrat |
| D04 | CSV préparés devenus incompatibles | Features calculées depuis les raw à chaque exécution | Réparer les deux anciens CSV | Plus de stale processed data | Contrat train/test identique |
| D05 | Plusieurs traitements des sentinelles | `clean_ranking_sentinels` unique | Imputation générique | Convention officielle explicite | Tests conditionnels |
| D06 | Corruption de certains `gameId` | Isoler chaque ID invalide par ligne | Grouper les chaînes corrompues | Évite de fusionner de faux matchs | Test IDs dupliqués invalides |
| D07 | Splits concurrents, date invalide | `GroupKFold(gameId)` unique | Split ligne ou calendrier | Zéro fuite match | Audit des 5 folds |
| D08 | Pool manuel et faux rythmes | 15 features comportementales + `killRank` optionnel | Catalogue de ratios | Contrat compact et interprétable | Zéro cible/ID/infini |
| D09 | Quatre boucles de CV différentes | `compare_models` unique | Services spécialisés par étape | Métriques et folds identiques | Benchmark exécuté |
| D10 | Tuning gain historique 0,000066 | Paramètres sobres gelés | Random Search + Optuna | Moins de coût et de biais de sélection | Comparaison 5 folds |
| D11 | Calibration gain négligeable | Projection de grille seulement | Isotonic/linéaire | Politique simple et métier | MAE 0,06156 → 0,06111 |
| D12 | CatBoost historique non reproductible | Nouveau gagnant XGBoost | Réutiliser le binaire historique | Bundle aligné avec 16 features | Manifeste + hashes |
| D13 | 17 notebooks et ordre caché | Un notebook narratif exécuté | Notebooks par micro-étape | Livrable conforme à l'énoncé | 10/10 cellules exécutées |
| D14 | 337 tests majoritairement architecturaux | 17 tests sur risques réels | Maintenir tests de ports supprimés | Suite courte, couverture 91 % | Pytest complet |

## Régression historique assumée

Le score historique CatBoost (0,06037) était lié à un ancien hash préparé et à
un schéma incompatible. Le nouveau score brut (0,06156) est inférieur en
performance de 0,00119 de MAE, soit environ 1,98 %, mais il est entièrement
reproductible depuis les données brutes, utilise un contrat plus honnête et
produit un bundle cohérent. Après projection, la MAE est 0,06111.

Cette différence est documentée plutôt que masquée. Réintroduire
`killRankPercentile` ou un tuning coûteux uniquement pour retrouver le chiffre
historique irait contre la règle de simplicité et la séparation des scénarios.
