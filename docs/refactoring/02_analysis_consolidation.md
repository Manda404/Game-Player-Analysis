# 02 — Consolidation des deux analyses précédentes

## Sources comparées

- analyse 1 : les dix documents conservés dans `docs.old/` ;
- analyse 2 : les documents actifs `docs/01` à `docs/13`, complétés par les
  tables de `reports/lead_scientific_analysis/` et
  `reports/consolidation_analysis/` ;
- arbitre : énoncé officiel, données brutes et contrôles reproductibles.

## Continuité analyse → décision → code

| Insight analytique | Source | Statut | Implémentation avant refactoring | Action retenue |
|---|---|---|---|---|
| Une ligne représente un joueur après la partie | Officiel + analyses 1/2 | CONFIRMED | Classes `MatchResult` et DataFrames | Garder la représentation DataFrame, plus directe |
| La cible est un rang normalisé borné | Officiel + analyses 1/2 | CONFIRMED | Validation loader + postprocessing | Conserver bornes et métriques de régression |
| La cible suit une grille définie par `maxRank` | Analyses 1/2 | CONFIRMED | `snap_rank_percentage` et `snap_to_valid_grid` | Une seule fonction dans `evaluation.py` |
| `rankPts=-1` signifie absent | Officiel + analyses 1/2 | CONFIRMED | `flag_special_values` et autres variantes | Une seule fonction de nettoyage |
| `killPts=0` et `winPts=0` sont parfois absents | Officiel + analyses 1/2 | CONFIRMED | Nettoyage partiel dans EDA | Conversion conditionnelle en `NaN`, flags conservés |
| `date` prouve une séparation temporelle réelle | Analyse 1 | CONTRADICTED | `TimeSeriesSplit(date)` encore comparé | Retirer le split temporel principal ; documenter le pseudo-temps |
| Des lignes d'un même match traversent les pseudo-mois | Analyse 2 | CONFIRMED | Détecté dans EDA, pas centralisé | Exclure `date` des features et du split principal |
| Le split doit garder chaque `gameId` ensemble | Analyses 1/2 | CONFIRMED | `GroupKFold(gameId)` | Conserver comme unique stratégie principale |
| `teamId` est local à une partie | Officiel + analyse 2 | CONFIRMED | Quelques groupements simples par `teamId` subsistent | Toute clé équipe est `(gameId, teamId)` |
| Les rosters observés sont trop incomplets pour des agrégats équipe/lobby | Analyse 2 | CONFIRMED | Pas d'agrégats dans le contrat courant | Interdiction explicite dans `features.py` et docs |
| `maxRank != numTeams` est une anomalie | Analyse 1 | CONTRADICTED | Aucun correctif destructif | Conserver les deux valeurs sans correction |
| `highestKill` est la plus longue distance d'élimination mais peut tromper | Officiel + analyse 2 | CONFIRMED | Candidate historique | Exclure du contrat compact final |
| La mobilité est le signal comportemental dominant | Analyses 1/2 | CONFIRMED | `walkDist`, `walkDistPerMinute` | Garder `walkDist` et renommer le rythme par minute de match |
| Loot/progression et combat ajoutent du signal | Analyses 1/2 | CONFIRMED | Plusieurs composites et brutes | Garder quelques brutes fortes et deux composites lisibles |
| `killRank` est très prédictif mais dépend du scénario | Analyses 1/2 | CONFIRMED | Contrats avec/sans, mais bundle avec | Publier systématiquement les résultats avec et sans |
| La segmentation décrit des participations, pas des joueurs persistants | Analyses 1/2 | CONFIRMED | Deux K-means concurrents | Conserver la conclusion, retirer le clustering du pipeline final |
| Train et test ont des distributions proches | Analyses 1/2 | CONFIRMED | PSI, KS et Evidently | Conserver un contrôle léger, retirer les implémentations multiples |
| Dix candidates améliorent le modèle brut sûr | Analyse 2 | CONTRADICTED | Candidate engineering dispersé | Traiter les features comme compaction/interprétation, pas gain garanti |
| CatBoost est le meilleur modèle courant à 0,06037 | Artefact historique | NOT_REPRODUCIBLE | Bundle sur ancien hash/ancien contrat | Refaire le benchmark sur un contrat gelé |
| Calibration isotone apporte un gain utile | Artefact historique | PARTIALLY_CONFIRMED | Chaîne reliability dédiée | Gain trop faible ; ne pas déployer dans le livrable principal |
| La publication modèle/schéma est cohérente | Documentation historique | CONTRADICTED | Schéma et sélection divergents | Toute nouvelle publication doit être atomique et vérifiée |

## Conclusions consolidées

1. La statistique indépendante est le match, donc `GroupKFold(gameId)` est
   obligatoire.
2. `date` est un ordre d'export/anonymisation, pas une horloge fiable de match.
3. Le dataset ne permet ni agrégats complets d'équipe/lobby, ni rétention, ni
   causalité.
4. Le scénario principal est post-match comportemental ; `killRank` forme une
   variante séparée à forte dépendance mécanique.
5. Les scores de ranking externes sont nettoyés pour l'analyse mais exclus du
   contrat principal.
6. Le benchmark historique doit être reproduit avant toute nouvelle
   soumission.

## Continuité finale vérifiée

| Règle consolidée | Implémentation de référence | Validation |
|---|---|---|
| Sentinelles de ranking | `cleaning.clean_ranking_sentinels` | `tests/test_cleaning.py` |
| Schéma, IDs, dates et intégrité raw | `data.load_train_test` | `tests/test_data.py` + hashes SHA-256 |
| Familles et KPI par mode | `data.game_mode_summary` | notebook exécuté + `tests/test_data.py` |
| Features et divisions sûres | `features.build_model_features` | `tests/test_features.py` |
| Groupement match sans fuite | `validation.make_group_folds` | `tests/test_validation.py` + audit notebook |
| Métriques et grille de rang | `evaluation.regression_metrics`, `snap_to_rank_grid` | `tests/test_modeling.py` + ablation OOF |
| Benchmark unique | `modeling.compare_models` | notebook et script CLI exécutés |
| Modèle, schéma et hashes alignés | `modeling.save_model_bundle` | test round-trip + manifeste final |
