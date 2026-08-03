# Audit de la documentation actuelle et historique

> Inventaire réalisé avant consolidation. Les fichiers marqués MERGE/ARCHIVE
> ont ensuite été remplacés par la structure active décrite dans
> `docs/archive/README.md`; l'archive ZIP reste la source historique complète.

## Périmètre et méthode

L'audit couvre :

- le sujet officiel description_Gameloft_DS_technical_test.pdf ;
- README.md, le notebook final et tous les fichiers sous docs/ ;
- les documents, notebooks et 97 sorties sous reports/ conservés dans
  dist/pre_refactor_2026-08-02.zip ;
- les liens, chiffres, contrats de features et résultats de modèles cités.

Les décisions utilisent les statuts KEEP, IMPROVE, MERGE, MOVE, REWRITE,
REMOVE et VERIFY. Aucun résultat historique n'est repris comme résultat final
sans correspondance avec les hashes bruts et le contrat courant.

## Verdict

Le noyau utile est le README, le notebook exécuté, les synthèses analytiques,
le journal de refactoring et le rapport final. La principale dette est
documentaire :

- docs/report/ contient dix copies sémantiquement identiques de docs.old/ dans
  l'archive ;
- ces copies ont 18 liens cassés et réintroduisent une validation temporelle
  non défendable ;
- docs/report/, docs/analysis/, docs/refactoring/ et le rapport final répètent
  les mêmes conclusions ;
- plusieurs textes présentent une hypothèse d'ordre d'export comme un fait ;
- docs.old/ est cité comme dossier courant alors qu'il n'existe que dans le
  ZIP ;
- le notebook conserve des sorties avec l'ancien chemin absolu.

## Sujet officiel

Le PDF officiel confirme :

- une ligne = statistiques post-partie d'un joueur ;
- la cible est le classement final de son équipe, porté par la ligne joueur ;
- train : 1er janvier au 30 avril 2024 ; test : mai 2024 ;
- date est définie comme date de partie ;
- aucun gameId commun entre train et test ;
- les étapes Data Cleaning, Data Analysis & Visualization, Feature Engineering
  et Modeling doivent être visibles dans un notebook commenté ;
- la méthode et la lisibilité comptent plus que le seul score.

Le fichier reports/game_player_analysis_report_fr.pdf du
ZIP est un ancien rapport du projet, pas le sujet officiel.

## Audit des fichiers actifs

| Document | Objectif | Information unique | Redondance / défaut | Qualité | Décision |
|---|---|---|---|---|---|
| README.md | Point d'entrée | architecture, commandes, résultat | docs.old absent ; date trop affirmative | Bonne | IMPROVE |
| notebooks/game_player_analysis.ipynb | Livrable principal | workflow exécutable | 4 figures seulement ; ancien chemin ; diagnostics absents | Bonne base | REWRITE |
| docs/final_report.md | Synthèse finale | benchmark courant | répète README et insights ; analyse d'erreur absente | Bonne base | REWRITE |
| docs/analysis/consolidated_insights.md | Faits → décisions | limites et KPI compacts | preuve visuelle insuffisante | Bonne | MERGE |
| docs/analysis/existing_analysis_summary.md | Histoire des analyses | ancien bundle incohérent | docs.old présenté comme actif | Bonne | MERGE |
| docs/analysis/feature_engineering_rationale.md | Contrat de features | formules et exclusions | ablation limitée ; sans killRank n'est pas early-game | Bonne base | REWRITE |
| docs/refactoring/00_current_project_inventory.md | État pré-refactoring | inventaire daté | titre et présent trompeurs aujourd'hui | Historique | MOVE |
| docs/refactoring/01_codebase_audit.md | Audit ancien code | motifs de simplification | décrit parfois l'ancien état comme actuel | Bonne | MERGE |
| docs/refactoring/02_analysis_consolidation.md | Arbitrages v1/v2 | provenance des décisions | date trop catégorique | Bonne | MERGE |
| docs/refactoring/03_duplication_report.md | Duplication | chiffres avant refactor | rendu faux par docs/report réintroduit | Moyenne | REWRITE |
| docs/refactoring/04_target_architecture.md | Architecture cible | limites des couches | arborescence périmée | Moyenne | MERGE |
| docs/refactoring/05_refactoring_decisions.md | ADR D01–D14 | décisions et régression assumée | états non datés | Bonne | KEEP / MOVE |
| docs/refactoring/06_deleted_and_merged_files.md | Manifeste de 238 chemins | traçabilité exhaustive | trop long pour parcours actif ; incohérence docs/report | Bonne archive | MOVE |

## Audit de docs/report/

Ces dix fichiers sont des copies de l'analyse v1 présente dans l'archive. Les
faits utiles sont transférés dans les documents consolidés, puis les copies
doivent quitter le parcours actif.

| Document | Apport à conserver | Problème | Décision |
|---|---|---|---|
| 01_data_inventory.md | dictionnaire et dimensions | split temporel présenté comme vrai | MERGE puis REMOVE |
| 02_data_quality_analysis.md | IDs corrompus, sentinelles, anomalies | temporalité contradictoire, liens cassés | MERGE puis REMOVE |
| 03_univariate_analysis.md | quantiles, zéros, extrêmes | redondant, lien cassé | MERGE puis REMOVE |
| 04_feature_relationships.md | Spearman, VIF, interactions | temporalité invalide, six liens cassés | MERGE puis REMOVE |
| 05_target_analysis.md | mécanique et distribution cible | mai présenté comme futur fiable | MERGE puis REMOVE |
| 06_kpi_analysis.md | KPI mobilité/combat/soutien | pseudo-mois interprétés comme saisonnalité | MERGE puis REMOVE |
| 07_player_segmentation.md | segmentation k=3 d'intensité | résultat historique non indispensable | MOVE archive |
| 08_key_findings.md | synthèse métier | forte répétition | MERGE puis REMOVE |
| 09_recommendations.md | roadmap données/ML | recommande le split temporel invalidé | REWRITE puis REMOVE |
| 10_reproducibility.md | ancienne procédure | script et rapports absents | REMOVE |

## Conclusions communes fiables

- 50 000 lignes train et 5 000 test, sans cellule NaN ni doublon exact ;
- une ligne représente un joueur après la partie, la cible étant un résultat
  d'équipe ;
- les sentinelles de classement sont fréquentes et doivent être traitées ;
- walkDist est le signal comportemental brut dominant ;
- collecte, soins et combat ajoutent une information complémentaire ;
- killRank est très prédictif et doit être isolé dans une ablation ;
- les identifiants et la cible sont exclus des features ;
- gameId doit structurer la validation ;
- les rosters sont trop incomplets pour des agrégats équipe/lobby fiables ;
- la grille de maxRank justifie d'évaluer le snapping séparément ;
- les données ne permettent ni causalité, ni rétention, ni churn.

## Contradictions et éléments à vérifier

| Sujet | Sources en conflit | Arbitrage final |
|---|---|---|
| Sémantique de date | officiel = date de partie ; données = plusieurs dates par gameId | champ incohérent ; GroupKFold principal, stress test pseudo-temporel seulement |
| Cible joueur/équipe | plusieurs docs parlent de rang joueur ; officiel parle du score de l'équipe | écrire « score d'équipe porté par une ligne joueur » |
| Contrat sans killRank | parfois appelé early-game | REWRITE : post-match sans rang de kills |
| CatBoost historique 0,06037 | ancien contrat/bundle incompatibles | archive uniquement |
| XGBoost courant 0,06156 | artefact reproductible | référence finale avant réexécution consolidée |
| RMSE 0,08666 / 0,08667 | moyenne des folds / métrique OOF globale | nommer l'agrégation |
| docs.old/ | cité comme actif | absent ; contenu seulement dans le ZIP |
| Segmentation k=3 / k=4 | intensité / styles, protocoles différents | analyses historiques, pas vérité unique |

## Audit des 97 sorties historiques de reports/

### consolidation_analysis — 11 fichiers

| Fichier | Information unique | Décision |
|---|---|---|
| artifact_consistency_checks.csv | incompatibilités ancien bundle/schema | MERGE, KEEP archive |
| categorical_candidate_target_profile.csv | cible par mode/régime | KEEP archive |
| consolidation_manifest.json | hashes et périmètre historique | KEEP archive |
| feature_candidate_incremental_value.csv | gain individuel de dix candidates | MERGE, VERIFY |
| feature_candidate_model_diagnostic.csv | cinq contrats historiques | VERIFY, KEEP archive |
| feature_candidate_statistics.csv | profil des candidates | KEEP archive |
| feature_candidate_train_test_drift.csv | drift des candidates | KEEP archive |
| hierarchy_coverage.csv | couverture équipe/match | MERGE |
| hierarchy_coverage_by_mode.csv | couverture par mode | KEEP archive |
| hierarchy_feasibility.json | 1 164 lignes éligibles aux agrégats | MERGE |
| mode_family_target_profile.csv | cible par famille de mode | MERGE |

### independent_raw_analysis — 38 fichiers

| Fichier | Information | Décision |
|---|---|---|
| figures/segment_profile_heatmap.png | profils k=3 | KEEP archive |
| figures/segment_target_boxplot.png | cible k=3 | KEEP archive |
| figures/target_distribution.png | cible | REGENERATE |
| figures/target_spearman_heatmap.png | corrélations | REGENERATE |
| binned_effect_sizes.csv | lift par quantiles | MERGE |
| binned_feature_target_profiles.csv | profils non linéaires | MERGE sélectif |
| categorical_and_temporal_tests.json | effets catégories/pseudo-temps | VERIFY |
| categorical_associations.csv | Cramér V | KEEP archive |
| categorical_frequencies.csv | fréquences, majoritairement IDs | KEEP archive |
| data_quality_checks.csv | 82 contrôles | MERGE |
| date_quality.json | cadence quasi régulière | MERGE |
| game_type_numeric_associations.csv | effets du mode | KEEP archive |
| game_type_target_summary.csv | cible/KPI par mode | MERGE |
| identifier_group_target_consistency.csv | cohérence cible | KEEP archive |
| identifier_train_test_overlap.json | zéro overlap | MERGE |
| kpi_by_gameType.csv | KPI par mode | MERGE sélectif |
| kpi_by_month.csv | KPI par pseudo-mois | VERIFY, KEEP archive |
| kpi_summary.csv | 23 KPI | MERGE |
| monthly_target_summary.csv | cible par pseudo-mois | VERIFY |
| numeric_pair_correlations.csv | 552 corrélations | REMOVE doublon si extrait |
| numeric_univariate_summary.csv | profils numériques | MERGE sélectif |
| ranking_system_target_summary.csv | régimes externes | MERGE |
| raw_file_fingerprints.json | hashes sources | MERGE |
| segment_behaviour_profile.csv | profils k=3 | KEEP archive |
| segment_cluster_selection.csv | silhouettes k=3 à 6 | KEEP archive |
| segment_game_type_composition.csv | segments × modes | KEEP archive |
| segment_methodology.json | protocole k=3 | KEEP archive |
| segment_profile_index_base100.csv | indices k=3 | KEEP archive |
| segment_target_summary.csv | cible k=3 | KEEP archive |
| segment_target_within_major_modes.csv | stabilité segment/mode | KEEP archive |
| target_distribution_bins.csv | cible en bins | REGENERATE |
| target_feature_relationships.csv | Spearman/Pearson/MI | MERGE |
| target_quantisation_check.json | compatibilité grille | MERGE, VERIFY tolérance |
| target_summary.json | résumé cible | MERGE |
| train_test_game_type_drift.csv | drift mode | MERGE |
| train_test_numeric_drift.csv | drift numérique | REMOVE doublon si extrait |
| variance_inflation_factors.csv | VIF | MERGE sélectif |
| walk_kills_target_interaction.csv | mobilité × kills | MERGE |

### lead_scientific_analysis — 48 fichiers

| Fichier | Information | Décision |
|---|---|---|
| figures/mobility_combat_heatmap.png | interaction | REGENERATE |
| figures/model_feature_sets_and_calibration.png | ancien modèle | VERIFY, KEEP archive |
| figures/segment_pca.png | k=4 | KEEP archive |
| figures/segment_profiles.png | k=4 | KEEP archive |
| figures/target_and_relationships.png | cible/relations | REGENERATE |
| figures/zero_concentration.png | zéros | REGENERATE |
| actionable_style_quadrants.csv | quadrants activité | MERGE |
| categorical_drift.csv | drift catégories | KEEP archive |
| categorical_profile.csv | profil catégories | KEEP archive |
| categorical_target_tests.csv | tests de cible | KEEP archive |
| cluster_selection.csv | k=4, silhouette | KEEP archive |
| date_structure.csv | structure de date | MERGE |
| feature_pair_correlations.csv | corrélations | KEEP archive |
| feature_target_relationships.csv | relations partielles/FDR | MERGE sélectif |
| game_time_conditional_checks.csv | durée/cible | MERGE |
| group_integrity.csv | incohérence date/gameId | MERGE |
| group_validation_permutation_importance.csv | importance ancien HGB | VERIFY, REGENERATE |
| group_validation_residuals_by_mode.csv | erreurs ancien HGB | VERIFY, REGENERATE |
| group_validation_residuals_by_quality_flag.csv | erreurs/anomalies | VERIFY, REGENERATE |
| group_validation_residuals_by_target_decile.csv | biais aux extrêmes | VERIFY, REGENERATE |
| identifier_quality.csv | IDs mal formés | MERGE |
| inventory.csv | dimensions/hashes | MERGE |
| kpis.csv | 22 KPI | MERGE |
| match_sampling_coverage.json | couverture sparse | MERGE |
| mobility_combat_interaction.csv | interaction 4×4 | MERGE |
| mode_business_profile.csv | KPI par mode | MERGE sélectif |
| model_ablation_diagnostics.csv | gains historiques | VERIFY |
| model_feature_set_comparison.csv | contrats HGB historiques | VERIFY |
| nonlinear_target_profiles.csv | profils en bins | MERGE sélectif |
| numeric_univariate_profile.csv | profil numérique | MERGE sélectif |
| numeric_vif.csv | VIF | MERGE sélectif |
| prediction_postprocessing_comparison.csv | ancien snapping | VERIFY |
| quality_issues.csv | 27 contrôles | MERGE |
| ranking_system_regimes.csv | régimes | MERGE |
| segment_index_profiles.csv | indices k=4 | KEEP archive |
| segment_mode_composition.csv | k=4 × modes | KEEP archive |
| segment_monthly_stability.csv | pseudo-mois | VERIFY |
| segment_pca_loadings.csv | PCA | KEEP archive |
| segment_pca_variance.csv | PCA | KEEP archive |
| segment_raw_profiles.csv | profils k=4 | KEEP archive |
| segment_target_profiles.csv | cible k=4 | KEEP archive |
| segmentation_methodology.json | méthode k=4 | KEEP archive |
| target_bins.csv | distribution | REGENERATE |
| target_mechanics.json | formule de grille | MERGE |
| target_summary.json | résumé cible | REMOVE doublon si extrait |
| train_test_drift.csv | drift | MERGE |
| train_test_identifier_overlap.csv | overlap | MERGE |
| validation_split_audit.csv | trois anciens splits | VERIFY, REGENERATE |

## Analyses utiles absentes du notebook courant

- cadence de date et spans intra-game ;
- zéros, sentinelles, longues queues et anomalies ;
- joueurs observés par partie et par équipe ;
- relations par quantiles avec intervalles d'incertitude ;
- redondances entre KPI et variables sources ;
- ablation par familles ;
- comparaison prédictive de quatre stratégies de split ;
- tuning courant ;
- train-validation gap et variance des folds ;
- résidus par cible, mode, grille de maxRank et activité ;
- importance par permutation ;
- inférence autonome.

## Structure documentaire finale

~~~text
docs/
├── analysis/
│   ├── data_quality.md
│   ├── exploratory_analysis.md
│   └── kpi_analysis.md
├── feature_engineering/
│   ├── feature_rationale.md
│   └── ablation_study.md
├── modeling/
│   ├── validation_strategy.md
│   ├── model_comparison.md
│   ├── tuning_results.md
│   ├── interpretability.md
│   └── error_analysis.md
├── review/
│   ├── 00_self_critique.md
│   ├── 01_docs_reports_audit.md
│   ├── 02_data_science_coverage.md
│   └── 03_final_changes.md
├── archive/
│   └── README.md
└── final_report.md
~~~

Le notebook reste la narration principale. Les documents approfondissent une
décision sans recopier le notebook. Le ZIP reste l'unique archive exhaustive.
