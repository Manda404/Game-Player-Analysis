# 06 — Fichiers supprimés, fusionnés ou déplacés

Tous les éléments ci-dessous restent récupérables dans
`dist/pre_refactor_2026-08-02.zip`. `data/raw/` et `docs.old/` n'ont pas été
modifiés.

| Éléments retirés | Fonction précédente | Destination de la logique utile | Vérification |
|---|---|---|---|
| `src/.../application/` | Ports et cas d'usage | Appels directs aux modules métier | Tests et notebook |
| `src/.../domain/` | Entités, enums, contrats | `features.py`, `evaluation.py`, `config.py` | Tests formules/contrats |
| `src/.../infrastructure/` | Pandas, ML, artefacts, graphiques | Sept modules spécialisés à la racine du package | Benchmark et tests |
| `src/.../presentation/` | Façades notebooks/scripts | Notebook + `scripts/run_analysis.py` | Imports/py_compile |
| `logging_config.py` | Logging fichier | Sorties notebook/CLI | Aucun consommateur |
| sept anciens scripts | Analyses, tuning, entraînement, inférence | Un script complet et trois docs analytiques | Exécution notebook |
| 17 anciens notebooks | EDA et pipeline par étapes | `notebooks/game_player_analysis.ipynb` | 10 cellules code exécutées |
| 31 anciens fichiers de tests | Architecture et services supprimés | Six fichiers de tests ciblés | Suite finale verte |
| `config/settings.dev.yaml` | Configuration multi-couche | `config.py` | Imports/tests |
| `data/processed/*` | Contrats préparés divergents | Transformation depuis raw | Colonnes train/test identiques |
| ancien `models/*` | Bundle CatBoost incohérent | `artifacts/model.*` et manifeste | Feature count/hash alignés |
| anciens `reports/*` | 97 tables/figures | `docs/analysis/*` | Valeurs clés conservées |
| `Gameloft-Data-Science-Overview/*` | Rapport historique obsolète | Notebook et `docs/final_report.md` | Résultats finaux cités |
| docs actives numérotées 00–13 | Deuxième analyse éclatée | Trois synthèses + rapport final | Liens et chiffres vérifiés |
| caches, logs et traces CatBoost | Fichiers générés | Aucun | Recréables, sans valeur livrable |

## Logique explicitement abandonnée

- architecture Clean Architecture locale ;
- sélection de features à partir d'un pool brut incomplet ;
- `killRankPercentile` dupliquant `killRank` ;
- calibration isotone/linéaire dans le chemin principal ;
- Random Search et Optuna ;
- dérive via trois bibliothèques ;
- clustering comme entrée modèle ;
- split temporel sur la pseudo-date ;
- agrégats équipe/match sur rosters incomplets.

## Contrôles de non-régression fonctionnelle

- mêmes hashes bruts ;
- mêmes 50 000 et 5 000 lignes chargées ;
- sentinelles comptées à l'identique ;
- zéro match partagé entre folds ;
- quatre familles demandées comparées ;
- soumission de 5 000 lignes bornées et sans null ;
- modèle, features, benchmark et hashes dans le même bundle.

## Manifeste exhaustif des chemins retirés

Les 238 chemins ci-dessous ont été comparés à l'état courant et restent
récupérables, octet pour octet, dans l'archive pré-refactoring. Les fichiers
conservant le même chemin mais entièrement réécrits ne figurent pas dans cette
liste : ils relèvent de `REFACTOR`, pas de `DELETE`.

<details>
<summary>Afficher le manifeste complet</summary>

```text
Gameloft-Data-Science-Overview/GamePlayerAnalysis-Overview.pdf
Gameloft-Data-Science-Overview/GamePlayerAnalysis-Overview.tex
Gameloft-Data-Science-Overview/IEEEtran.cls
Gameloft-Data-Science-Overview/figures/cible_et_correlations.png
Gameloft-Data-Science-Overview/figures/comparaison_decoupages.png
Gameloft-Data-Science-Overview/figures/erreur_par_taille.png
Gameloft-Data-Science-Overview/figures/matrice_correlations.png
Gameloft-Data-Science-Overview/figures/stabilite_train_test.png
Gameloft-Data-Science-Overview/generer_figures.py
config/settings.dev.yaml
data/processed/test_processed.csv
data/processed/train_processed.csv
docs/00_existing_analysis_audit.md
docs/01_data_inventory.md
docs/02_data_quality_analysis.md
docs/03_univariate_analysis.md
docs/04_feature_relationships.md
docs/05_target_analysis.md
docs/06_kpi_analysis.md
docs/07_player_segmentation.md
docs/08_key_findings.md
docs/09_recommendations.md
docs/10_analysis_comparison.md
docs/10_feature_engineering_candidates.md
docs/11_feature_engineering_and_modeling.md
docs/11_validation_strategy.md
docs/12_leakage_assessment.md
docs/12_reproducibility.md
docs/13_consolidated_analysis_report.md
models/catboost_winrank.cbm
models/feature_schema.json
models/model_benchmark.json
models/model_manifest.json
models/oof_diagnostics.json
models/preprocessing.json
models/selected_feature_evaluation.json
models/selected_features.json
models/training_evaluation.json
models/tuning.json
notebooks/01_load_data.ipynb
notebooks/02_eda_guide.ipynb
notebooks/03_feature_engineering.ipynb
notebooks/04_feature_selection.ipynb
notebooks/05_split_strategy.ipynb
notebooks/06_modeling.ipynb
notebooks/07_hyperparameter_tuning.ipynb
notebooks/08_inference.ipynb
notebooks/eda/02_1_data_quality_target.ipynb
notebooks/eda/02_2_match_structure.ipynb
notebooks/eda/02_3_player_behavior.ipynb
notebooks/eda/02_4_feature_relationships.ipynb
notebooks/eda/02_5_drift_insights.ipynb
notebooks/eda/02_6_robustness_interactions.ipynb
notebooks/eda/02_7_export_order_stability.ipynb
notebooks/eda/02_8_sentinels_outliers.ipynb
notebooks/eda/02_9_semantic_integrity.ipynb
notebooks/eda/README.md
reports/consolidation_analysis/tables/artifact_consistency_checks.csv
reports/consolidation_analysis/tables/categorical_candidate_target_profile.csv
reports/consolidation_analysis/tables/consolidation_manifest.json
reports/consolidation_analysis/tables/feature_candidate_incremental_value.csv
reports/consolidation_analysis/tables/feature_candidate_model_diagnostic.csv
reports/consolidation_analysis/tables/feature_candidate_statistics.csv
reports/consolidation_analysis/tables/feature_candidate_train_test_drift.csv
reports/consolidation_analysis/tables/hierarchy_coverage.csv
reports/consolidation_analysis/tables/hierarchy_coverage_by_mode.csv
reports/consolidation_analysis/tables/hierarchy_feasibility.json
reports/consolidation_analysis/tables/mode_family_target_profile.csv
reports/independent_raw_analysis/figures/segment_profile_heatmap.png
reports/independent_raw_analysis/figures/segment_target_boxplot.png
reports/independent_raw_analysis/figures/target_distribution.png
reports/independent_raw_analysis/figures/target_spearman_heatmap.png
reports/independent_raw_analysis/tables/binned_effect_sizes.csv
reports/independent_raw_analysis/tables/binned_feature_target_profiles.csv
reports/independent_raw_analysis/tables/categorical_and_temporal_tests.json
reports/independent_raw_analysis/tables/categorical_associations.csv
reports/independent_raw_analysis/tables/categorical_frequencies.csv
reports/independent_raw_analysis/tables/data_quality_checks.csv
reports/independent_raw_analysis/tables/date_quality.json
reports/independent_raw_analysis/tables/game_type_numeric_associations.csv
reports/independent_raw_analysis/tables/game_type_target_summary.csv
reports/independent_raw_analysis/tables/identifier_group_target_consistency.csv
reports/independent_raw_analysis/tables/identifier_train_test_overlap.json
reports/independent_raw_analysis/tables/kpi_by_gameType.csv
reports/independent_raw_analysis/tables/kpi_by_month.csv
reports/independent_raw_analysis/tables/kpi_summary.csv
reports/independent_raw_analysis/tables/monthly_target_summary.csv
reports/independent_raw_analysis/tables/numeric_pair_correlations.csv
reports/independent_raw_analysis/tables/numeric_univariate_summary.csv
reports/independent_raw_analysis/tables/ranking_system_target_summary.csv
reports/independent_raw_analysis/tables/raw_file_fingerprints.json
reports/independent_raw_analysis/tables/segment_behaviour_profile.csv
reports/independent_raw_analysis/tables/segment_cluster_selection.csv
reports/independent_raw_analysis/tables/segment_game_type_composition.csv
reports/independent_raw_analysis/tables/segment_methodology.json
reports/independent_raw_analysis/tables/segment_profile_index_base100.csv
reports/independent_raw_analysis/tables/segment_target_summary.csv
reports/independent_raw_analysis/tables/segment_target_within_major_modes.csv
reports/independent_raw_analysis/tables/target_distribution_bins.csv
reports/independent_raw_analysis/tables/target_feature_relationships.csv
reports/independent_raw_analysis/tables/target_quantisation_check.json
reports/independent_raw_analysis/tables/target_summary.json
reports/independent_raw_analysis/tables/train_test_game_type_drift.csv
reports/independent_raw_analysis/tables/train_test_numeric_drift.csv
reports/independent_raw_analysis/tables/variance_inflation_factors.csv
reports/independent_raw_analysis/tables/walk_kills_target_interaction.csv
reports/lead_scientific_analysis/figures/mobility_combat_heatmap.png
reports/lead_scientific_analysis/figures/model_feature_sets_and_calibration.png
reports/lead_scientific_analysis/figures/segment_pca.png
reports/lead_scientific_analysis/figures/segment_profiles.png
reports/lead_scientific_analysis/figures/target_and_relationships.png
reports/lead_scientific_analysis/figures/zero_concentration.png
reports/lead_scientific_analysis/tables/actionable_style_quadrants.csv
reports/lead_scientific_analysis/tables/categorical_drift.csv
reports/lead_scientific_analysis/tables/categorical_profile.csv
reports/lead_scientific_analysis/tables/categorical_target_tests.csv
reports/lead_scientific_analysis/tables/cluster_selection.csv
reports/lead_scientific_analysis/tables/date_structure.csv
reports/lead_scientific_analysis/tables/feature_pair_correlations.csv
reports/lead_scientific_analysis/tables/feature_target_relationships.csv
reports/lead_scientific_analysis/tables/game_time_conditional_checks.csv
reports/lead_scientific_analysis/tables/group_integrity.csv
reports/lead_scientific_analysis/tables/group_validation_permutation_importance.csv
reports/lead_scientific_analysis/tables/group_validation_residuals_by_mode.csv
reports/lead_scientific_analysis/tables/group_validation_residuals_by_quality_flag.csv
reports/lead_scientific_analysis/tables/group_validation_residuals_by_target_decile.csv
reports/lead_scientific_analysis/tables/identifier_quality.csv
reports/lead_scientific_analysis/tables/inventory.csv
reports/lead_scientific_analysis/tables/kpis.csv
reports/lead_scientific_analysis/tables/match_sampling_coverage.json
reports/lead_scientific_analysis/tables/mobility_combat_interaction.csv
reports/lead_scientific_analysis/tables/mode_business_profile.csv
reports/lead_scientific_analysis/tables/model_ablation_diagnostics.csv
reports/lead_scientific_analysis/tables/model_feature_set_comparison.csv
reports/lead_scientific_analysis/tables/nonlinear_target_profiles.csv
reports/lead_scientific_analysis/tables/numeric_univariate_profile.csv
reports/lead_scientific_analysis/tables/numeric_vif.csv
reports/lead_scientific_analysis/tables/prediction_postprocessing_comparison.csv
reports/lead_scientific_analysis/tables/quality_issues.csv
reports/lead_scientific_analysis/tables/ranking_system_regimes.csv
reports/lead_scientific_analysis/tables/segment_index_profiles.csv
reports/lead_scientific_analysis/tables/segment_mode_composition.csv
reports/lead_scientific_analysis/tables/segment_monthly_stability.csv
reports/lead_scientific_analysis/tables/segment_pca_loadings.csv
reports/lead_scientific_analysis/tables/segment_pca_variance.csv
reports/lead_scientific_analysis/tables/segment_raw_profiles.csv
reports/lead_scientific_analysis/tables/segment_target_profiles.csv
reports/lead_scientific_analysis/tables/segmentation_methodology.json
reports/lead_scientific_analysis/tables/target_bins.csv
reports/lead_scientific_analysis/tables/target_mechanics.json
reports/lead_scientific_analysis/tables/target_summary.json
reports/lead_scientific_analysis/tables/train_test_drift.csv
reports/lead_scientific_analysis/tables/train_test_identifier_overlap.csv
reports/lead_scientific_analysis/tables/validation_split_audit.csv
scripts/analyze_raw_data_from_scratch.py
scripts/evaluate_model_reliability.py
scripts/evaluate_selected_features.py
scripts/predict_submission.py
scripts/run_consolidation_analysis.py
scripts/run_lead_scientific_analysis.py
scripts/train_final_model.py
src/game_player_analysis/README.md
src/game_player_analysis/application/__init__.py
src/game_player_analysis/application/interfaces.py
src/game_player_analysis/application/modeling_interfaces.py
src/game_player_analysis/application/modeling_use_cases.py
src/game_player_analysis/application/use_cases.py
src/game_player_analysis/config/__init__.py
src/game_player_analysis/domain/__init__.py
src/game_player_analysis/domain/entities.py
src/game_player_analysis/domain/enums.py
src/game_player_analysis/domain/modeling.py
src/game_player_analysis/infrastructure/__init__.py
src/game_player_analysis/infrastructure/data_loader.py
src/game_player_analysis/infrastructure/data_profile.py
src/game_player_analysis/infrastructure/drift.py
src/game_player_analysis/infrastructure/eda.py
src/game_player_analysis/infrastructure/explainability.py
src/game_player_analysis/infrastructure/feature_engineering.py
src/game_player_analysis/infrastructure/feature_selection.py
src/game_player_analysis/infrastructure/feature_sets.py
src/game_player_analysis/infrastructure/hyperparameter_tuning.py
src/game_player_analysis/infrastructure/inference.py
src/game_player_analysis/infrastructure/model_artifacts.py
src/game_player_analysis/infrastructure/model_evaluation.py
src/game_player_analysis/infrastructure/model_factory.py
src/game_player_analysis/infrastructure/modeling_services.py
src/game_player_analysis/infrastructure/postprocessing.py
src/game_player_analysis/infrastructure/production_evaluation.py
src/game_player_analysis/infrastructure/production_services.py
src/game_player_analysis/infrastructure/production_training.py
src/game_player_analysis/infrastructure/reliability.py
src/game_player_analysis/infrastructure/selection_artifacts.py
src/game_player_analysis/infrastructure/split_analysis.py
src/game_player_analysis/infrastructure/tabular_store.py
src/game_player_analysis/infrastructure/tuning_service.py
src/game_player_analysis/infrastructure/visualization.py
src/game_player_analysis/infrastructure/workflow_artifacts.py
src/game_player_analysis/logging_config.py
src/game_player_analysis/presentation/__init__.py
src/game_player_analysis/presentation/__main__.py
src/game_player_analysis/presentation/api.py
src/game_player_analysis/presentation/game_player_workflow.py
src/game_player_analysis/presentation/notebook_setup.py
src/game_player_analysis/presentation/production_model_workflow.py
src/game_player_analysis/presentation/workflow_context.py
tests/fixtures/README.md
tests/fixtures/mock_match_data.csv
tests/integration/test_load_and_analyze_pipeline.py
tests/unit/application/test_modeling_use_cases.py
tests/unit/application/test_use_cases.py
tests/unit/config/test_config.py
tests/unit/domain/test_match_result.py
tests/unit/domain/test_modeling_contracts.py
tests/unit/infrastructure/test_data_loader.py
tests/unit/infrastructure/test_data_profile.py
tests/unit/infrastructure/test_drift.py
tests/unit/infrastructure/test_eda.py
tests/unit/infrastructure/test_explainability.py
tests/unit/infrastructure/test_feature_engineering.py
tests/unit/infrastructure/test_feature_selection.py
tests/unit/infrastructure/test_feature_sets.py
tests/unit/infrastructure/test_hyperparameter_tuning.py
tests/unit/infrastructure/test_inference.py
tests/unit/infrastructure/test_model_evaluation.py
tests/unit/infrastructure/test_model_factory.py
tests/unit/infrastructure/test_postprocessing.py
tests/unit/infrastructure/test_production_services.py
tests/unit/infrastructure/test_reliability.py
tests/unit/infrastructure/test_split_analysis.py
tests/unit/infrastructure/test_tabular_store.py
tests/unit/infrastructure/test_tuning_service.py
tests/unit/infrastructure/test_visualization.py
tests/unit/infrastructure/test_workflow_artifacts.py
tests/unit/presentation/test_game_player_workflow.py
tests/unit/test_architecture.py
tests/unit/test_model_artifacts.py
tests/unit/test_selection_artifacts.py
```

</details>
