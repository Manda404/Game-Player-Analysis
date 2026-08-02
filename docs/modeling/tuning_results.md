# Tuning

Six configurations XGBoost sont tirées d'un espace borné couvrant nombre
d'arbres, learning rate, profondeur, poids minimal d'enfant, sous-échantillonnage
et régularisations L1/L2. Seed et folds sont figés.

La meilleure configuration testée obtient 0,06177 contre 0,06165 pour la
configuration de référence : gain -0,00012. Le seuil de promotion fixé vaut
+0,0001 de MAE. La configuration tunée est donc rejetée.

Preuves :
[`tuning_trials.csv`](../../artifacts/metrics/tuning_trials.csv),
[`tuning_decision.json`](../../artifacts/metadata/tuning_decision.json) et
[`10_tuning_results.png`](../../artifacts/figures/10_tuning_results.png).
