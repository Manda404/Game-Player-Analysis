# Résultats du tuning

CatBoost est sélectionné avant optimisation. Huit configurations sont tirées
par `RandomizedSearchCV` et évaluées sur les mêmes cinq folds groupés.

La configuration par défaut obtient 0,061448 ± 0,001095 de MAE. Le meilleur
essai aléatoire obtient 0,061671 ± 0,001222 : le « gain » vaut donc -0,000223.
Le seuil de promotion de +0,0001 n'est pas atteint et le tuning est rejeté.

Le protocole complet figure dans
[`hyperparameter_tuning.md`](hyperparameter_tuning.md). Preuves :
[`tuning_trials.csv`](../../artifacts/metrics/tuning_trials.csv),
[`tuning_decision.json`](../../artifacts/metadata/tuning_decision.json) et
[`10_tuning_results.png`](../../artifacts/figures/10_tuning_results.png).
