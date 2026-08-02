# Comparaison des modèles

La comparaison corrigée utilise les hyperparamètres d'apprentissage par défaut,
les mêmes 16 features, les mêmes 40 128 lignes de développement et les mêmes
cinq folds groupés par `gameId`.

| Modèle | MAE | Écart-type | Train MAE | Lecture |
|---|---:|---:|---:|---|
| CatBoost | **0,061448** | 0,001095 | 0,055269 | retenu |
| XGBoost | 0,063448 | 0,001073 | 0,050440 | moins bon sur les 5 folds |
| LightGBM | 0,063593 | 0,000912 | 0,059488 | moins bon sur les 5 folds |
| Random Forest | 0,066677 | 0,000771 | 0,024722 | surajustement fort |
| Ridge | 0,088310 | 0,001395 | 0,088256 | baseline linéaire |
| Dummy médiane | 0,267877 | 0,000822 | 0,267862 | baseline naïve |
| Dummy moyenne | 0,268151 | 0,000792 | 0,268143 | baseline naïve |

CatBoost bat XGBoost de 0,0019999 MAE en moyenne et dans chacun des cinq folds.
Le rejeu des anciennes configurations personnalisées explique l'ancien gagnant
XGBoost : leur écart n'était alors que 0,000054 MAE.

Voir la
[comparaison détaillée](initial_model_comparison.md), la
[réconciliation CatBoost/XGBoost](catboost_xgboost_reconciliation.md) et
[`initial_model_comparison.csv`](../../artifacts/metrics/initial_model_comparison.csv).
