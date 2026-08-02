# Comparaison initiale non optimisée

## Règle de décision annoncée

La MAE de validation est le critère principal. RMSE, R², dispersion entre
folds, écart train-validation et coût de calcul sont secondaires. Tous les
modèles reçoivent les mêmes lignes, features et cinq folds groupés. Aucun
hyperparamètre d'apprentissage n'est réglé avant cette comparaison.

## Résultats sur le développement

| Rang | Modèle | MAE moyenne ± écart-type | Min–max | RMSE | R² | MAE train | Écart |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | CatBoost | **0,061448 ± 0,001095** | 0,060766–0,063341 | 0,086460 | 0,920835 | 0,055269 | 0,006180 |
| 2 | XGBoost | 0,063448 ± 0,001073 | 0,062617–0,065308 | 0,089602 | 0,914989 | 0,050440 | 0,013009 |
| 3 | LightGBM | 0,063593 ± 0,000912 | 0,063063–0,065216 | 0,089087 | 0,915970 | 0,059488 | 0,004105 |
| 4 | Random Forest | 0,066677 ± 0,000771 | 0,066054–0,068012 | 0,094343 | 0,905764 | 0,024722 | 0,041955 |
| 5 | Ridge | 0,088310 ± 0,001395 | 0,086679–0,090082 | 0,122248 | 0,841775 | 0,088256 | 0,000054 |
| 6 | Dummy médiane | 0,267877 ± 0,000822 | 0,267188–0,269101 | 0,307808 | -0,002935 | 0,267862 | 0,000016 |
| 7 | Dummy moyenne | 0,268151 ± 0,000792 | 0,267432–0,269250 | 0,307374 | -0,000112 | 0,268143 | 0,000008 |

CatBoost réduit la MAE du dummy médian de 0,20643 et bat XGBoost de 0,0019999
en moyenne. Dans la comparaison appariée, XGBoost est moins bon dans les cinq
folds ; l'intervalle bootstrap descriptif à 95 % de `MAE_XGB - MAE_CAT` vaut
[0,001829 ; 0,002192]. CatBoost est donc le seul candidat sélectionné pour le
tuning. Random Forest présente le surajustement le plus fort.

Les temps dépendent de la machine et servent de diagnostic, pas de critère
absolu. Le tableau complet est publié dans
[`initial_model_comparison.csv`](../../artifacts/metrics/initial_model_comparison.csv).
