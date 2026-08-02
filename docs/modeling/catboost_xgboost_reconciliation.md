# Réconciliation CatBoost contre XGBoost

## Pourquoi le gagnant semblait avoir changé

Le changement ne venait pas d'une supériorité robuste de XGBoost. Le tableau
pré-audit comparait des configurations déjà personnalisées. Sur les mêmes
40 128 lignes, 16 features et cinq folds groupés, leur rejeu donne :

| Configuration pré-audit | MAE | Écart-type | MAE train |
|---|---:|---:|---:|
| XGBoost personnalisé | 0,061999 | 0,001053 | 0,051368 |
| CatBoost personnalisé | 0,062053 | 0,001187 | 0,058519 |

L'avantage apparent de XGBoost n'est que de **0,000054 MAE**, négligeable face
à la variabilité des folds. Par rapport à leurs défauts, la personnalisation
améliorait XGBoost d'environ 0,00145 mais dégradait CatBoost d'environ 0,00060.
Elle suffisait donc à inverser artificiellement le classement initial.

## Ancienne version, refactor pré-audit et audit corrigé

| Élément | Archive pré-refactor | Refactor pré-audit | Audit corrigé | Impact |
|---|---|---|---|---|
| Split | 5 folds groupés | 5 folds groupés sur tout le train | holdout gelé puis 5 folds groupés sur développement | séparation sélection/final renforcée |
| Features | ancien contrat de 16 variables | contrat actuel de 16 variables | même contrat actuel pour tous | l'ancien score n'est pas directement comparable |
| Catégories | matrice numérique | matrice numérique | matrice numérique commune | aucun avantage natif CatBoost |
| CatBoost | défauts puis tuning léger | 800 itérations, profondeur 6, lr 0,05, L2 5 | défauts | supprime le pré-réglage |
| XGBoost | défauts | 500 arbres, profondeur 6, lr 0,05, subsampling 0,85 | défauts | supprime le pré-réglage |
| Métrique | MAE | MAE | MAE principale, diagnostics secondaires | cohérent |
| Gagnant | CatBoost, 0,060370 | XGBoost, 0,06165 annoncé | CatBoost, 0,061448 | CatBoost restauré par l'équité, pas par préférence |

L'ancienne conclusion CatBoost était directionnellement cohérente avec une
comparaison initiale par défaut, mais son score utilisait un autre contrat de
features et ne peut pas être repris comme preuve du pipeline actuel. La
conclusion refactorée en faveur de XGBoost n'était pas robuste. Le nouveau
résultat l'est au sein des cinq folds courants : CatBoost gagne les cinq.

Les artefacts de preuve sont
[`pre_audit_configuration_comparison.csv`](../../artifacts/metrics/pre_audit_configuration_comparison.csv)
et
[`model_fold_uncertainty.csv`](../../artifacts/metrics/model_fold_uncertainty.csv).
