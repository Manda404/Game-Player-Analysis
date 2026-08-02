# Qualité des données

Le tableau reproductible est
[`artifacts/metrics/data_quality.csv`](../../artifacts/metrics/data_quality.csv)
et la figure associée
[`02_data_quality.png`](../../artifacts/figures/02_data_quality.png).

- Train 50 000 × 30, test 5 000 × 29.
- Zéro cellule manquante brute et zéro doublon exact.
- 93 `gameId` train et 10 test ne respectent pas les 14 caractères hexadécimaux.
- 19 191 `rankPts=-1` train ; les zéros conditionnels de `killPts`/`winPts`
  sont convertis en valeurs absentes avec drapeaux.
- 53 lignes train ont `kills>0` et `damages=0`; 102 ont une activité de combat
  avec distance totale nulle. Elles sont conservées et signalées.

La règle de validation groupe les mêmes identifiants malformés ensemble. Elle
est conservatrice : mieux vaut sur-grouper une collision que laisser une fuite
entre deux folds.
