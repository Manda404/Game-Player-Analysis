# Analyse des KPI

Le tableau complet est
[`kpi_evaluation.csv`](../../artifacts/metrics/kpi_evaluation.csv).

| KPI | Règle | Décision |
|---|---|---|
| distance totale | marche + véhicule + nage | analytique, redondant avec marche |
| mobilité/seconde | distance totale / `gameTime` | analytique, durée ≠ survie |
| dégâts/kill | dégâts / kills, zéro si aucun kill | candidat, testé par ablation |
| headshot ratio | headshots / kills, absent si aucun kill | analytique, défini sur 43,03 % |
| activité combat | kills + assists + knocks | retenu après ablation |
| activité ressources | weapons + upgrades + heals | retenu après ablation |
| support | assists + revives | analytique, très clairsemé |

Les KPI sont descriptifs ou prédictifs ; aucun n'est interprété causalement.
L'ajout progressif de combat et ressources améliore la MAE d'environ 0,0029 à
chaque étape.
