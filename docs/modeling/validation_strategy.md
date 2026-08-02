# Stratégie de validation

## Protocole principal

GroupKFold à 5 folds sur un groupe `gameId` conservateur. Un identifiant valide
est conservé ; une valeur malformée répétée reste groupée ; seules les vraies
valeurs absentes seraient isolées par ligne. L'audit obtient zéro groupe et zéro
ID brut partagés sur chaque fold.

## Diagnostics secondaires

| Split | Lignes train/validation | Jeux déjà vus | MAE |
|---|---|---:|---:|
| aléatoire ligne | 40 000 / 10 000 | 56,64 % | 0,06169 |
| groupé | 39 901 / 10 099 | 0 % | 0,06168 |
| pseudo-temporel naïf | 37 815 / 12 185 | 54,97 % | 0,06169 |
| pseudo-temporel purgé | 29 279 / 12 185 | 0 % | 0,06222 |

Le pseudo-temporel utilise janvier–mars contre avril, puis retire tout groupe
vu en avril du train. Il mesure une sensibilité avec moins de données, pas une
chronologie fiable. Le test officiel de mai n'entre jamais dans la sélection.
