# 05 — Analyse de la target

## Définition probable

`winRankPercentage` est une cible continue bornée entre 0 et 1. L'hypothèse la mieux soutenue est une place finale normalisée : **1 = meilleur classement**, **0 = dernier classement**. La compatibilité à 100 % avec `1 + (1-target) × (maxRank-1)` arrondi à une place entière confirme que la cible dépend de la place et de la taille de la partie.

Cette définition reste une hypothèse jusqu'à validation par le propriétaire des données.

## Distribution

| Statistique | Valeur |
|---|---:|
| Observations | 50 000 |
| Valeurs distinctes | 1 445 |
| Moyenne | 0,472333 |
| Médiane | 0,457850 |
| Écart-type | 0,307449 |
| Q1 / Q3 | 0,2000 / 0,7407 |
| P5 / P95 | 0,0107 / 0,9630 |
| Asymétrie | 0,102230 |
| Valeur 0 | 2 450 (4,900 %) |
| Valeur 1 | 1 430 (2,860 %) |
| Cible ≥ 0,9 | 5 249 (10,498 %) |

La cible n'est pas déséquilibrée comme le serait une classification rare : chaque tranche de largeur 0,1 hors zéro contient entre 8,184 % et 11,208 % des observations. Les bornes sont toutefois surreprésentées par la discrétisation des rangs.

![Distribution et fonction de répartition de la cible](../reports/independent_raw_analysis/figures/target_distribution.png)

### Conséquences analytiques

- Il s'agit d'un problème de **régression bornée**, pas d'une classification gagnant/perdant.
- MAE et RMSE doivent être rapportées ; la RMSE pénalise davantage les grosses erreurs, la MAE est plus lisible en points de classement normalisé.
- Les prédictions doivent être contrôlées dans `[0,1]` et évaluées séparément aux bornes 0/1.
- La granularité varie avec `maxRank`; les résidus peuvent présenter des bandes même avec un bon modèle.

## Relations feature–target

| Feature | Pearson | Spearman | Information mutuelle | Lecture principale |
|---|---:|---:|---:|---|
| `walkDist` | 0,812 | **0,866** | 0,710 | Signal monotone le plus fort ; très probablement mêlé au temps de survie |
| `killRank` | -0,713 | **-0,715** | 0,913 | Très informatif, direction inversée ; risque de fuite post-partie |
| `upgrades` | 0,635 | 0,681 | 0,322 | Relation graduée et non linéaire ; proxy de progression/temps en jeu |
| `weapons` | 0,582 | 0,665 | 0,321 | Collecte associée au rang, avec rendement décroissant |
| `heals` | 0,427 | 0,564 | 0,201 | Spearman nettement supérieur à Pearson : relation monotone non linéaire |
| `highestKill` | 0,410 | 0,450 | 0,146 | Performance de combat ; zéro partagé avec l'absence de kills |
| `damages` | 0,442 | 0,449 | 0,149 | Signal de combat continu |
| `rideDist` | 0,341 | 0,433 | 0,120 | Relation non linéaire et zero-inflated |
| `kills` | 0,417 | 0,425 | 0,120 | Positif mais inférieur à la mobilité/collecte |
| `killStreaks` | 0,370 | 0,385 | 0,097 | Redondant avec `kills` |
| `assists` | 0,301 | 0,301 | 0,071 | Signal de coopération modéré, non applicable en solo |
| `headshots` | 0,281 | 0,285 | 0,052 | Signal faible à modéré et très sparse |
| `revives` | 0,249 | 0,259 | 0,050 | Coopération, dépend fortement du mode |
| `knocks` | 0,283 | 0,257 | 0,112 | Redondant avec combat ; relation non strictement monotone |
| `swimDist` | 0,153 | 0,238 | 0,039 | Faible, sparse, probablement contextuel |
| `vehicleDestr` | 0,071 | 0,072 | 0,001 | Effet très faible |
| `rankPts` brut | 0,007 | 0,062 | 0,041 | Pearson non significatif (`p=0,111`) ; sentinelles rendent la valeur brute invalide |
| `numTeams` | 0,037 | 0,048 | **1,004** | Peu d'effet sur la moyenne, mais dépendance structurelle via granularité de la cible |
| `maxRank` | 0,035 | 0,046 | **2,282** | Forte information sur les valeurs possibles de la cible, pas sur le rang moyen |
| `winPts` brut | 0,014 | 0,044 | 0,022 | Effet brut minuscule, régime de score confondu |
| `roadKills` | 0,031 | 0,034 | 0,001 | Extrêmement rare |
| `killPts` brut | 0,019 | 0,022 | 0,018 | Effet brut minuscule, régime de score confondu |
| `teamKills` | 0,014 | 0,021 | 0,005 | Effet quasi nul |
| `gameTime` | -0,003 | 0,004 | 0,012 | Aucune relation globale (`p` Pearson 0,510 ; Spearman 0,357) |

L'information mutuelle n'est pas normalisée. Les valeurs élevées de `maxRank` et `numTeams` proviennent principalement du pas de discrétisation autorisé par chaque taille de partie. Elles ne signifient pas que ces variables prédisent à elles seules qui gagne.

L'absence de relation de `gameTime` avec la cible, combinée à sa dépendance au mode, indique que la colonne représente plus probablement la **durée totale de la partie** que le temps de survie individuel.

## Effets de seuil et non-linéarités

| Feature et seuil descriptif | Cible moyenne groupe bas | Cible moyenne groupe haut | Écart |
|---|---:|---:|---:|
| `walkDist` : 0–40,86 vs 2 911+ | 0,0929 | 0,8697 | +0,7767 |
| `weapons` : 0–1 vs 8+ | 0,1411 | 0,7440 | +0,6029 |
| `upgrades` : 0–1 vs 5+ | 0,3562 | 0,8907 | +0,5345 |
| `damages` : 0–18,83 vs 334,3+ | 0,3225 | 0,7885 | +0,4660 |
| `highestKill` : 0–3,748 vs 75,19+ | 0,3731 | 0,8051 | +0,4320 |
| `heals` : 0–1 vs 6+ | 0,3856 | 0,7662 | +0,3807 |
| `rideDist` : 0–630,3 vs 2 327+ | 0,4124 | 0,7369 | +0,3245 |

Ces seuils sont descriptifs et choisis à partir de quantiles/valeurs discrètes ; ils ne sont pas des seuils causaux ni des règles produit validées.

`killRank` est globalement décroissant, mais pas parfaitement monotone au milieu : la cible moyenne remonte dans les bandes 39–57. L'extrême est néanmoins net : rang kill 1–10 → cible moyenne 0,8005 ; rang 87–100 → 0,0334. Cette irrégularité justifie un modèle non linéaire et un audit de la définition exacte du rang.

## Effet du mode et du temps

Pour les six modes principaux, la cible moyenne varie de 0,4525 (`squad`) à 0,4890 (`duo-fpp`). Les taux de cible ≥0,9 restent proches de 10 %. Le mode a donc un effet faible sur la cible normalisée, même s'il transforme fortement les distributions des autres features.

Les moyennes mensuelles sont : janvier 0,4718, février 0,4711, mars 0,4751, avril 0,4711. Aucune tendance n'est détectée. La stabilité temporelle soutient l'utilisation de mai comme test futur représentatif, sans garantir les périodes au-delà.

Les trois régimes de score ont des cibles moyennes proches : 0,4710 (`rankPts_only`), 0,4730 (`legacy_kill_win_only`) et 0,4993 (`both_systems`, seulement 969 lignes). Le régime n'explique donc pas la cible, mais doit être traité pour éviter la colinéarité.

## Variables les plus utiles selon le cas d'usage

### Analyse a posteriori de la performance finale

`walkDist`, `killRank`, `upgrades`, `weapons`, `heals`, `damages`, `kills` et `rideDist` sont les candidats principaux. Un modèle non linéaire est approprié, avec gestion explicite de la sparsité et de la redondance.

### Prédiction précoce ou intervention en cours de partie

`killRank` doit être exclu et `walkDist`, `weapons`, `heals`, `upgrades`, `damages`, `kills` doivent être limités à la fenêtre d'observation disponible. Sinon le modèle prédit essentiellement la progression déjà réalisée, pas le futur résultat.

## Limites d'interprétation

1. Aucune relation présentée ne prouve une causalité.
2. La survie latente est un facteur commun non observé qui augmente simultanément distance, collecte, soins et rang final.
3. Les p-values deviennent minuscules avec 50 000 lignes, même pour des effets négligeables.
4. Les variables de rang ou de score peuvent être postérieures à l'événement prédit.
5. Les effets doivent être revalidés par mode et sur un véritable holdout temporel groupé par partie.
