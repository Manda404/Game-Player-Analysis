# 06 — Analyse des KPI

## Périmètre

Les KPI sont calculés au niveau **participation joueur-partie**, unité probable d'une ligne. Ils décrivent l'activité observée dans la partie, pas la fréquence de jeu d'un compte au fil du temps. Les résultats sources se trouvent dans [`kpi_summary.csv`](../reports/independent_raw_analysis/tables/kpi_summary.csv), [`kpi_by_month.csv`](../reports/independent_raw_analysis/tables/kpi_by_month.csv) et [`kpi_by_gameType.csv`](../reports/independent_raw_analysis/tables/kpi_by_gameType.csv).

## KPI de résultat

| KPI | Formule | Résultat | Interprétation / utilité | Limite |
|---|---|---:|---|---|
| Rang normalisé moyen | `mean(winRankPercentage)` | **0,472333** | Niveau central de performance ; utile pour suivre une population comparable | Masque le mode et la taille de partie ; pas une probabilité de victoire |
| Taux de rang élevé | `mean(target >= 0,9)` | **10,498 %** | Part des participations dans le haut du classement normalisé | « Top 10 % » de l'échelle, pas nécessairement top 10 joueurs |
| Taux de cible 1 | `mean(target == 1)` | **2,860 %** | Part des lignes gagnantes/placées premières | Plusieurs membres d'une équipe partagent la victoire ; échantillon de parties incomplet |
| Taux de cible 0 | `mean(target == 0)` | **4,900 %** | Part au dernier rang normalisé | Dépend de la granularité `maxRank` |

Le taux de rang élevé est stable par mois : 10,132 % en février à 10,856 % en mars. Le rang moyen varie seulement de 0,4711 à 0,4751.

## KPI d'activité et mobilité

| KPI | Formule | Résultat | Interprétation / utilité | Limite |
|---|---|---:|---|---|
| Distance totale moyenne | `mean(walkDist + rideDist + swimDist)` | **1 761,907** unités | Intensité de mobilité/exposition au jeu | Très asymétrique ; la moyenne est tirée par les véhicules |
| Distance totale médiane | `median(totalDistance)` | **792,000** unités | Joueur central, plus robuste que la moyenne | Ne distingue pas marche, véhicule et nage |
| Taux sans déplacement | `mean(totalDistance == 0)` | **2,136 %** | Détecte inactivité, élimination immédiate ou télémétrie manquante | 102 lignes ont pourtant du combat ; ne pas assimiler automatiquement à AFK |
| Durée moyenne de partie | `mean(gameTime) / 60` | **26,304 min** | Contexte de durée des matchs et comparaison des modes | `gameTime` n'est probablement pas le temps de jeu/survie individuel |

La forte différence moyenne/médiane de distance confirme la longue queue de mobilité. Ce KPI doit toujours être accompagné de la médiane et, en production, de quantiles.

## KPI de combat

| KPI | Formule | Résultat | Interprétation / utilité | Limite |
|---|---|---:|---|---|
| Participation au combat létal | `mean(kills > 0)` | **43,034 %** | Part des observations avec au moins une élimination | N'inclut pas les joueurs qui infligent des dégâts sans kill |
| Kills moyens | `mean(kills)` | **0,9210** | Intensité offensive moyenne | Distribution très asymétrique ; médiane 0 |
| Dégâts moyens | `mean(damages)` | **130,002** | KPI plus gradué que les kills | 53 lignes combinent kill positif et zéro dégât |
| Part agrégée de headshots | `sum(headshots) / sum(kills)` | **24,455 %** | Indicateur d'adresse si `headshots` désigne des headshot kills | Ce n'est pas la moyenne des ratios individuels ; définition métier à confirmer |

Parmi les observations avec cible ≥0,9, les kills moyens sont **2,6138** contre **0,7225** ailleurs, soit 3,62 fois plus. Les dégâts moyens sont **318,561** contre **107,886**, soit 2,95 fois plus. Ces écarts décrivent une association avec la performance, pas l'effet causal d'une intervention.

## KPI de soutien et de ressources

| KPI | Formule | Résultat | Interprétation / utilité | Limite |
|---|---|---:|---|---|
| Participation au soutien | `mean(assists + revives > 0)` | **25,832 %** | Mesure l'implication coopérative en équipe | Non applicable structurellement en solo ; quelques valeurs solo positives demandent une validation sémantique |
| Soins moyens | `mean(heals)` | **1,3770** | Utilisation de ressources de survie | 59,542 % de zéros ; très dépendant du temps passé en jeu |
| Armes moyennes | `mean(weapons)` | **3,6641** | Intensité de collecte/loot | « Acquises » ou « utilisées » n'est pas précisé ; maximum anormal de 73 |

Le taux de soutien illustre l'effet du mode : environ **31,72 %** dans `squad`/`squad-fpp`, **24,56–26,15 %** en duo, et seulement **4,61–5,11 %** en solo. Un benchmark global de soutien serait donc trompeur ; il faut le suivre par famille de mode.

## KPI discriminants du haut de classement

| KPI | Cible ≥0,9 | Cible <0,9 | Ratio haut/autres | Lecture |
|---|---:|---:|---:|---|
| Kills moyens | 2,6138 | 0,7225 | **3,62×** | Le haut de classement combine davantage de combat |
| Dégâts moyens | 318,561 | 107,886 | **2,95×** | Signal de combat robuste |
| Distance totale moyenne | 4 195,295 | 1 476,487 | **2,84×** | Forte association avec progression/survie et mobilité |

La distance reste un signal très fort même lorsque les kills sont contrôlés par bande. Le tableau d'interaction du document `04` indique qu'un joueur très mobile sans kill a une cible moyenne de 0,7662, supérieure à celle d'un joueur Q3 avec 3+ kills (0,6890). Cette observation suggère que la survie/positionnement compte au moins autant que le combat brut.

## KPI temporels

| Mois | Rang moyen | Cible ≥0,9 | Cible =1 | Distance moyenne | Kills moyens | Dégâts moyens |
|---|---:|---:|---:|---:|---:|---:|
| Janvier | 0,4718 | 10,395 % | 2,680 % | 1 754,1 | 0,9195 | 130,17 |
| Février | 0,4711 | 10,132 % | 2,652 % | 1 732,2 | 0,9100 | 128,87 |
| Mars | 0,4751 | 10,856 % | 3,125 % | 1 784,8 | 0,9241 | 130,77 |
| Avril | 0,4711 | 10,579 % | 2,971 % | 1 774,5 | 0,9301 | 130,10 |

Les variations sont petites et sans tendance monotone. Mars présente un léger pic, mais l'analyse statistique de la cible ne détecte pas d'effet mois.

## KPI impossibles à calculer avec ces données

| KPI demandé potentiel | Pourquoi il n'est pas mesurable | Données nécessaires |
|---|---|---|
| Rétention J1/J7/J30 | Les `playerId` répétés sont des collisions de format, pas des joueurs suivis | Identifiant joueur stable et journal de sessions |
| Churn | Aucune définition d'inactivité et aucun historique fiable par joueur | Sessions datées, fenêtre d'observation et définition métier |
| Fréquence de jeu | Une ligne ressemble à une participation, mais les joueurs ne se répètent pas réellement | Compte des sessions/parties par joueur |
| Durée de session joueur | `gameTime` est probablement la durée totale du match | Heure de début/fin de session ou temps de survie individuel |
| Progression/niveau | Aucune variable de niveau, XP ou historique de progression | Événements de progression par joueur |
| Conversion, ARPU, ARPPU | Aucun achat, revenu, devise ou statut payeur | Transactions et exposition aux offres |
| Lifetime value | Ni monétisation ni continuité d'identité | Identifiant stable, transactions, horizon longitudinal |

Créer des approximations de ces KPI à partir des présentes colonnes donnerait une précision factice. La priorité data est donc de relier une table de sessions et une table de transactions à un `playerId` fiable.

## Recommandations de suivi

1. Piloter `target_mean`, `top_rank_rate`, dégâts, kills et distance par `gameType` et période, avec médianes/P90.
2. Distinguer KPI de résultat (`target`) et KPI d'activité observée ; ne pas appeler `gameTime` « engagement joueur ».
3. Suivre les taux d'anomalie (`zero_distance_with_combat`, identifiants mal formés, régime de score) comme KPI de qualité.
4. Ne déclencher aucune action produit sur un seuil descriptif sans expérimentation contrôlée.
