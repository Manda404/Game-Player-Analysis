# 08 — Principaux enseignements

## Synthèse exécutive

L'échantillon contient 50 000 participations étiquetées de janvier à avril 2024 et 5 000 participations non étiquetées en mai. La performance finale est beaucoup plus associée à un ensemble **mobilité + collecte + survie apparente** qu'au combat seul. `walkDist` est la variable la plus monotone avec la cible (`ρ=0,866`), devant `killRank` (`ρ=-0,715`), `upgrades` (`ρ=0,681`) et `weapons` (`ρ=0,665`). Cette observation est descriptive : survivre plus longtemps permet aussi de marcher et de collecter davantage.

Trois profils comportementaux stables émergent : 49,3 % de participations à faible activité et faible rang moyen (0,253), 29,4 % orientées mobilité/collecte avec peu de combat mais bon rang (0,652), et 21,3 % très actives en combat et polyvalentes (0,732). La cible n'a pas servi à créer les segments.

Les données sont numériquement stables entre train et test, mais comportent deux défauts structurels : identifiants corrompus par notation scientifique et coexistence de systèmes de classement signalée par `-1`/`0`. Elles ne sont pas longitudinales au niveau joueur ; rétention, churn et monétisation sont donc hors périmètre mesurable.

## Dix constats prioritaires

### 1. La cible est un rang final normalisé et discrétisé

Les 50 000 valeurs sont dans `[0,1]`. La formule `1 + (1-target) × (maxRank-1)` produit une place quasi entière sur 100 % des lignes compte tenu de l'arrondi à quatre décimales. La moyenne est 0,4723, la médiane 0,4579 et l'écart-type 0,3074.

**Conséquence.** Utiliser une régression bornée, rapporter MAE et RMSE, contrôler les bornes et tenir compte d'une granularité dépendante de `maxRank`.

### 2. Mobilité et collecte structurent davantage le rang que les kills seuls

Le décile supérieur de `walkDist` atteint une cible moyenne de 0,8697 contre 0,0929 dans le décile inférieur. À l'intérieur de chaque quartile de marche, davantage de kills reste favorable, mais l'écart entre quartiles de marche est plus grand que l'écart entre bandes de kills.

Les joueurs avec cible ≥0,9 parcourent en moyenne 4 195 unités contre 1 476 pour les autres (2,84×), réalisent 2,614 kills contre 0,722 (3,62×) et infligent 318,6 dégâts contre 107,9 (2,95×).

**Interprétation métier.** Le classement récompense ou reflète une stratégie combinant positionnement, progression et combat, pas l'agression pure.

### 3. `killRank` est très informatif mais potentiellement inutilisable en production

`killRank` a `ρ=-0,715` avec la cible et une information mutuelle de 0,913. Il s'agit probablement d'un rang calculé après observation des autres joueurs.

**Risque.** Pour une prédiction précoce, cette feature est une fuite. Le cas d'usage et l'instant de scoring doivent être fixés avant toute sélection de variables.

### 4. Le mode de jeu change le contexte, très peu la cible normalisée

`gameType` explique 96,8 % de la variance de `maxRank` et 95,7 % de `numTeams`, mais seulement 0,23 % de la cible. Les p-values sont minuscules avec 50 000 lignes, alors que η² et V de Cramér montrent un effet métier faible.

**Conséquence.** Le mode est indispensable pour contextualiser les KPI de soutien et de taille de partie, mais ce n'est pas un levier principal de rang normalisé.

### 5. Trois segments décrivent des stratégies différentes

| Segment | Part | Caractéristique | Cible moyenne | Action potentielle |
|---|---:|---|---:|---|
| Faible activité | 49,252 % | Peu de mouvement, loot et combat | 0,2528 | Onboarding, premières actions, diagnostic technique |
| Mobiles/collecte | 29,406 % | Forte mobilité et soins, combat sous la moyenne | 0,6518 | Objectifs de carte/survie, entraînement combat optionnel |
| Combattants actifs | 21,342 % | Combat, soutien et progression très élevés | 0,7315 | Défis avancés, compétition, maîtrise |

Les segments expliquent 50,3 % de la variance de cible sans l'avoir utilisée. Leur ordre persiste dans les six modes principaux. Le silhouette de 0,194 rappelle toutefois qu'il s'agit d'un continuum.

### 6. Les identifiants ont subi une corruption irréversible

Dans le train, 93 `playerId`, 87 `teamId` et 93 `gameId` sont mal formés ; dans le test, respectivement 14, 8 et 10. Des valeurs comme `5,44E+13` fusionnent plusieurs hashes. Tous les `playerId` répétés du train sont de telles collisions.

**Conséquence.** Impossible de mesurer la rétention. Les identifiants mal formés ne doivent pas servir au groupement ni être reconstruits arbitrairement.

### 7. Deux systèmes de scores sont mélangés

59,680 % du train ont `killPts=winPts=0` avec `rankPts` disponible ; 38,382 % ont `rankPts=-1` avec les anciens scores disponibles ; 1,938 % ont les trois. Les proportions du test sont très proches.

**Conséquence.** Les scores bruts créent des VIF de 32 à 130 et des corrélations artificielles. Il faut encoder le régime et traiter les valeurs comme manquantes conditionnelles.

### 8. La multicolinéarité est localisée et traitable

`maxRank`/`numTeams` ont Pearson 0,9979 et VIF ≈247. Le bloc combat contient aussi des redondances : `kills`/`killStreaks` Spearman 0,9709, `kills`/`highestKill` 0,9491, VIF de `kills` 8,14.

**Conséquence.** Sélection/régularisation pour les modèles linéaires ; importance groupée et permutation pour les arbres ; ne pas additionner ces métriques comme KPI indépendants.

### 9. Le test de mai ressemble au train

PSI maximal 0,00354, différence moyenne standardisée absolue maximale 0,0279 et aucune p-value KS sous 0,289. Les proportions de modes principaux varient de moins de 0,51 point.

**Conséquence.** Le holdout temporel est représentatif des quatre mois observés, sans garantie pour d'autres saisons ou changements de produit.

### 10. Les KPI longitudinaux et financiers manquent

Les données ne permettent pas de calculer fréquence, rétention, churn, conversion, ARPU, ARPPU ou LTV. `gameTime` est probablement la durée totale du match et n'est pas corrélé à la cible (`ρ=0,004`), donc ce n'est pas une durée de session joueur.

**Conséquence.** Toute ambition CRM, live ops ou monétisation nécessite un `playerId` stable, des sessions et des transactions.

## KPI à retenir

| Domaine | KPI de référence | Valeur actuelle |
|---|---|---:|
| Résultat | Rang normalisé moyen | 0,4723 |
| Résultat | Taux cible ≥0,9 | 10,498 % |
| Résultat | Taux cible =1 | 2,860 % |
| Mobilité | Distance totale moyenne / médiane | 1 761,9 / 792,0 |
| Qualité/activité | Distance nulle | 2,136 % |
| Combat | Au moins un kill | 43,034 % |
| Combat | Kills / dégâts moyens | 0,921 / 130,0 |
| Précision | Part agrégée de headshots | 24,455 % |
| Coopération | Assistance ou revive | 25,832 % global ; à suivre par mode |

## Anomalies majeures

1. 19 191 `rankPts=-1` dans le train et 1 888 dans le test.
2. Identifiants convertis en notation scientifique, avec collisions réelles.
3. 53 lignes train et 5 test avec kills positifs et zéro dégât.
4. 102 lignes train et 5 test avec distance nulle mais activité de combat.
5. 5 lignes train avec moins de 10 unités parcourues et au moins 5 kills.
6. Extrêmes : 35 kills, 3 725 dégâts, 31 290 de distance véhicule, 73 armes.
7. Dates uniformément espacées, probablement synthétiques plutôt qu'horodatages d'événements.

## Hypothèses à faire valider

| Hypothèse | Niveau de confiance | Validation nécessaire |
|---|---|---|
| `winRankPercentage` = place normalisée | Très élevé | Dictionnaire métier/formule officielle |
| `gameTime` = durée totale du match | Élevé | Définition source et niveau d'agrégation |
| `highestKill` = distance du kill le plus lointain | Moyen | Documentation de télémétrie |
| `upgrades` = boosts/améliorations consommés | Faible à moyen | Nom source original et définition |
| `date` = index temporel synthétique | Élevé | Pipeline de génération des fichiers |
| `rankPts=-1`, `killPts=winPts=0` = indisponibilité par régime | Très élevé | Historique des systèmes de ranking |
| `playerId` = identifiant de participation plutôt que compte persistant | Moyen | Contrat de schéma et source du hash |

## Conclusion métier

La meilleure lecture des données est celle d'un jeu où **rester actif, se déplacer et collecter** est une condition forte de bon classement, tandis que le combat différencie davantage les meilleurs au sein d'un même niveau de progression. La plus grande opportunité produit apparente concerne la moitié des participations à faible activité, mais on ne sait pas encore si elles correspondent à des débutants, déconnexions, abandons ou simples mauvaises parties. L'action immédiate doit donc être double : réparer l'observabilité, puis tester des interventions d'onboarding ciblées plutôt que conclure causalement à partir des corrélations.
