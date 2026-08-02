# 04 — Relations entre les features

## Méthodes

**Question.** Quelles variables sont redondantes, dépendantes ou interactives, et quelles relations risquent de biaiser l'interprétation ou la modélisation ?

Les méthodes appliquées sont : corrélations de Pearson et Spearman, VIF, ANOVA et Kruskal-Wallis, taille d'effet η², chi-deux et V de Cramér, tableaux croisés, analyse conjointe `walkDist × kills`, cohérence intra-groupe et dérive train/test par KS, différence moyenne standardisée et PSI.

Les tables complètes sont disponibles dans [`numeric_pair_correlations.csv`](../reports/independent_raw_analysis/tables/numeric_pair_correlations.csv), [`variance_inflation_factors.csv`](../reports/independent_raw_analysis/tables/variance_inflation_factors.csv), [`game_type_numeric_associations.csv`](../reports/independent_raw_analysis/tables/game_type_numeric_associations.csv), [`categorical_associations.csv`](../reports/independent_raw_analysis/tables/categorical_associations.csv) et [`walk_kills_target_interaction.csv`](../reports/independent_raw_analysis/tables/walk_kills_target_interaction.csv).

## Dépendances entre variables numériques

### 1. Taille et structure de partie

`maxRank` et `numTeams` sont presque la même information : Pearson **0,9979**, Spearman **0,9671**, VIF respectifs **247,63** et **246,60**. La différence entre les deux représente probablement la taille moyenne des équipes et les groupes incomplets.

**Impact.** Dans un modèle linéaire, conserver les deux coefficients les rend instables. Recommandation : garder `maxRank`, ou construire `players_per_team_proxy = maxRank / numTeams`, puis vérifier par validation. Pour un modèle d'arbres, la redondance est moins destructrice mais dilue l'importance.

### 2. Systèmes de classement

Les trois scores sont artificiellement colinéaires parce que leur disponibilité s'exclut presque mutuellement :

| Paire | Pearson | Spearman |
|---|---:|---:|
| `rankPts` / `winPts` | -0,9939 | -0,8373 |
| `killPts` / `winPts` | 0,9834 | 0,9453 |
| `killPts` / `rankPts` | -0,9756 | -0,8373 |

Les VIF sont 130,46 pour `winPts`, 89,69 pour `rankPts` et 32,41 pour `killPts`. Cette relation ne signifie pas qu'un meilleur `rankPts` réduit réellement `winPts`; elle provient avant tout des sentinelles et des régimes de score.

**Traitement.** Construire `ranking_system`, remplacer les valeurs indisponibles par manquant, puis calculer éventuellement des scores normalisés à l'intérieur de chaque régime. Ne pas inclure les trois colonnes brutes dans une régression.

### 3. Bloc combat

Les relations les plus fortes sont cohérentes avec des métriques imbriquées :

- `kills` / `killStreaks` : Spearman **0,9709** ;
- `kills` / `highestKill` : **0,9491** ;
- `killRank` / `kills` : **-0,8842** ;
- `killRank` / `killStreaks` : **-0,8566** ;
- `damages` / `kills` : Pearson **0,8867**, Spearman **0,7983** ;
- `damages` / `knocks` : Pearson **0,7337** ;
- `knocks` / `kills` : Pearson **0,7105**.

Les corrélations de rang très hautes sont renforcées par les zéros partagés. `kills`, `killStreaks` et `highestKill` ne sont donc pas interchangeables dans tous les cas : `highestKill` conserve une information de distance parmi les joueurs qui ont tué, et `damages` est plus gradué que `kills`.

Le VIF de `kills` est **8,14**, celui de `damages` **5,97**, de `killRank` **5,12** et de `killStreaks` **4,92**. Pour une régression, il faut sélectionner, régulariser ou regrouper ces features. Pour une analyse métier, les KPI doivent éviter de compter plusieurs fois la même dimension « intensité de combat ».

### 4. Collecte, soin et mobilité

`walkDist` est corrélé à `weapons` (**ρ=0,7174**) et à `upgrades` (**ρ=0,6871**). `upgrades` est également lié à `heals` (**ρ=0,6931**). Ce bloc décrit vraisemblablement l'exposition au jeu : plus un joueur reste actif longtemps, plus il se déplace, collecte et utilise des ressources.

**Limite majeure.** Ces relations sont compatibles avec une causalité inversée ou un facteur commun : survivre longtemps permet de marcher et de looter. Elles ne prouvent pas qu'accorder davantage d'armes ou de soins causerait un meilleur classement.

## Relations catégorielles

### `gameType` et variables numériques

Le mode explique presque entièrement la structure de partie : η² **0,9678** pour `maxRank` et **0,9569** pour `numTeams`. Il explique plus modestement `knocks` (7,59 % de variance), `weapons` (4,29 %), `assists` (4,22 %), `gameTime` (2,98 %), `roadKills` (2,81 %) et `revives` (2,71 %).

En revanche, il n'explique que **0,2275 %** de la variance de `winRankPercentage`. L'ANOVA (`p=2,91×10⁻¹⁷`) et Kruskal-Wallis (`p=3,48×10⁻¹⁹`) sont significatifs à cause des 50 000 lignes, mais l'effet métier est négligeable. Le V de Cramér entre `gameType` et les quintiles de cible n'est que **0,0320**.

### Autres associations catégorielles

| Association | V de Cramér | p-value chi-deux | Lecture |
|---|---:|---:|---|
| `gameType` / quintile cible | 0,0320 | 1,50×10⁻²⁷ | Statistiquement détectable, très faible |
| `gameType` / `ranking_system` | 0,0453 | 1,04×10⁻³³ | Faible dépendance des régimes de score au mode |
| `gameType` / mois | 0,0000 corrigé | 0,500 | Composition stable entre janvier et avril |
| `ranking_system` / mois | 0,0015 | 0,399 | Pas de migration temporelle observable |
| `ranking_system` / quintile cible | 0,0098 | 0,025 | Effet négligeable malgré une p-value sous 5 % |

Certaines cellules de modes rares sont vides ; le chi-deux est donc surtout descriptif pour ces associations. Le résultat ne justifie aucune conclusion sur les petits modes spéciaux.

## Interaction entre déplacement et combat

La cible moyenne augmente à la fois avec le quartile de `walkDist` et le nombre de kills :

| Quartile `walkDist` | 0 kill | 1 kill | 2 kills | 3+ kills |
|---|---:|---:|---:|---:|
| Q1 — plus faible | 0,1288 (n=9 933) | 0,1823 | 0,2029 | 0,3091 (n=140) |
| Q2 | 0,3082 | 0,3356 | 0,3632 | 0,4062 |
| Q3 | 0,5600 | 0,6063 | 0,6373 | 0,6890 |
| Q4 — plus élevé | 0,7662 (n=4 259) | 0,8224 | 0,8461 | 0,8917 (n=3 448) |

À nombre de kills fixé, passer de Q1 à Q4 est associé à environ +0,58 à +0,64 de cible. À déplacement fixé, passer de 0 à 3+ kills est associé à +0,13 à +0,18. La combinaison Q4/3+ atteint 0,8917, contre 0,1288 pour Q1/0 kill.

**Interprétation.** Le déplacement/survie apparente structure davantage le rang que le combat seul. Mais `walkDist` est accumulée pendant la partie : une meilleure survie donne davantage de temps pour marcher. Cette table ne permet donc pas de conclure qu'imposer davantage de déplacement améliore causalement la performance.

## Dépendance de groupe et risque de validation

Parmi les 585 `teamId` répétés, **581 groupes (99,316 %) ont une cible identique** pour tous leurs membres observés. Cela confirme que la place est attribuée au niveau équipe. Les quatre groupes non constants peuvent provenir d'identifiants corrompus ou d'une anomalie.

`gameId` est répété dans 13 493 groupes et 32 510 lignes ; la cible varie normalement entre équipes d'une même partie (écart-type intra-partie moyen 0,2605). Un split aléatoire place donc des joueurs de la même partie — et parfois de la même équipe — des deux côtés de la validation.

**Recommandation.** Le jeu de validation doit être temporel et, à l'intérieur d'une fenêtre, groupé par `gameId`. Pour les `gameId` mal formés, chaque ligne doit être traitée comme groupe indépendant ou exclue du calcul de groupe.

Les ensembles train et test ne partagent aucun `playerId`, `teamId` ou `gameId`, ce qui empêche la mémorisation directe d'identifiants entre les deux périodes.

## Dérive train/test

Sur les 24 features numériques communes :

- PSI maximal : **0,00354** (`maxRank`), très inférieur au seuil usuel de vigilance 0,10 ;
- différence moyenne standardisée absolue maximale : **0,02793** (`roadKills`) ;
- statistique KS maximale : **0,01454** (`gameTime`) ; toutes les p-values KS sont ≥ 0,289.

Il n'existe donc **aucune dérive numérique mesurable importante** entre janvier-avril et mai. La composition des six modes majeurs varie au maximum de +0,506 point (`squad-fpp`) ou -0,360 point (`duo-fpp`). Six modes très rares du train sont absents du test, sans impact de volume significatif.

La cible elle-même est stable dans le train : moyenne mensuelle entre 0,4711 et 0,4751, ANOVA `p=0,678`, η² `0,000030`, Spearman date/cible `ρ=-0,0010`, `p=0,825`.

## Risques de modélisation

1. **Multicolinéarité :** taille de partie, scores de classement et bloc combat.
2. **Fuite temporelle :** `killRank` et possiblement `gameTime`/scores selon l'instant de prédiction.
3. **Fuite de groupe :** membres de la même partie ou équipe dans plusieurs folds.
4. **Causalité inversée :** distance, armes, soins et upgrades sont aussi des conséquences du temps passé en jeu.
5. **Significativité trompeuse :** avec 50 000 lignes, des effets minuscules ont de très faibles p-values ; toujours lire η², V de Cramér et les écarts absolus.

![Carte des corrélations de Spearman](../reports/independent_raw_analysis/figures/target_spearman_heatmap.png)
