# 03 — Analyse univariée

## Méthode

**Question.** Comment chaque variable importante se distribue-t-elle, où se concentrent les observations et quels seuils ou plafonds sont visibles ?

Pour chaque variable numérique, les statistiques calculées sont : moyenne, médiane, variance, écart-type, minimum, P1, P5, Q1, Q3, P95, P99, maximum, asymétrie, part de zéros et nombre d'outliers IQR. La table complète, incluant train et test, est disponible dans [`numeric_univariate_summary.csv`](../reports/independent_raw_analysis/tables/numeric_univariate_summary.csv).

## Synthèse numérique — train

| Variable | Moyenne | Médiane | Écart-type | P95 | P99 | Maximum | Zéros | Asymétrie |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `assists` | 0,231 | 0 | 0,589 | 1 | 3 | 11 | 82,718 % | 3,566 |
| `upgrades` | 1,105 | 0 | 1,713 | 5 | 7 | 14 | 56,864 % | 1,920 |
| `damages` | 130,002 | 84,700 | 168,761 | 457 | 766,602 | 3 725 | 27,810 % | 2,809 |
| `knocks` | 0,657 | 0 | 1,127 | 3 | 5 | 19 | 62,166 % | 2,873 |
| `headshots` | 0,225 | 0 | 0,596 | 1 | 3 | 20 | 83,228 % | 4,828 |
| `heals` | 1,377 | 0 | 2,706 | 7 | 13 | 41 | 59,542 % | 3,438 |
| `killRank` | 47,576 | 47 | 27,419 | 91 | 96 | 100 | 0 % | 0,025 |
| `killPts` | 504,010 | 0 | 625,913 | 1 486 | 1 666 | 2 120 | 59,680 % | 0,518 |
| `kills` | 0,921 | 0 | 1,545 | 4 | 7 | 35 | 56,966 % | 3,206 |
| `killStreaks` | 0,544 | 0 | 0,712 | 2 | 3 | 7 | 56,966 % | 1,221 |
| `highestKill` | 22,846 | 0 | 50,521 | 124,905 | 241,200 | 977,1 | 57,140 % | 3,929 |
| `gameTime` | 1 578,266 | 1 436 | 259,234 | 1 964 | 2 063 | 2 237 | 0 % | 0,281 |
| `maxRank` | 44,397 | 30 | 23,722 | 97 | 99 | 100 | 0 % | 1,365 |
| `numTeams` | 42,900 | 30 | 23,189 | 95 | 97 | 100 | 0 % | 1,372 |
| `rankPts` | 891,430* | 1 443* | 736,594* | 1 573 | 1 755 | 3 248 | 1,938 % | -0,368 |
| `revives` | 0,165 | 0 | 0,467 | 1 | 2 | 7 | 86,684 % | 3,496 |
| `rideDist` | 602,280 | 0 | 1 492,832 | 4 032 | 6 868,100 | 31 290 | 74,258 % | 3,427 |
| `roadKills` | 0,003 | 0 | 0,065 | 0 | 0 | 4 | 99,734 % | 26,071 |
| `swimDist` | 4,406 | 0 | 28,966 | 12,091 | 123,800 | 1 060 | 93,334 % | 12,455 |
| `teamKills` | 0,023 | 0 | 0,166 | 0 | 1 | 4 | 97,876 % | 8,132 |
| `vehicleDestr` | 0,008 | 0 | 0,094 | 0 | 0 | 3 | 99,212 % | 12,302 |
| `walkDist` | 1 155,222 | 683,750 | 1 183,299 | 3 384 | 4 395,020 | 11 260 | 2,204 % | 1,080 |
| `weapons` | 3,664 | 3 | 2,466 | 8 | 10 | 73 | 5,116 % | 2,321 |
| `winPts` | 606,701 | 0 | 739,526 | 1 561 | 1 634 | 1 953 | 59,680 % | 0,404 |
| `winRankPercentage` | 0,472 | 0,458 | 0,307 | 0,963 | 1 | 1 | 4,900 % | 0,102 |

\* Les statistiques brutes de `rankPts` ne sont pas interprétables comme une performance moyenne tant que les 19 191 sentinelles `-1` ne sont pas remplacées par manquant. La colonne contient aussi 969 vrais zéros apparents, soit 1,938 %.

## Lecture par famille de variables

### Combat

- La majorité ne réalise aucune élimination : `kills=0` pour 56,966 % des observations. La médiane est 0 et la moyenne 0,921 ; le maximum de 35 est cinq fois le P99 de 7.
- `damages` est moins concentré à zéro (27,810 %) et apporte donc un signal plus gradué que `kills`. La médiane de 84,7 est nettement sous la moyenne de 130,0, confirmant une longue queue à droite.
- `headshots`, `knocks`, `assists` et `highestKill` sont très asymétriques. Leur présence/absence peut être plus robuste que leur valeur brute dans certains modèles.
- `killRank` est presque symétrique entre 1 et 100 (asymétrie 0,025), contrairement aux autres variables de combat. Son caractère ordinal inversé — faible = meilleur — devra être explicité.
- `killStreaks` est zéro exactement aussi souvent que `kills` à 0 (56,966 %), cohérent avec une série impossible sans élimination.

### Déplacement et engagement

- `walkDist` est la composante principale : moyenne 1 155,2, médiane 683,75 ; seulement 2,204 % de zéros.
- `rideDist` décrit deux populations : 74,258 % n'utilisent aucun véhicule, alors que les utilisateurs peuvent dépasser 6 868 au P99. Une variable `used_vehicle` est justifiée.
- La nage est exceptionnelle : 93,334 % de zéros. `swimDist` doit être traitée en deux parties (`swam` + intensité positive).
- `gameTime` est la variable continue la moins asymétrique hors rangs : médiane 1 436 secondes, Q1 1 367, Q3 1 849. La concentration autour de plateaux de durée suggère des familles de modes ou des parties terminées/survécues différemment.

### Ressources et soutien

- 56,864 % des observations ont `upgrades=0`, 59,542 % `heals=0` et 5,116 % `weapons=0`.
- `weapons` a une médiane de 3 et un P99 de 10 ; le maximum de 73 est extrêmement isolé et nécessite un contrôle individuel.
- Les actions coopératives sont rares : 82,718 % sans assistance et 86,684 % sans réanimation. Leur lecture dépend fortement de `gameType` : en solo, ces zéros sont structurels et non un manque d'engagement.

### Contexte de partie

- `maxRank` et `numTeams` ont des médianes de 30, mais des P95 de 97 et 95. La distribution mélange les formats solo, duo et squad ; une moyenne globale masque donc des pics par mode.
- `maxRank` est toujours supérieur ou égal à `numTeams`, ce qui soutient leurs définitions probables.
- `killPts`, `winPts` et `rankPts` ne doivent pas être analysés isolément : leur distribution est dominée par le régime de score décrit dans le document qualité.

## Variables catégorielles

| `gameType` | Train | Part train | Test | Part test |
|---|---:|---:|---:|---:|
| `squad-fpp` | 19 767 | 39,534 % | 2 002 | 40,040 % |
| `duo-fpp` | 11 310 | 22,620 % | 1 113 | 22,260 % |
| `squad` | 6 999 | 13,998 % | 701 | 14,020 % |
| `solo-fpp` | 5 934 | 11,868 % | 614 | 12,280 % |
| `duo` | 3 518 | 7,036 % | 359 | 7,180 % |
| `solo` | 2 039 | 4,078 % | 188 | 3,760 % |
| Tous modes rares | 433 | 0,866 % | 23 | 0,460 % |

Les six modes principaux représentent **99,134 %** du train et **99,540 %** du test. `squad-fpp` seul compte pour environ 40 %. Les comparaisons de modes rares auront une très forte incertitude ; elles ne doivent pas fonder une action produit sans davantage de données.

Les trois autres variables catégorielles (`playerId`, `teamId`, `gameId`) sont des identifiants à forte cardinalité, non des segments comportementaux. Elles ne doivent pas être one-hot encodées.

## Comparabilité train/test à ce stade

Les médianes et parts de zéros sont très proches pour les variables centrales. Exemples : `kills` moyen 0,921 dans le train contre 0,892 dans le test ; dégâts moyens 130,002 contre 126,042 ; distance à pied moyenne 1 155,222 contre 1 146,163 ; durée moyenne 1 578,266 contre 1 572,285. Les maxima sont souvent plus faibles dans le test, résultat attendu avec dix fois moins de lignes. La dérive sera évaluée formellement par KS, PSI et différence moyenne standardisée dans `04_feature_relationships.md`.

## Limites et conséquences

- Une moyenne globale mélange des modes aux règles différentes.
- Les zéros peuvent signifier « aucun événement », « fonctionnalité non applicable » ou « score indisponible » selon la colonne.
- Les queues longues rendent les modèles linéaires sur valeurs brutes et les moyennes de petits segments fragiles.
- La règle IQR ne doit pas provoquer une suppression mécanique dans les variables zero-inflated.
- La target fait l'objet d'un document dédié car sa distribution résulte du rang et de la taille de partie.
