# 01 — Inventaire des données brutes

## Périmètre et méthode

**Question.** Quels fichiers constituent la source brute, quelle est leur structure et quel rôle analytique joue chaque colonne ?

**Méthode.** Lecture indépendante de tous les fichiers de `data/raw` avec `pandas.read_csv(sep=";")`, comptage des lignes, des colonnes, des valeurs distinctes, des valeurs nulles et des doublons exacts. La colonne `date` a aussi été convertie avec `errors="coerce"` afin de vérifier sa validité. Aucun fichier brut n'a été modifié.

**Date d'analyse :** 2026-08-01.

## Fichiers disponibles

| Fichier | Format | Taille disque | Lignes | Colonnes | Rôle probable | Période |
|---|---:|---:|---:|---:|---|---|
| `data/raw/train.csv` | CSV texte ASCII, séparateur `;`, fins de ligne CRLF | 8 235 082 octets | 50 000 | 30 | Jeu d'entraînement étiqueté | 2024-01-01 00:00:00 à 2024-04-30 00:00:00 |
| `data/raw/test.csv` | CSV texte ASCII, séparateur `;`, fins de ligne CRLF | 790 855 octets | 5 000 | 29 | Jeu de test non étiqueté | 2024-05-01 00:00:00 à 2024-05-31 00:00:00 |

Le schéma est identique entre les deux fichiers à une exception près : `winRankPercentage` est uniquement présent dans `train.csv`. C'est donc la **variable cible**. Les dates sont strictement postérieures dans le test ; les fichiers matérialisent un découpage temporel et non un échantillonnage aléatoire.

## Contrôles structurels immédiats

| Contrôle | `train.csv` | `test.csv` | Interprétation |
|---|---:|---:|---|
| Lignes entièrement dupliquées | 0 | 0 | Pas de doublon exact à supprimer |
| Cellules nulles reconnues par le parseur | 0 | 0 | Complétude syntaxique de 100 %, à distinguer des sentinelles métier comme `-1` |
| Dates invalides après conversion | 0 | 0 | Les 55 000 horodatages sont lisibles |
| Dates triées par ordre croissant | Oui | Oui | L'ordre des lignes porte une information temporelle |
| Colonnes strictement constantes | 0 | 0 | Toutes les colonnes varient |
| `playerId` distincts | 49 996 / 50 000 | 5 000 / 5 000 | Quatre répétitions dans le train, mais certaines sont dues à des identifiants manifestement altérés |
| `teamId` distincts | 49 407 | 4 974 | Identifiant de groupe ; faible répétition dans cet extrait |
| `gameId` distincts | 30 983 | 4 431 | Plusieurs joueurs d'une même partie peuvent être présents |

### Colonnes quasi constantes ou fortement concentrées

Aucune colonne n'est constante. Plusieurs variables de comptage ou de distance sont cependant dominées par zéro et devront être traitées comme variables **zero-inflated**, pas comme variables quasi constantes à supprimer automatiquement :

| Colonne | Part de la modalité dominante dans train | Part dans test | Modalité dominante probable |
|---|---:|---:|---:|
| `roadKills` | 99,734 % | 99,840 % | 0 |
| `vehicleDestr` | 99,212 % | 99,280 % | 0 |
| `teamKills` | 97,876 % | 97,980 % | 0 |
| `swimDist` | 93,334 % | 93,480 % | 0 |
| `revives` | 86,684 % | 86,940 % | 0 |
| `headshots` | 83,228 % | 83,380 % | 0 |
| `assists` | 82,718 % | 81,960 % | 0 |
| `rideDist` | 74,258 % | 74,980 % | 0 |

## Dictionnaire des variables

Les significations ci-dessous sont déduites des noms et des valeurs observées. Elles constituent des **hypothèses sémantiques** tant qu'aucun dictionnaire officiel n'est fourni.

| Colonne | Type lu | Type analytique | Valeurs distinctes train / test | Signification probable et rôle |
|---|---|---|---:|---|
| `playerId` | chaîne | identifiant technique | 49 996 / 5 000 | Identifiant du joueur ou de sa participation. Très forte cardinalité ; à exclure comme prédicteur brut. Des valeurs en notation scientifique avec virgule suggèrent une corruption de format. |
| `teamId` | chaîne | identifiant de groupe | 49 407 / 4 974 | Identifiant de l'équipe dans une partie. Sert au groupement et au contrôle de fuite, pas comme variable brute. |
| `gameId` | chaîne | identifiant de groupe | 30 983 / 4 431 | Identifiant de la partie. Plusieurs lignes peuvent partager une partie ; toute validation doit séparer les groupes et/ou le temps. Quelques valeurs ressemblent à des identifiants corrompus. |
| `assists` | entier | numérique discret | 12 / 6 | Nombre d'assistances réalisées. |
| `upgrades` | entier | numérique discret | 15 / 13 | Hypothèse : nombre d'objets/équipements améliorés ou de boosts consommés. Le terme exact doit être confirmé par le métier. |
| `damages` | réel | numérique continu | 10 675 / 2 455 | Dégâts infligés pendant la partie. |
| `knocks` | entier | numérique discret | 18 / 12 | Adversaires mis à terre, sans nécessairement les éliminer. |
| `headshots` | entier | numérique discret | 12 / 7 | Nombre d'éliminations ou impacts à la tête ; la définition exacte doit être confirmée. |
| `heals` | entier | numérique discret | 35 / 26 | Nombre d'objets/actes de soin utilisés. |
| `killRank` | entier | rang ordinal | 100 / 100 | Rang du joueur selon les éliminations dans la partie ; une valeur faible est a priori meilleure. |
| `killPts` | entier | score numérique avec sentinelles possibles | 988 / 669 | Score de classement lié aux éliminations ; les zéros peuvent signifier « non classé » plutôt qu'une performance nulle. |
| `kills` | entier | numérique discret | 24 / 14 | Nombre d'adversaires éliminés. |
| `killStreaks` | entier | numérique discret | 8 / 5 | Nombre maximal ou nombre de séries d'éliminations ; définition exacte à confirmer. |
| `highestKill` | réel | numérique continu | 11 494 / 1 976 | Très probablement la plus grande distance d'une élimination, malgré un nom ambigu. |
| `gameTime` | entier | durée | 1 077 / 689 | Durée de la partie ou temps de survie, probablement en secondes. |
| `gameType` | chaîne | catégorielle nominale | 16 / 10 | Mode de jeu : solo/duo/squad, perspective FPP/TPP et modes spéciaux. |
| `maxRank` | entier | numérique/ordinal de contexte | 96 / 67 | Taille/rang maximal théorique de la partie. |
| `numTeams` | entier | numérique discret de contexte | 98 / 72 | Nombre d'équipes dans la partie. |
| `rankPts` | entier | score avec sentinelle | 857 / 428 | Points de classement généraux ; `-1` est probablement une valeur « non classé/inconnu ». |
| `revives` | entier | numérique discret | 8 / 6 | Nombre de coéquipiers réanimés. |
| `rideDist` | réel | distance continue | 7 983 / 1 172 | Distance parcourue en véhicule. |
| `roadKills` | entier | numérique discret rare | 5 / 2 | Éliminations réalisées avec un véhicule. |
| `swimDist` | réel | distance continue | 2 960 / 325 | Distance parcourue à la nage. |
| `teamKills` | entier | numérique discret rare | 5 / 3 | Éliminations de coéquipiers ; métrique de comportement potentiellement toxique. |
| `vehicleDestr` | entier | numérique discret rare | 4 / 3 | Véhicules détruits. |
| `walkDist` | réel | distance continue | 16 662 / 3 977 | Distance parcourue à pied. |
| `weapons` | entier | numérique discret | 39 / 18 | Nombre d'armes acquises/utilisées ; définition exacte à confirmer. |
| `winPts` | entier | score avec sentinelles possibles | 500 / 280 | Points de classement liés aux victoires ; le zéro domine et peut coder « non classé ». |
| `winRankPercentage` | réel | cible continue bornée | 1 445 / absent | Position finale normalisée entre 0 et 1 ; 1 correspond probablement au meilleur classement. |
| `date` | chaîne, convertible en date | temporelle | 50 000 / 5 000 | Horodatage artificiellement régulier ou date d'observation. Train : janvier-avril ; test : mai. À utiliser pour dérive et validation temporelle. |

## Catégories observées dans `gameType`

Le train contient 16 modalités : les six modes principaux (`solo`, `duo`, `squad`, avec ou sans suffixe `-fpp`) dominent largement. Les modes spéciaux ou « normal » sont rares. Le test ne contient que 10 modalités : certaines catégories du train disparaissent donc dans la période de mai. Ce point sera quantifié dans l'analyse de qualité et de dérive.

## Target et unité d'observation

- **Target :** `winRankPercentage`, variable continue bornée observée dans le train uniquement.
- **Unité d'observation probable :** une participation d'un joueur à une partie, et non un joueur permanent. L'unicité presque totale de `playerId`, alors que `gameId` se répète, est cohérente avec cette lecture.
- **Conséquence méthodologique :** le test est temporellement futur. Une validation aléatoire surestimerait potentiellement la généralisation. La validation recommandée est temporelle, avec contrôle de groupement par `gameId`.
- **Risque de fuite structurelle :** `killRank`, `maxRank`, `numTeams`, `gameTime` et certains scores de classement peuvent contenir une information calculée après ou pendant la fin de la partie. Leur disponibilité au moment réel de la prédiction doit être clarifiée.

## Limites et prochaines vérifications

1. L'absence de nulls ne garantit pas l'absence de valeurs manquantes métier : `-1` et certains zéros doivent être audités comme sentinelles.
2. Les identifiants affichés sous la forme `5,44E+13`, `3,24E+13`, `4,46E+13`, etc. semblent avoir été altérés par un tableur. Il faut mesurer leur portée avant tout groupement.
3. Les relations arithmétiques attendues entre rangs, nombre d'équipes et cible doivent être testées pour détecter une cible reconstruisible ou une fuite.
4. La régularité exacte des timestamps sera analysée : elle peut signaler une date synthétique servant uniquement à imposer l'ordre train/test.
5. Les définitions de `upgrades`, `highestKill`, `killStreaks` et `weapons` doivent être confirmées auprès du propriétaire des données.
