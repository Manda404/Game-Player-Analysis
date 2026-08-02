# 02 — Analyse de la qualité des données

## Méthode et traçabilité

**Question.** Les données sont-elles suffisamment fiables pour calculer des KPI et entraîner un modèle sans introduire de biais ou de fuite ?

**Méthode.** Contrôles exhaustifs sur les 50 000 lignes du train et les 5 000 lignes du test : nulls syntaxiques, doublons, validité des identifiants, bornes physiques, sentinelles, cohérences entre colonnes, fréquence des catégories, validité des dates et règles logiques. Les valeurs extrêmes sont repérées par quantiles et par règle IQR, mais jamais supprimées automatiquement.

Les résultats auditables se trouvent dans :

- [`data_quality_checks.csv`](../reports/independent_raw_analysis/tables/data_quality_checks.csv) ;
- [`raw_file_fingerprints.json`](../reports/independent_raw_analysis/tables/raw_file_fingerprints.json) ;
- [`date_quality.json`](../reports/independent_raw_analysis/tables/date_quality.json) ;
- [`target_quantisation_check.json`](../reports/independent_raw_analysis/tables/target_quantisation_check.json).

## Diagnostic exécutif

Les fichiers sont **complets au sens technique** — aucun null, aucun doublon exact, aucune date illisible — mais pas propres au sens métier. Trois problèmes ont un impact direct : des identifiants transformés en notation scientifique, deux régimes de scores codés par `-1` et `0`, et quelques contradictions entre combat et déplacement. La colonne `date` est presque certainement synthétique. Enfin, plusieurs variables semblent calculées après la partie et peuvent constituer une fuite selon le moment où la prédiction est censée être faite.

## Problèmes détectés et traitements recommandés

| Priorité | Colonne(s) | Train | Test | Constat mesuré | Impact potentiel | Traitement recommandé |
|---:|---|---:|---:|---|---|---|
| P0 | `playerId` | 93 valeurs mal formées (0,186 %) | 14 (0,280 %) | Des hashes ont été convertis en chaînes comme `5,44E+13`, voire `3,89E+109` ou `1,70E+260`. | Identifiants distincts fusionnés ; faux joueurs récurrents ; groupements et détection de fuite corrompus. | Ne jamais tenter de reconstruire les hashes. Ajouter `playerId_malformed`; remplacer par un identifiant de ligne uniquement si un identifiant est techniquement requis ; exclure la valeur brute du modèle. |
| P0 | `teamId` | 87 (0,174 %) | 8 (0,160 %) | Même corruption de notation scientifique. | Faux regroupements d'équipe, fuite ou validation groupée incorrecte. | Ajouter un flag et considérer chaque valeur corrompue comme non fiable pour le groupement. |
| P0 | `gameId` | 93 (0,186 %) | 10 (0,200 %) | Même corruption ; seulement 56 valeurs mal formées distinctes dans le train, ce qui prouve des collisions. | Des parties indépendantes peuvent être fusionnées. | Ne pas grouper sur les valeurs mal formées ; utiliser un surrogate row-level ou exclure ces lignes des contrôles de groupe. |
| P0 | `rankPts`, `killPts`, `winPts` | 19 191 `rankPts=-1` (38,382 %) | 1 888 (37,760 %) | `-1` est une sentinelle, pas un score. | Moyennes et corrélations artificielles ; fausse relation ordinale. | Créer `rankPts_missing`, remplacer `-1` par manquant pour les calculs continus, conserver le régime de score comme feature catégorielle. |
| P0 | mêmes scores | 29 840 lignes (59,680 %) avec `killPts=winPts=0` et `rankPts` disponible ; 19 191 (38,382 %) avec `rankPts=-1` et les deux anciens scores disponibles ; 969 (1,938 %) avec les trois disponibles | 3 000 (60,000 %) ; 1 888 (37,760 %) ; 112 (2,240 %) | Les données combinent au moins deux systèmes de classement. Les zéros de `killPts`/`winPts` codent l'indisponibilité, non une performance nulle. | Mélange de populations/régimes et risque de proxy temporel ou de mode de jeu. | Construire un `ranking_system` à trois modalités et traiter les scores conditionnellement. Ne jamais imputer tous les zéros comme de vrais scores. |
| P1 | `kills`, `damages` | 53 lignes (0,106 %) | 5 (0,100 %) | `kills>0` alors que `damages=0`. | Télémétrie contradictoire, exception métier ou erreur d'agrégation. | Conserver un flag ; faire confirmer si certaines éliminations n'infligent pas de dégâts ; analyse de sensibilité avec/sans ces lignes. |
| P1 | distances, combat | 102 lignes (0,204 %) | 5 (0,100 %) | Distance totale nulle avec `kills>0` ou `damages>0`. | AFK impossible avec combat, valeur arrondie ou télémétrie incomplète. | Flag `zero_distance_with_combat`; ne supprimer qu'après validation métier. |
| P1 | distances, `kills` | 5 lignes (0,010 %) | 0 | Moins de 10 unités de distance et au moins 5 kills. | Cas fortement suspects pouvant déformer les segments et modèles. | Audit individuel ; clipping robuste pour la segmentation ; exclusion seulement dans un scénario de sensibilité. |
| P1 | `date` | cadence médiane 205,636 s ; 9 écarts distincts arrondis ; un écart maximal de 86 605,636 s | cadence constante de 518,503701 s | Dates ordonnées, uniformément réparties sur la fenêtre, sans timestamps répétés. Le test est parfaitement régulier. | La date ressemble à un index synthétique et ne doit pas être interprétée comme l'heure réelle d'une partie. | Utiliser uniquement pour respecter l'ordre temporel et mesurer la dérive ; ne pas dériver heure/jour de semaine sans preuve métier. |
| P2 | `gameType` | 10 modalités rares totalisant 433 lignes (0,866 %) | 4 modalités rares totalisant 23 lignes (0,460 %) | Les six modes principaux couvrent 99,134 % du train. | Estimates instables, encodage fragile, catégories absentes du test. | Regrouper les modes rares en famille `special/normal` pour certaines analyses ; conserver la valeur détaillée pour audit. |
| P2 | variables zero-inflated | jusqu'à 99,734 % de zéros | jusqu'à 99,840 % | `roadKills`, `vehicleDestr`, `teamKills`, `swimDist`, etc. sont très rares. | La moyenne et la règle IQR sont peu informatives ; risque de sur-apprentissage sur des événements rares. | Combiner flag binaire (`>0`) et valeur transformée `log1p`; régulariser ou exclure selon validation. |

Les pourcentages utilisent le nombre de lignes de chaque fichier comme dénominateur, sauf mention contraire.

## Contrôles satisfaisants

- 0 doublon exact dans les deux fichiers.
- 0 cellule vide reconnue par le parseur et 0 date invalide.
- 0 valeur négative dans les compteurs, distances, dégâts, durées et la cible ; la seule valeur négative observée est la sentinelle `rankPts=-1`.
- 100 % des valeurs de `winRankPercentage` sont dans `[0,1]`.
- 0 ligne où `headshots > kills`, `killStreaks > kills` ou `roadKills > kills`.
- 0 ligne où `maxRank < numTeams`.
- 0 partie avec `maxRank`, `numTeams` ou `gameTime` non positif.

Ces validations soutiennent l'hypothèse que `headshots` représente bien des headshot kills et que `killStreaks`/`roadKills` sont des sous-comptes de `kills`.

## Identifiants dupliqués : faux signal de rétention

Le train contient 49 996 `playerId` distincts pour 50 000 lignes. Les sept lignes portant un identifiant répété correspondent uniquement à trois chaînes corrompues : `5,44E+13` apparaît trois fois, `3,24E+13` et `2,68E+13` deux fois chacune. Elles appartiennent à des équipes, parties et dates différentes.

**Conclusion factuelle :** les quatre doublons de cardinalité ne prouvent pas que des joueurs reviennent. Ils sont beaucoup plus probablement causés par une collision issue de la conversion en notation scientifique. Les données ne permettent donc pas de mesurer directement la rétention ou le churn individuel.

## Discrétisation de la cible

La formule hypothétique

```text
placement_implicite = 1 + (1 - winRankPercentage) × (maxRank - 1)
```

donne une distance médiane à l'entier le plus proche de **0,0008 place**, un P95 de **0,0032**, et 100 % des lignes sont compatibles avec une cible arrondie à quatre décimales. Ce n'est pas une preuve absolue de la formule, mais une validation très forte : `winRankPercentage` encode vraisemblablement une place finale normalisée par `maxRank`.

**Impact :** la cible est mécaniquement discrétisée par la taille de partie. Les métriques d'erreur et les graphiques doivent tenir compte de cette granularité ; `maxRank` peut améliorer le modèle sans pour autant suffire à reconstruire la cible.

## Valeurs extrêmes

Les extrêmes les plus visibles dans le train sont : 35 kills (P99 = 7), 3 725 dégâts (P99 = 766,602), 31 290 de distance véhicule (P99 = 6 868,1), 11 260 de distance à pied (P99 = 4 395,02), 977,1 pour `highestKill` (P99 = 241,2) et 73 armes (P99 = 10).

La règle IQR signale parfois mécaniquement toute valeur positive comme aberrante lorsque Q1=Q3=0 : elle marque par exemple 17,282 % des `assists` et 16,772 % des `headshots`. Ces observations sont rares mais plausibles, pas automatiquement erronées. Pour les modèles et la segmentation, le traitement recommandé est `log1p`, clipping au P99 et indicateurs binaires, accompagné d'une analyse de sensibilité.

## Fuite de données potentielle

La décision dépend du cas d'usage, qui n'est pas fourni :

| Variable | Risque | Raison | Décision conditionnelle |
|---|---|---|---|
| `killRank` | Très élevé | Rang calculé à partir des kills de l'ensemble des joueurs, probablement final ou post-partie. | À exclure pour une prédiction en cours de partie ; acceptable uniquement pour expliquer a posteriori le résultat. |
| `winPts`, `rankPts`, `killPts` | Élevé | Scores historiques ou post-match dont le moment de disponibilité est inconnu ; ils codent aussi un régime de classement. | Vérifier le timestamp de calcul. Utiliser le régime séparément des valeurs. |
| `gameTime` | Élevé pour une prédiction précoce | Peut correspondre au temps de survie, donc à une conséquence directe du classement. | Exclure si la prédiction est faite avant la fin. |
| `maxRank`, `numTeams` | Faible à moyen | Contexte de partie connu au démarrage, mais impliqué dans la normalisation de la cible. | Conserver si connu au moment de scorer ; surveiller l'effet mécanique. |
| `teamId`, `gameId` | Très élevé comme valeurs brutes | Identifiants à haute cardinalité, susceptibles de mémoriser des groupes. | Ne pas encoder comme catégories ordinaires ; utiliser seulement pour splitter et agréger prudemment. |

## Limites

- Aucun dictionnaire métier ne permet de trancher les contradictions restantes.
- Les fichiers sont des échantillons incomplets de parties : on ne peut pas vérifier toutes les sommes au niveau match.
- La notion d'« aberrant » reste dépendante du mode de jeu ; les modes spéciaux peuvent légitimement produire des extrêmes.
- L'absence de vrais joueurs récurrents empêche d'évaluer la qualité d'une mesure de rétention/churn.
