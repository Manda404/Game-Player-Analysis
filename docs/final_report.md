# Rapport Data Science final

## Résumé exécutif

Le projet répond au test technique Gameloft par un notebook exécuté et un
pipeline Python unique. Le modèle publié est XGBoost avec 16 variables
post-match. Sur cinq folds groupés par `gameId`, il atteint une MAE de
**0,06165 ± 0,00063**, un RMSE de 0,08679 et un R² moyen de 0,92030.

Cette performance doit être lue avec deux réserves majeures : `killRank` est
calculé après le match et explique la plus forte part du signal ; la date
officielle est incompatible avec les groupes `gameId` observés. Le résultat est
donc un benchmark post-match robuste aux fuites de match, pas une validation
early-game ni temporelle.

Les résultats complets sont dans
[`artifacts/metrics/`](../artifacts/metrics/) et les preuves visuelles dans
[`artifacts/figures/`](../artifacts/figures/).

## Sujet et unité statistique

Une ligne contient les statistiques post-partie d'un joueur. La cible
`winRankPercentage` représente le classement normalisé de son équipe : 1 pour
la première place et 0 pour la dernière, selon `maxRank`. Une équipe peut donc
partager la même cible entre plusieurs joueurs, même si le fichier n'observe
qu'une petite partie des rosters.

Le train contient 50 000 lignes attribuées à janvier–avril 2024 ; le test en
contient 5 000 attribuées à mai. Aucun `gameId` brut ne relie train et test.

## Qualité et nettoyage

Les deux fichiers n'ont ni cellule manquante brute ni doublon exact. Les valeurs
de ranking contiennent cependant des sentinelles documentées :

- `rankPts=-1` signifie indisponible ;
- lorsque `rankPts` est disponible, `killPts=0` et `winPts=0` signifient
  indisponible ;
- des drapeaux de disponibilité sont créés et aucune ligne n'est supprimée.

Le train contient 93 `gameId` au format corrompu ou transformé. Une même valeur
malformée répétée reste dans un groupe conservateur commun ; elle ne peut plus
traverser l'entraînement et la validation. Les anomalies gameplay sont
signalées sans correction arbitraire : 53 lignes ont des kills avec zéro dégât
et 102 une activité de combat avec distance totale nulle.

## Limites structurelles découvertes

### Date incohérente

Parmi les identifiants valides, 13 464 matchs train ont plusieurs lignes. Tous
portent plusieurs dates, avec une étendue médiane de 45,03 jours, un 95e
percentile de 99,53 jours et un maximum de 119,30 jours. La cause n'est pas
observable : le projet qualifie `date` d'incohérente/corrompue, sans supposer un
export ou une anonymisation particuliers.

### Couverture partielle des équipes et matchs

Le train n'observe en moyenne que 1,61 joueur par `gameId`, avec huit au maximum.
Parmi 49 411 couples `(gameId, teamId)`, 98,82 % sont des singletons et seulement
2,34 % des lignes ont un coéquipier observé. Les moyennes/sommes équipe ou lobby
ont donc été rejetées : elles coderaient surtout la couverture du fichier.

## Analyse et KPI

La cible a une moyenne de 0,47233, une médiane de 0,45785 et un écart-type de
0,30745. Les corrélations de Spearman les plus fortes sont `walkDist` (0,866),
`killRank` (-0,715), `upgrades` (0,681), `weapons` (0,665) et `heals` (0,564).
Les profils par quantiles confirment des relations non linéaires.

Sept KPI ont été évalués avec des règles explicites pour les divisions par zéro.
La distance totale, la mobilité/seconde et le headshot ratio restent analytiques
car trop redondants, mal alignés sur un temps de survie ou trop clairsemés. Les
indicateurs combat, ressources et dégâts/kill passent par l'ablation.

## Features et ablation

Le contrat comportemental contient 15 variables : huit mesures brutes, quatre
features dérivées et trois indicateurs de mode. Le contrat post-match ajoute
uniquement `killRank`.

| Étape | Features | MAE |
|---|---:|---:|
| Contexte | 4 | 0,26779 |
| + mobilité | 7 | 0,09907 |
| + combat | 11 | 0,09616 |
| + ressources | 15 | 0,09326 |
| + `killRank` post-match | 16 | 0,06160 |

L'essentiel du signal comportemental vient de la mobilité. Combat et ressources
apportent chacun environ 0,0029 de MAE. `killRank` apporte 0,03166, confirmant le
changement de scénario. La projection sur la grille `maxRank` améliore la MAE
de 0,06165 à 0,06121, mais dégrade le RMSE de 0,08680 à 0,08751 ; elle est
appliquée à la soumission, pas présentée comme un gain universel.

## Validation

Le protocole principal est un GroupKFold à cinq folds. Chaque fold contient
40 000 lignes d'entraînement et 10 000 de validation, avec zéro groupe prudent
et zéro valeur brute de `gameId` partagés.

| Holdout | Matchs déjà vus en validation | MAE |
|---|---:|---:|
| Aléatoire ligne | 56,64 % | 0,06169 |
| Groupé `gameId` | 0 % | 0,06168 |
| Jan–mars → avril naïf | 54,97 % | 0,06169 |
| Jan–mars → avril purgé | 0 % | 0,06222 |

Le test purgé retire 8 536 lignes antérieures dont le match réapparaît en avril.
Il constitue un stress test de distribution, pas une preuve chronologique.

## Modèles, tuning et surajustement

| Rang | Modèle | MAE | MAE train | Diagnostic |
|---:|---|---:|---:|---|
| 1 | XGBoost | **0,06165** | 0,05261 | écart modéré |
| 2 | LightGBM | 0,06168 | 0,05531 | écart modéré |
| 3 | CatBoost | 0,06170 | 0,05886 | faible écart |
| 4 | XGBoost tuned | 0,06177 | 0,05409 | écart modéré |
| 5 | Random Forest | 0,06542 | 0,03033 | écart élevé |
| 6 | Ridge | 0,08823 | 0,08817 | faible écart |

Les trois premiers sont séparés par moins d'un écart-type de fold. XGBoost est
retenu pour la MAE et la simplicité de publication, sans revendication de
supériorité robuste. Un bootstrap descriptif des différences de MAE appariées
par fold est exporté dans `model_fold_uncertainty.csv`; avec seulement cinq
folds, il documente l'incertitude et n'est pas présenté comme un test
d'hypothèse. Six essais aléatoires bornés testent profondeur,
régularisation, sous-échantillonnage et taux d'apprentissage. Le meilleur essai
est moins bon de 0,00012 ; il est rejeté selon un seuil de gain minimal fixé à
0,0001. Random Forest montre l'écart train-validation le plus préoccupant.

## Interprétabilité et erreurs

Sur un holdout groupé indépendant, permuter `killRank` augmente la MAE de 0,266,
loin devant la mobilité/minute (0,087), les kills (0,075) et `maxRank` (0,054).
Cette importance est prédictive, non causale.

Les erreurs par famille de mode valent 0,047 en solo, 0,053 en duo, 0,071 en
squad et 0,100 sur les modes spéciaux. Les petites grilles de rang sont plus
difficiles, mais leurs effectifs sont faibles. Les résidus montrent une
régression vers la moyenne aux cibles hautes. Les 25 plus grandes erreurs OOF
sont exportées dans `artifacts/metrics/largest_errors.csv`.

## Publication et inférence

Le bundle contient le modèle, les 16 features ordonnées, les paramètres, les
versions, la seed, les empreintes des données, les métriques et le SHA-256 du
modèle. L'inférence valide le CSV officiel, reconstruit les features dans le
même ordre, refuse les valeurs non finies, borne les prédictions et conserve
l'ordre des lignes dans la soumission.

## Limites et prochaines données nécessaires

- aucune cible sur le test officiel ;
- aucun roster complet ni match intégral ;
- aucune date de match exploitable ;
- pas de snapshots disponibles au moment d'une décision early-game ;
- faible support des modes spéciaux et petites grilles ;
- aucune conclusion causale sur les comportements.

Une version produit nécessite une date cohérente, un holdout futur étiqueté,
les rosters complets, une identité longitudinale stable et des variables
horodatées au moment réel de la prédiction.
