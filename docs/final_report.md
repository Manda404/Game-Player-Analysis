# Rapport Data Science final

## Résumé exécutif

Le projet répond au test technique Gameloft par un notebook et un pipeline
Python reproductibles. Après réaudit de la sélection, le modèle publié est
**CatBoost avec ses paramètres d'apprentissage par défaut** et 16 variables
post-match. Il obtient 0,061448 ± 0,001095 de MAE sur cinq folds groupés du
développement puis **0,060797 de MAE**, 0,086596 de RMSE et 0,920827 de R² sur
le holdout groupé gelé du cycle d'audit.

Deux réserves dominent l'interprétation : `killRank` est calculé après le match
et apporte une grande part du signal ; la date officielle est incompatible avec
les groupes `gameId` observés. Le résultat est donc un benchmark post-match
robuste aux fuites de match, pas une validation early-game ou temporelle.

## Sujet, données et qualité

Une ligne contient les statistiques post-partie d'un joueur. La cible
`winRankPercentage` représente le classement normalisé de son équipe : 1 pour
la première place et 0 pour la dernière, selon `maxRank`. Le train contient
50 000 lignes attribuées à janvier–avril 2024 et le test 5 000 lignes attribuées
à mai. Aucun `gameId` brut ne relie train et test.

Les fichiers n'ont ni cellule manquante brute ni doublon exact. Les sentinelles
métier sont conservées avec indicateurs : `rankPts=-1`, puis `killPts=0` et
`winPts=0` lorsque le ranking est disponible. Le train contient 93 `gameId`
malformés ; une même valeur répétée reste dans un groupe conservateur commun.
Les anomalies gameplay sont signalées sans correction arbitraire : 53 lignes
ont des kills avec zéro dégât et 102 une activité de combat avec distance totale
nulle.

## Limites structurelles

Tous les `gameId` valides multi-lignes portent plusieurs dates, avec une étendue
médiane de 45,03 jours. La colonne ne permet donc pas une validation temporelle
fiable. Le train n'observe en moyenne que 1,61 joueur par `gameId` ; 98,82 % des
couples `(gameId, teamId)` sont singletons et seulement 2,34 % des lignes ont
un coéquipier observé. Les agrégats équipe/lobby sont rejetés car ils coderaient
surtout la couverture de l'extrait.

## Drift train/test

La version refactorée ne mesurait que des différences standardisées de moyenne.
L'audit rétablit PSI, KS, Wasserstein normalisée, masse à zéro, changements de
catégories et validation adversariale.

| Diagnostic | Maximum / résultat | Lecture |
|---|---:|---|
| PSI numérique | 0,00509 (`damages`) | faible |
| KS | 0,01312 (`damages`) | faible ; p=0,411 descriptive |
| différence moyenne standardisée | 0,02347 (`damages`) | faible |
| PSI catégoriel | 0,00844 (`gameType`) | faible |
| déplacement de modalité | -0,622 point (`legacy`) | faible |
| adversarial ROC AUC | 0,49325 ± 0,00542 | aucune séparation linéaire utile |

Aucun drift matériel n'est détecté sur le contrat mesuré. Cette conclusion ne
signifie pas distributions identiques : le drift de performance/concept est
inobservable sans cible test, et le drift temporel est indécidable avec la date
incohérente.

## Analyse, features et ablation

La cible a une moyenne de 0,47233, une médiane de 0,45785 et un écart-type de
0,30745. Les corrélations de Spearman les plus fortes sont `walkDist` (0,866),
`killRank` (-0,715), `upgrades` (0,681), `weapons` (0,665) et `heals` (0,564).

Le contrat comportemental contient 15 variables : mesures brutes, features
dérivées et indicateurs de mode. Le contrat post-match ajoute `killRank`.

| Étape | Features | MAE groupée |
|---|---:|---:|
| contexte | 4 | 0,267714 |
| + mobilité | 7 | 0,098414 |
| + combat | 11 | 0,095515 |
| + ressources | 15 | 0,092661 |
| + `killRank` post-match | 16 | 0,061500 |

La mobilité fournit l'essentiel du signal comportemental. Combat et ressources
apportent chacun environ 0,0029 MAE ; `killRank` apporte 0,03116 et confirme le
changement de scénario. La projection sur la grille `maxRank` améliore la MAE
de 0,061448 à 0,060982 mais dégrade légèrement le RMSE de 0,086477 à 0,086882.

## Validation

Un holdout groupé est gelé avant le cycle de sélection. Les 40 128 lignes de
développement alimentent seules la comparaison, l'ablation et le tuning dans
cinq folds `GroupKFold`. Chaque partition a zéro groupe prudent et zéro
`gameId` brut partagés. Le test officiel n'entre jamais dans la sélection.

Les diagnostics sur le train complet donnent 0,060949 de MAE pour un split
aléatoire ligne, 0,061160 pour un split groupé, 0,060944 pour janvier–mars vers
avril naïf et 0,061277 après purge des matchs. Ce dernier est un stress test
avec moins de données, pas une preuve temporelle.

Le nombre de folds a lui aussi été audité. Pour K=3, 5, 7 et 10, CatBoost bat
XGBoost dans les 25 folds cumulés. Sept folds améliore nominalement la MAE de
0,000189 par rapport à cinq, mais l'écart reste inférieur à l'incertitude des
folds tandis que le coût de calcul augmente sensiblement. Dix folds augmente
encore la charge et la dispersion. Le pipeline conserve donc cinq folds, avec
environ 8 026 lignes indépendantes par validation.

Le holdout final contient 9 872 lignes, zéro match partagé et donne 0,060797 de
MAE. Il est indépendant des décisions du présent audit mais pas historiquement
vierge : l'EDA antérieure avait observé toutes les lignes étiquetées.

## Comparaison, réconciliation et tuning

| Rang | Modèle initial par défaut | MAE ± écart-type | MAE train |
|---:|---|---:|---:|
| 1 | CatBoost | **0,061448 ± 0,001095** | 0,055269 |
| 2 | XGBoost | 0,063448 ± 0,001073 | 0,050440 |
| 3 | LightGBM | 0,063593 ± 0,000912 | 0,059488 |
| 4 | Random Forest | 0,066677 ± 0,000771 | 0,024722 |
| 5 | Ridge | 0,088310 ± 0,001395 | 0,088256 |
| 6 | Dummy médiane | 0,267877 ± 0,000822 | 0,267862 |

CatBoost bat XGBoost dans les cinq folds, de 0,0019999 MAE en moyenne ;
l'intervalle bootstrap descriptif de la différence XGBoost moins CatBoost vaut
[0,001829 ; 0,002192].

L'ancien gagnant XGBoost venait d'une comparaison pré-réglée : 500 arbres,
profondeur 6, learning rate 0,05 et subsampling 0,85, tandis que CatBoost avait
une autre configuration personnalisée. Rejouées sur le même protocole, ces
configurations donnent 0,061999 pour XGBoost et 0,062053 pour CatBoost, soit un
écart négligeable de 0,000054. La conclusion XGBoost n'était donc pas robuste.

Seul CatBoost, choisi avant optimisation, passe dans un `RandomizedSearchCV` de
huit essais sur les mêmes folds. Le meilleur essai donne 0,061671, moins bien
que le défaut de 0,000223. Le tuning est rejeté selon le seuil de gain minimal
de 0,0001 fixé avant le holdout. Aucun early stopping n'est mélangé à la CV.

## Interprétabilité, erreurs et publication

L'importance par permutation sur holdout et les valeurs TreeSHAP natives de
CatBoost convergent : `killRank` domine, devant la marche par minute, `kills`
et `maxRank`. Les 2 000 explications SHAP, tirées de manière déterministe des
9 872 lignes du holdout, reconstruisent chacune la prédiction brute du modèle.
Ces importances restent prédictives et non causales. Les erreurs sont plus
fortes sur les modes spéciaux et les petites grilles, dont le support est
faible. Les diagnostics globaux, locaux et par sous-groupes sont exportés sous
`artifacts/metrics/`.

Le bundle final contient CatBoost, les 16 features ordonnées, paramètres,
versions, seeds, empreintes des données, métriques et SHA-256. L'inférence
reconstruit les features, refuse les valeurs non finies, borne les prédictions
et conserve l'ordre des lignes.

## Conclusion et données nécessaires

Le protocole final suit Dummy → Ridge → candidats par défaut → sélection
CatBoost → tuning rejeté → holdout groupé → refit complet → inférence test. Une
version produit nécessite encore une date cohérente, un futur holdout étiqueté
jamais observé, les rosters complets, une identité longitudinale stable et des
variables disponibles au moment réel de la décision.
