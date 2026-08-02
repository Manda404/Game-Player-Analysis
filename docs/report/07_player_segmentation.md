# 07 — Segmentation des joueurs

## Objectif et méthode

**Question.** Existe-t-il des profils comportementaux distincts qui restent utiles au-delà du seul mode de jeu et qui sont associés à des niveaux de performance différents ?

La segmentation utilise K-means sur 13 variables comportementales : assists, upgrades, dégâts, knocks, headshots, soins, kills, revives, trois distances, armes et durée de partie. La cible, les identifiants, les rangs et les scores de classement sont exclus.

Prétraitement : clipping au P99, transformation `log1p`, standardisation. Le nombre de clusters est choisi par le meilleur silhouette entre k=3 et k=6 sur un échantillon reproductible de 5 000 lignes.

| k | Silhouette | Inertie |
|---:|---:|---:|
| 3 | **0,1943** | 440 621 |
| 4 | 0,1544 | 403 408 |
| 5 | 0,1708 | 368 614 |
| 6 | 0,1396 | 345 989 |

Le k retenu est **3**. Le silhouette de 0,194 signifie que les frontières sont souples et que les comportements forment plutôt un continuum. En revanche, la solution est très reproductible entre cinq autres initialisations : ARI moyen **0,9905**, minimum **0,9819**.

Les libellés métier ci-dessous sont attribués après calcul ; les identifiants 0/1/2 sont arbitraires.

## Vue d'ensemble

| ID | Libellé métier | Taille | Part | Cible moyenne | Médiane | Cible ≥0,9 | Cible =1 | Cible =0 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Combattants très actifs | 10 671 | 21,342 % | **0,7315** | 0,7912 | **30,372 %** | **10,027 %** | 0,197 % |
| 1 | Faible activité / sortie précoce probable | 24 626 | 49,252 % | **0,2528** | 0,2083 | **0,788 %** | 0,219 % | **9,823 %** |
| 2 | Mobiles orientés survie et collecte | 14 703 | 29,406 % | **0,6518** | 0,6667 | **12,338 %** | 2,081 % | 0,068 % |

La cible n'a pas été utilisée pour former ces groupes, mais l'appartenance explique **50,30 %** de sa variance (`η²=0,503`; Kruskal-Wallis `p<10⁻³⁰⁰`). Cette association très forte est cohérente avec le fait que les features accumulées reflètent aussi la durée de survie. Elle ne prouve pas que transformer un joueur d'un segment en un autre causerait mécaniquement le même gain.

![Profil comportemental des segments](../reports/independent_raw_analysis/figures/segment_profile_heatmap.png)

## Segment 0 — Combattants très actifs

**Taille :** 10 671 observations, 21,342 %.

| Variable | Moyenne segment | Indice vs moyenne globale |
|---|---:|---:|
| Kills | 3,012 | 327 |
| Dégâts | 350,621 | 270 |
| Knocks | 2,007 | 305 |
| Headshots | 0,895 | 397 |
| Assists | 0,659 | 285 |
| Revives | 0,495 | 299 |
| Upgrades | 2,785 | 252 |
| Soins | 3,305 | 240 |
| Distance à pied | 2 098,644 | 182 |
| Armes | 5,118 | 140 |

Ce segment combine combat, soutien, collecte et mobilité, plutôt qu'un pur style agressif immobile. Il obtient les meilleurs résultats : rang moyen 0,7315 et 30,37 % dans la zone ≥0,9.

**Intérêt métier.** Joueurs candidats aux modes compétitifs, défis avancés et événements de maîtrise.

**Actions possibles.** Contenu classé exigeant, objectifs de précision/équipe, récompenses cosmétiques de maîtrise, surveillance anti-triche sur les extrêmes plutôt que sur le segment entier.

**Risque.** Les statistiques de combat élevées peuvent refléter un niveau intrinsèque ou une survie plus longue. Ne pas récompenser uniquement les kills au détriment du positionnement.

## Segment 1 — Faible activité / sortie précoce probable

**Taille :** 24 626 observations, 49,252 %.

| Variable | Moyenne segment | Indice vs moyenne globale |
|---|---:|---:|
| Kills | 0,267 | 29 |
| Dégâts | 53,925 | 41 |
| Upgrades | 0,082 | 7 |
| Soins | 0,129 | 9 |
| Distance véhicule | 21,548 | 4 |
| Distance à pied | 326,106 | 28 |
| Armes | 2,199 | 60 |
| Durée de partie | 1 535,690 | 97 |

La durée de partie restant proche de la moyenne confirme qu'elle n'est pas le temps individuel de survie. La combinaison faible distance/faible loot/faible combat suggère néanmoins une élimination précoce ou une très faible participation. La cible moyenne est 0,2528 et 9,823 % finissent à 0.

**Intérêt métier.** Principal réservoir d'amélioration de l'activation et de l'onboarding.

**Actions possibles.** Tutoriels contextuels, missions de premiers déplacements/loot, conseils de zone d'atterrissage, matchmaking d'apprentissage, diagnostic des abandons et problèmes techniques.

**Risque.** Sans ancienneté ni historique, on ne sait pas distinguer débutants, joueurs occasionnels, déconnexions et mauvais départs. Les actions doivent être testées, pas imposées à tout le segment.

## Segment 2 — Mobiles orientés survie et collecte

**Taille :** 14 703 observations, 29,406 %.

| Variable | Moyenne segment | Indice vs moyenne globale |
|---|---:|---:|
| Kills | 0,499 | 54 |
| Dégâts | 97,305 | 75 |
| Headshots | 0,036 | 16 |
| Upgrades | 1,598 | 145 |
| Soins | 2,068 | 150 |
| Distance véhicule | 1 334,173 | 222 |
| Distance nage | 9,339 | 212 |
| Distance à pied | 1 859,197 | 161 |
| Armes | 5,063 | 138 |

Ce segment réalise moins de combat que la moyenne mais se déplace, collecte et se soigne beaucoup. Sa cible moyenne de 0,6518 montre qu'une stratégie orientée positionnement/survie est fortement associée à la réussite, même sans volume de kills élevé.

**Intérêt métier.** Joueurs stratégiques sensibles aux objectifs de carte, à l'exploration, aux véhicules et à la survie.

**Actions possibles.** Défis de positionnement/mobilité, objectifs d'exploration, contenu véhicule, récompenses de survie, entraînement combat optionnel pour convertir davantage de bonnes positions en victoires.

**Risque.** Ne pas qualifier ce profil de « passif » sans données sur la stratégie ou le rythme réel ; il atteint déjà un rang élevé.

## Robustesse par mode

L'association segment/mode est faible (`V de Cramér=0,0956`). Les segments ne sont donc pas simplement solo/duo/squad. L'ordre des cibles reste le même dans chacun des six modes principaux :

| Mode | Combattants actifs | Faible activité | Mobiles/collecte |
|---|---:|---:|---:|
| `duo` | 0,7497 | 0,2418 | 0,6455 |
| `duo-fpp` | 0,7526 | 0,2722 | 0,6763 |
| `solo` | 0,8285 | 0,2634 | 0,6951 |
| `solo-fpp` | 0,8174 | 0,3004 | 0,7210 |
| `squad` | 0,6936 | 0,2131 | 0,6056 |
| `squad-fpp` | 0,7167 | 0,2374 | 0,6360 |

Cette robustesse renforce l'utilité descriptive des profils. Les différences absolues entre modes rappellent toutefois qu'un benchmark de segment doit être ajusté au mode.

![Distribution de la cible par segment](../reports/independent_raw_analysis/figures/segment_target_boxplot.png)

## Limites et validation future

1. Le clustering décrit des **participations**, pas des personnes persistantes ; un même vrai joueur pourrait changer de segment d'une partie à l'autre.
2. Le silhouette faible interdit de traiter les labels comme une vérité naturelle.
3. Les features sont accumulées pendant toute la partie et intègrent une conséquence du rang/survie.
4. Les valeurs extrêmes sont clippées pour la segmentation, mais restent disponibles dans les données sources et tables descriptives.
5. Il faut valider la stabilité sur d'autres mois, jeux ou régions, puis relier les segments à la rétention et à la monétisation lorsque ces données seront disponibles.
