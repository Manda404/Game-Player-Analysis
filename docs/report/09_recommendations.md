# 09 — Recommandations

## Feuille de route priorisée

### P0 — À faire avant toute mise en production

| # | Recommandation | Justification chiffrée | Livrable / critère de succès |
|---:|---|---|---|
| 1 | Définir le cas d'usage et l'instant exact de prédiction | `killRank`, distance, loot et scores peuvent être postérieurs à l'instant de décision | Contrat écrit « données disponibles à T » ; deux listes de features : précoce et post-partie |
| 2 | Corriger l'ingestion des identifiants en chaîne | Jusqu'à 0,28 % d'IDs mal formés ; toutes les répétitions joueur observées sont des collisions | Zéro nouvel identifiant ne respectant pas le regex source ; récupération des hashes originaux si possible |
| 3 | Formaliser les sentinelles et le régime de score | 38,382 % `rankPts=-1`, 59,680 % `killPts=winPts=0` | Dictionnaire signé ; feature `ranking_system`; valeurs indisponibles stockées comme nulls, pas comme scores |
| 4 | Imposer une validation temporelle groupée par `gameId` | 32 510 lignes appartiennent à des `gameId` répétés ; 99,316 % des équipes répétées partagent la cible | Aucun `gameId` valide entre train et validation ; holdout futur ; résultats par mode/segment |
| 5 | Construire un baseline sans fuite | `killRank` est fortement lié à la cible mais probablement final | Comparer médiane, modèle linéaire régularisé et boosting sur deux feature sets ; rapport MAE/RMSE et stabilité temporelle |

### P1 — Améliorer l'analyse et le produit

| # | Recommandation | Pourquoi | Critère de succès |
|---:|---|---|---|
| 6 | Déployer un tableau de bord KPI par mode et mois | Les KPI de soutien et structure sont fortement dépendants du mode | Rang moyen, top-rank, médiane/P90 distance, dégâts, kills, soutien et taux d'anomalie, avec alertes de dérive |
| 7 | Tester un parcours d'activation pour le segment faible activité | 49,252 % des lignes, cible moyenne 0,2528, 9,823 % à zéro | A/B test : hausse des premières actions, distance, loot et rang, sans dégradation de rétention future |
| 8 | Servir du contenu différencié aux deux profils performants | Mobiles/collecte 29,406 %, combattants actifs 21,342 %, stratégies distinctes | Expériences séparées : objectifs mobilité/survie vs défis maîtrise/combat ; mesure d'engagement et de satisfaction |
| 9 | Auditer les anomalies de télémétrie/comportement | 102 combats sans déplacement, 53 kills sans dégâts, 5 cas <10 m avec 5+ kills | Taxonomie confirmée ; taux de faux positifs ; règles qualité ou anti-triche validées par experts |
| 10 | Mesurer les effets causalement | Les plus fortes features sont accumulées pendant la survie | Tests randomisés ou quasi-expériences ; ne promouvoir une règle produit qu'après effet incrémental mesuré |

### P2 — Enrichir la donnée

| # | Recommandation | Données à ajouter | Valeur attendue |
|---:|---|---|---|
| 11 | Créer une identité joueur stable et gouvernée | `account_id` non transformé, mapping privacy-safe, historique | Rétention, fréquence, churn, transitions de segments |
| 12 | Ajouter une table de sessions et événements horodatés | session start/end, join, death, quit, actions, snapshots | Temps de survie réel, comportement avant abandon, prédiction précoce |
| 13 | Ajouter progression et contexte | niveau, XP, ancienneté, plateforme, région, acquisition, version | Distinguer novice, difficulté, performance technique et cohortes |
| 14 | Ajouter transactions et exposition aux offres | achat, prix, devise, item, impression/clic, remboursement | Conversion, ARPU, ARPPU, LTV et monétisation par segment |
| 15 | Conserver la composition complète des parties | Tous les joueurs/équipes d'un match et résultat exact | Agrégats adversaires, rang relatif, contrôles de cohérence match-level |

## Feature engineering recommandé

### Features comportementales robustes

| Feature | Formule | Objectif | Précaution |
|---|---|---|---|
| `totalDistance` | `walkDist + rideDist + swimDist` | Mobilité totale | Très corrélée au résultat et à la survie |
| `usedVehicle`, `swam` | `rideDist>0`, `swimDist>0` | Séparer occurrence et intensité zero-inflated | Conserver aussi `log1p(distance)` |
| Parts de mobilité | composante / `totalDistance` | Style de déplacement | Guard si distance totale nulle |
| `supportActions` | `assists + revives` | Coopération | Mettre non applicable/ajusté en solo |
| `headshotRate` | `headshots / kills` | Précision | Définir 0/NA si aucun kill |
| `damagePerKill` | `damages / max(kills,1)` + flag zéro kill | Efficacité/engagement de combat | Valeurs extrêmes et kills sans dégâts |
| `lootIntensity` | combinaison standardisée armes/upgrades/soins | Progression/collecte | Éviter de compter trois fois la survie |
| `ranking_system` | disponibilité des trois scores | Corriger les sentinelles | Ne pas interpréter comme niveau joueur |
| `teamSizeProxy` | `maxRank / numTeams` | Contextualiser solo/duo/squad | `gameType` reste plus explicite |
| Famille/perspective | extraire solo/duo/squad et FPP/TPP | Réduire la rareté catégorielle | Conserver un flag mode spécial |
| Interactions | mobilité × combat, mode × soutien | Capturer des stratégies | Valider hors temps et expliquer avec effets partiels |

Toutes les variables à longue queue devraient être testées avec `log1p` et clipping appris uniquement sur le train. Les identifiants bruts ne sont pas des features.

## Stratégie de modélisation proposée

1. **Split :** janvier-mars pour entraînement, avril pour validation de développement, mai pour test final si la cible est ensuite disponible. Groupement par `gameId` à chaque frontière possible.
2. **Deux scénarios :**
   - modèle « early game » sans `killRank`, résultat final, scores post-match ni agrégats futurs ;
   - modèle « explicatif post-partie » pouvant inclure les informations finales, clairement étiqueté comme non prédictif en temps réel.
3. **Baselines :** prédiction médiane globale puis médiane par mode ; Ridge/Elastic Net après transformations ; gradient boosting non linéaire.
4. **Métriques :** MAE et RMSE globales, par mois, mode, segment et zones `target=0`, `(0,0.9)`, `>=0.9`, `=1`.
5. **Interprétation :** permutation importance ou SHAP sur holdout, importance groupée pour les familles colinéaires, courbes partielles avec avertissement non causal.
6. **Robustesse :** ablation des features à risque, résultats avec/sans anomalies, test de calibration et clipping `[0,1]`.

## Actions produit proposées comme hypothèses de test

### Segment faible activité

- Hypothèse : aider à réaliser les premières actions (se déplacer, trouver une arme, comprendre la zone) augmente l'activation dans la partie.
- Test : tutoriel contextuel ou mission initiale randomisée.
- Mesures primaires : taux d'arme acquise, distance médiane, premier dégât/kill, rang normalisé.
- Garde-fous : frustration, abandon, performance technique, équilibre de matchmaking.

### Segment mobile/collecte

- Hypothèse : objectifs de positionnement, exploration ou véhicules renforcent l'engagement sans forcer un style combat.
- Test : missions de carte/survie et entraînement combat optionnel.
- Mesures : complétion, retour ultérieur lorsque disponible, conversion de bonne position en top-rank/victoire.

### Segment combattant actif

- Hypothèse : défis de maîtrise et modes compétitifs améliorent la profondeur d'engagement.
- Test : objectifs précision/équipe et ladder contrôlé.
- Mesures : participation, complétion, équilibre, diversité des stratégies et signalements anti-triche.

## Monitoring recommandé

| Domaine | Indicateur | Seuil initial d'alerte |
|---|---|---|
| Schéma | colonnes/types manquants | Toute variation non versionnée |
| Identifiants | part non conforme | >0 % sur nouvelles données après correction |
| Sentinelles | mix de `ranking_system` | Variation >5 points ou apparition d'un régime inconnu |
| Dérive | PSI feature | 0,10 vigilance ; 0,25 investigation urgente |
| Performance | MAE/RMSE par mode/segment | Dégradation >10 % vs référence |
| Qualité | distance nulle avec combat | Dépassement du baseline 0,204 % train |
| Segments | part de chaque profil | Variation >5 points à composition de modes constante |

Les seuils doivent être recalibrés après plusieurs périodes ; ils servent de point de départ, pas de vérité universelle.

## Ordre d'exécution recommandé

1. Semaine 1 : dictionnaire métier, cas d'usage, audit source des identifiants et sentinelles.
2. Semaines 2–3 : pipeline de préparation, split temporel groupé, baselines sans fuite et dashboard qualité.
3. Semaines 3–5 : instrumentation identité/session/progression et définition des KPI longitudinaux.
4. Ensuite : A/B tests sur le segment faible activité, puis personnalisation des deux segments performants.
